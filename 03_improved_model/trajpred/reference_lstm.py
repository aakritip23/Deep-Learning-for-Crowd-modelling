"""Re-implementation of the reference models (Social LSTM, Vanilla LSTM) inside our harness.

Why: the professor's repo ships no trained weights and needs ~6 s/batch on CPU, so it cannot be re-scored
here.  This version follows Alahi et al. (CVPR'16) and the professor's configuration (rnn 128, embedding 64,
4x4 social grid, bivariate-Gaussian output trained with NLL, teacher forcing, positions relative to the
first observed point) but is trained and scored with exactly the same data, split and metrics as the
Transformer models.  Prediction = Gaussian mean, rolled out autoregressively for 12 steps.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import SOURCES, T_ALL, T_OBS, T_PRED


# ----------------------------------------------------------------------------- scene batches
class SceneDataset:
    def __init__(self, scenes):
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]


def scene_collate(scenes):
    """Pad scenes to the largest pedestrian count.  pos [B, 20, P, 2] absolute metres (NaN future -> 0)."""
    B, P = len(scenes), max(len(s.ids) for s in scenes)
    pos = np.zeros((B, T_ALL, P, 2), dtype=np.float32)
    mask = np.zeros((B, P), dtype=bool)
    src = np.full((B, P), -1, dtype=np.int64)
    known = np.zeros((B, T_ALL, P), dtype=bool)
    for b, s in enumerate(scenes):
        n = len(s.ids)
        xy = np.transpose(s.xy, (1, 0, 2))  # [20, n, 2]
        ok = np.isfinite(xy).all(-1)
        pos[b, :, :n] = np.where(ok[..., None], xy, 0.0)
        known[b, :, :n] = ok
        mask[b, :n] = True
        src[b, :n] = SOURCES.index(s.source) if s.source in SOURCES else -1
    t = torch.from_numpy
    return {"pos": t(pos), "mask": t(mask), "src": t(src), "known": t(known)}


# ----------------------------------------------------------------------------- model
def gaussian_nll(out, target, valid, eps=1e-6):
    """Bivariate Gaussian negative log-likelihood. out [..., 5] (mux, muy, log sx, log sy, atanh rho)."""
    mux, muy = out[..., 0], out[..., 1]
    sx, sy = torch.exp(out[..., 2]).clamp(min=1e-3), torch.exp(out[..., 3]).clamp(min=1e-3)
    rho = torch.tanh(out[..., 4]).clamp(-0.999, 0.999)
    dx, dy = (target[..., 0] - mux) / sx, (target[..., 1] - muy) / sy
    z = dx ** 2 + dy ** 2 - 2 * rho * dx * dy
    nll = z / (2 * (1 - rho ** 2)) + torch.log(sx) + torch.log(sy) + 0.5 * torch.log(1 - rho ** 2) + math.log(2 * math.pi)
    return (nll * valid).sum() / valid.sum().clamp(min=1)


class ReferenceLSTM(nn.Module):
    """social=True -> Social LSTM (grid pooling of neighbours' hidden states); False -> Vanilla LSTM."""

    def __init__(self, social=True, rnn=128, emb=64, grid=4, nbr_size=4.0, dropout=0.5, pos_scale=5.0,
                 mode="abs", vel_scale=0.25):
        """mode 'abs': input/predict positions relative to the first observed point (original formulation).
        mode 'vel': input/predict per-step displacements (velocities, standardised by vel_scale) and integrate -
        the usual refinement; the rollout no longer has to reproduce its own absolute input."""
        super().__init__()
        self.social, self.rnn, self.grid, self.nbr_size, self.pos_scale = social, rnn, grid, nbr_size, pos_scale
        self.mode, self.vel_scale = mode, vel_scale
        self.emb_in = nn.Linear(2, emb)
        self.emb_soc = nn.Linear(grid * grid * rnn, emb) if social else None
        self.cell = nn.LSTMCell(2 * emb if social else emb, rnn)
        self.out = nn.Linear(rnn, 5)
        self.drop = nn.Dropout(dropout)

    def pool(self, abs_t, mask, h):
        """abs_t [B,P,2] absolute positions, h [B,P,H] previous hidden states -> social embedding [B,P,emb]."""
        B, P, _ = abs_t.shape
        d = abs_t[:, None, :, :] - abs_t[:, :, None, :]  # [B, i, j, 2]: neighbour j seen from i
        half = self.nbr_size / 2
        inside = (d.abs() < half).all(-1) & mask[:, None, :] & mask[:, :, None]
        inside &= ~torch.eye(P, dtype=torch.bool, device=d.device)[None]
        cell = torch.floor((d + half) / (self.nbr_size / self.grid)).clamp(0, self.grid - 1).long()
        occ = F.one_hot(cell[..., 0] + cell[..., 1] * self.grid, self.grid ** 2).float() * inside.unsqueeze(-1)
        pooled = torch.einsum("bijg,bjh->bigh", occ, h)  # [B, P, G*G, H]
        return self.drop(F.relu(self.emb_soc(pooled.flatten(2))))

    def step(self, rel, abs_t, mask, state):
        B, P, _ = rel.shape
        h, c = state
        x = self.drop(F.relu(self.emb_in(rel)))
        if self.social:
            x = torch.cat([x, self.pool(abs_t, mask, h)], dim=-1)
        h, c = self.cell(x.reshape(B * P, -1), (h.reshape(B * P, -1), c.reshape(B * P, -1)))
        h, c = h.view(B, P, -1), c.view(B, P, -1)
        return self.out(h), (h, c)

    def init_state(self, B, P, device):
        z = torch.zeros(B, P, self.rnn, device=device)
        return z, z.clone()

    def forward_train(self, pos, mask, known):
        """Teacher forcing over all 19 transitions.  Returns NLL (scaled units)."""
        if self.mode == "vel":
            return self._forward_train_vel(pos, mask, known)
        B, T, P, _ = pos.shape
        x0 = pos[:, 0:1]
        rel = (pos - x0) / self.pos_scale
        state, loss_terms, outs = self.init_state(B, P, pos.device), [], []
        for t in range(T - 1):
            out, state = self.step(rel[:, t], pos[:, t], mask, state)
            outs.append(out)
        out = torch.stack(outs, 1)  # [B, 19, P, 5]
        valid = (mask[:, None, :] & known[:, 1:] & known[:, :-1]).float()
        return gaussian_nll(out, rel[:, 1:], valid)

    @torch.no_grad()
    def predict(self, pos, mask):
        """Observe 8 steps (teacher forced), then roll out 12 steps with the predicted mean. -> [B, 12, P, 2] metres."""
        if self.mode == "vel":
            return self._predict_vel(pos, mask)
        B, T, P, _ = pos.shape
        x0 = pos[:, 0]
        state = self.init_state(B, P, pos.device)
        rel_in = (pos[:, 0] - x0) / self.pos_scale
        preds = []
        for t in range(T_ALL - 1):
            if t >= T_OBS:  # feed back own prediction
                rel_in = preds[-1]
            else:
                rel_in = (pos[:, t] - x0) / self.pos_scale
            abs_in = x0 + rel_in * self.pos_scale
            out, state = self.step(rel_in, abs_in, mask, state)
            if t >= T_OBS - 1:
                preds.append(out[..., :2])
        rel_pred = torch.stack(preds, 1)  # [B, 12, P, 2]
        return x0[:, None] + rel_pred * self.pos_scale


    # ------------------------------------------------------------------ velocity formulation
    def _forward_train_vel(self, pos, mask, known):
        B, T, P, _ = pos.shape
        vel = torch.diff(pos, dim=1, prepend=pos[:, :1]) / self.vel_scale  # vel[t] = (pos[t]-pos[t-1])/scale
        state, outs = self.init_state(B, P, pos.device), []
        for t in range(T - 1):
            out, state = self.step(vel[:, t], pos[:, t], mask, state)
            outs.append(out)
        out = torch.stack(outs, 1)  # [B, 19, P, 5] predicts vel[1:]
        valid = (mask[:, None, :] & known[:, 1:] & known[:, :-1]).float()
        return gaussian_nll(out, vel[:, 1:], valid)

    @torch.no_grad()
    def _predict_vel(self, pos, mask):
        B, T, P, _ = pos.shape
        vel = torch.diff(pos, dim=1, prepend=pos[:, :1]) / self.vel_scale
        state = self.init_state(B, P, pos.device)
        cur, v_in, preds = pos[:, 0], vel[:, 0], []
        for t in range(T_ALL - 1):
            if t < T_OBS:  # observed part: teacher forcing with true positions / velocities
                cur, v_in = pos[:, t], vel[:, t]
            out, state = self.step(v_in, cur, mask, state)
            if t >= T_OBS - 1:
                v_in = out[..., :2]  # feed back the predicted mean velocity
                cur = cur + v_in * self.vel_scale
                preds.append(cur)
        return torch.stack(preds, 1)  # [B, 12, P, 2]


@torch.no_grad()
def evaluate_reference(model, loader, device):
    from .metrics import MetricAccumulator
    model.eval()
    acc = MetricAccumulator()
    for b in loader:
        pos, mask = b["pos"].to(device), b["mask"].to(device)
        pred = model.predict(pos, mask).cpu()  # [B,12,P,2]
        gt, m = b["pos"][:, T_OBS:], b["mask"]
        for bi in range(pred.shape[0]):
            idx = m[bi].nonzero().flatten()
            acc.add(pred[bi][:, idx].permute(1, 0, 2).numpy(), gt[bi][:, idx].permute(1, 0, 2).numpy(),
                    b["src"][bi][idx].tolist())
    return acc.summary()
