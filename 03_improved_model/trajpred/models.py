"""Social Transformer trajectory predictor.

obs [B, 8, 2] (canonical frame, metres)  ->  pred [B, 12, 2]

* TrajEncoder      : Transformer encoder over the 8 observed (position, velocity) steps.
* SocialAttention  : the target attends over its neighbours' encodings (+ relative position / velocity);
                     a learned "null" key keeps the attention defined for pedestrians without neighbours.
* Decoder          : MLP -> 12 per-step velocity *residuals* that are cumulated and added to a
                     constant-velocity extrapolation (zero-initialised => the untrained model equals
                     the constant-velocity baseline).
`social=False` gives the individual Transformer used as the ablation / core model.
"""
import torch
from torch import nn

from .data import T_OBS, T_PRED


class TrajEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, nlayers=3, ff=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(4, d_model)  # (x, y, vx, vy)
        self.pos = nn.Parameter(torch.randn(1, T_OBS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(T_OBS * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, tracks):  # [M, 8, 2] (already scaled)
        vel = torch.diff(tracks, dim=1, prepend=tracks[:, :1])
        h = self.enc(self.inp(torch.cat([tracks, vel], dim=-1)) + self.pos)
        return self.out(h.flatten(1))


class SocialAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1, pos_scale=5.0):
        super().__init__()
        self.pos_scale = pos_scale
        self.rel = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_t, h_n, nbr, obs, nbr_mask, vel_scale):
        """h_t [B,d]; h_n [B,K,d]; nbr [B,K,8,2]; obs [B,8,2] (metres); nbr_mask [B,K] True = real neighbour."""
        B = h_t.shape[0]
        v_t = (obs[:, -1] - obs[:, 0]) / (T_OBS - 1)
        v_n = (nbr[:, :, -1] - nbr[:, :, 0]) / (T_OBS - 1)
        rel = torch.cat([(nbr[:, :, -1] - obs[:, None, -1]) / self.pos_scale,
                         (v_n - v_t[:, None]) / vel_scale], dim=-1)
        kv = torch.cat([self.null.expand(B, -1, -1), h_n + self.rel(rel)], dim=1)
        ignore = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=h_t.device), ~nbr_mask], dim=1)
        out, _ = self.attn(h_t.unsqueeze(1), kv, kv, key_padding_mask=ignore, need_weights=False)
        return self.norm(h_t + out.squeeze(1))


class TrajPredictor(nn.Module):
    def __init__(self, vel_scale=0.3, d_model=128, nhead=4, nlayers=3, ff=256, dropout=0.1, social=True,
                 source_emb=False, n_sources=4):
        super().__init__()
        self.social = social
        # optional learned embedding of the source dataset (biwi/crowds/mot/stanford; index 0 = unknown)
        self.source_emb = nn.Embedding(n_sources + 1, d_model) if source_emb else None
        if self.source_emb is not None:
            nn.init.normal_(self.source_emb.weight, std=0.1)
        self.register_buffer("vel_scale", torch.tensor(float(vel_scale)))
        self.encoder = TrajEncoder(d_model, nhead, nlayers, ff, dropout)
        self.social_attn = SocialAttention(d_model, nhead, dropout) if social else None
        self.decoder = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                     nn.Linear(2 * d_model, T_PRED * 2))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, obs, nbr=None, nbr_mask=None, source=None):
        s = self.vel_scale
        h = self.encoder(obs / s)
        if self.social and nbr is not None:
            B, K = nbr.shape[:2]
            # encode only real neighbours (padded slots are masked out anyway) - big saving in crowded batches
            h_n = h.new_zeros(B, K, h.shape[-1])
            if nbr_mask.any():
                h_n[nbr_mask] = self.encoder(nbr[nbr_mask] / s)
            h = self.social_attn(h, h_n, nbr, obs, nbr_mask, s)
        if self.source_emb is not None and source is not None:
            h = h + self.source_emb(source + 1)  # source -1 (unknown) -> index 0
        res =torch.cumsum(self.decoder(h).view(-1, T_PRED, 2), dim=1) * s
        v = (obs[:, -1] - obs[:, 0]) / (T_OBS - 1)
        steps = torch.arange(1, T_PRED + 1, device=obs.device, dtype=obs.dtype)[None, :, None]
        return obs[:, -1:, :] + v[:, None, :] * steps + res


def displacement_loss(pred, gt, fde_weight=0.5, eps=1e-6):
    """Mean Euclidean distance (= ADE) + weighted final-step distance (= FDE); optimises the ranking metric."""
    d = torch.sqrt(((pred - gt) ** 2).sum(-1) + eps)
    return d.mean() + fde_weight * d[:, -1].mean()
