"""Paired comparison of two models (or two ensembles) on exactly the same trajectories.

  python compare_models.py --data-root <data> --a runs/official/indiv_tf/best.pt --b runs/official/social_tf/best.pt
  python compare_models.py ... --a A1.pt A2.pt --b B.pt --protocol holdout --holdout video1,video2 --tta

Reports mean Avg = (ADE+FDE)/2 per trajectory for A and B, the paired difference (B - A; negative = B better) with a
95% bootstrap confidence interval over trajectories, the share of trajectories where B is better, and the same per
source dataset.  The bootstrap resamples trajectories; it does not capture seed-to-seed variance (use several seeds).

--cluster: the held-out evaluation set contains every trajectory three times (the original and two rotated, noisy copies
of the dataset's augmentation: same video, pedestrian id and frames).  Rows are then not independent and the plain
bootstrap is too narrow; with --cluster the bootstrap resamples the unique trajectories and decides the verdict on that
interval (recommended for --protocol holdout).
"""
import argparse

import numpy as np
import torch

from trajpred.data import SOURCES, T_OBS, TrajDataset, build_protocol
from trajpred.predict import load_model, per_sample_errors
from trajpred.reference_lstm import ReferenceLSTM, SceneDataset, scene_collate
from torch.utils.data import DataLoader


def lstm_errors(ckpt, scenes, device):
    """Per-trajectory (ADE, FDE, source) of a re-implemented LSTM checkpoint, in the same order as TrajDataset."""
    ck = torch.load(ckpt, map_location=device)
    a = ck["args"]
    model = ReferenceLSTM(not a["no_social"], a["rnn"], a["emb"], a["grid"], a["nbr_size"], a["dropout"],
                          mode=a.get("mode", "abs"), vel_scale=a.get("vel_scale", 0.25)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ades, fdes, srcs = [], [], []
    with torch.no_grad():
        for b in DataLoader(SceneDataset(scenes), 64, shuffle=False, collate_fn=scene_collate):
            pred = model.predict(b["pos"].to(device), b["mask"].to(device)).cpu()
            gt = b["pos"][:, T_OBS:]
            for bi in range(pred.shape[0]):
                idx = b["mask"][bi].nonzero().flatten()
                d = np.linalg.norm(pred[bi][:, idx].permute(1, 0, 2).numpy()
                                   - gt[bi][:, idx].permute(1, 0, 2).numpy(), axis=-1)
                ades.append(d.mean(1)); fdes.append(d[:, -1]); srcs.append(b["src"][bi][idx].numpy())
    return np.concatenate(ades), np.concatenate(fdes), np.concatenate(srcs)


def errors_for(cks, scenes, device, tta):
    """Transformer checkpoint(s) (ensemble) or a single re-implemented LSTM checkpoint (detected from its args)."""
    if "rnn" in torch.load(cks[0], map_location="cpu")["args"]:
        assert len(cks) == 1, "LSTM checkpoints cannot be ensembled here"
        return lstm_errors(cks[0], scenes, device)
    loaded = [load_model(c, device) for c in cks]
    return per_sample_errors([m for m, _ in loaded], scenes, device, tta, loaded[0][1])


def boot_ci(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def cluster_boot_ci(diff, groups, n=4000, seed=0):
    """Bootstrap over groups of rows (e.g. the augmented copies of one trajectory): returns (CI, number of groups)."""
    _, inv = np.unique(groups, return_inverse=True)
    g = np.bincount(inv, weights=diff) / np.bincount(inv)      # mean difference per unique trajectory
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    return np.percentile(g[idx].mean(axis=1), [2.5, 97.5]), len(g)


def report(name, a, b, groups=None):
    d = b - a
    lo, hi = boot_ci(d)
    extra = ""
    if groups is not None:
        (lo, hi), ng = cluster_boot_ci(d, groups)
        extra = f" cluster-bootstrap over {ng} unique trajectories"
    verdict = "B better" if hi < 0 else "A better" if lo > 0 else "no clear difference"
    print(f"{name:10s} n={len(d):6d}  A {a.mean():.3f}  B {b.mean():.3f}  diff(B-A) {d.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  B better on {100 * (d < 0).mean():.0f}%  -> {verdict}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--a", nargs="+", required=True, help="checkpoint(s) of model/ensemble A (Transformer or LSTM)")
    ap.add_argument("--b", nargs="+", required=True, help="checkpoint(s) of model/ensemble B")
    ap.add_argument("--protocol", default="official", choices=["official", "holdout"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--max-files-per-source", type=int, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cluster", action="store_true",
                    help="bootstrap over unique trajectories (the augmented copies of one trajectory count once)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    _, ev = build_protocol(args.data_root, args.protocol, [h for h in args.holdout.split(",") if h],
                           args.max_files_per_source)
    res = []
    for cks in (args.a, args.b):
        ade, fde, src = errors_for(cks, ev, device, args.tta)
        res.append((0.5 * (ade + fde), src))
    (a, src), (b, _) = res
    groups = None
    if args.cluster:        # same order as TrajDataset (and as the error arrays above)
        index = TrajDataset(ev, 16).index
        keys = [(ev[si].video, int(float(ev[si].ids[pi])), int(ev[si].frames[pi][0])) for si, pi in index]
        ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        groups = np.array([ids[k] for k in keys])
    report("ALL", a, b, groups)
    for i, s in enumerate(SOURCES):
        m = src == i
        if m.sum() > 1:
            report(s, a[m], b[m], None if groups is None else groups[m])


if __name__ == "__main__":
    main()
