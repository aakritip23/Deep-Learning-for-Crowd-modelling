"""Ensemble / test-time-augmentation prediction, validation scoring and TrajNet submission writing.

  # score checkpoints (single model or ensemble) on a split / protocol
  python -m trajpred.predict --data-root <data> --ckpt runs/a/best.pt runs/b/best.pt --tta
  # write the TrajNet test submission (the test .txt files with every '?' replaced by a prediction)
  python -m trajpred.predict --data-root <data> --ckpt runs/a/best.pt --tta --write-submission out/submission
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import T_OBS, T_PRED, TrajDataset, build_protocol, collate, load_scenes, to_world
from .metrics import MetricAccumulator, format_table
from .models import TrajPredictor


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    model = TrajPredictor(ck["vel_scale"], a["d_model"], a["nhead"], a["nlayers"], a["ff"], a["dropout"],
                          social=not a["no_social"], source_emb=a.get("source_emb", False)).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), a["max_nbrs"]


@torch.no_grad()
def predict_batch(models, batch, device, tta=True):
    """Average over models (and the mirrored input if tta).  Returns canonical-frame predictions [B,12,2] numpy."""
    obs, nbr, mask = batch["obs"].to(device), batch["nbr"].to(device), batch["nbr_mask"].to(device)
    src = batch["source"].to(device)
    flip = torch.tensor([1.0, -1.0], device=device)
    preds = []
    for m in models:
        preds.append(m(obs, nbr, mask, src))
        if tta:
            preds.append(m(obs * flip, nbr * flip, mask, src) * flip)
    return torch.stack(preds).mean(0).cpu().numpy()


def run(models, scenes, device, tta=True, max_nbrs=16, with_world=False):
    ds = TrajDataset(scenes, max_nbrs)
    acc, world = MetricAccumulator(), {}
    for batch in DataLoader(ds, 256, shuffle=False, collate_fn=collate):
        pred = predict_batch(models, batch, device, tta)
        gt = batch["fut"].numpy()
        if np.isfinite(gt).all():
            acc.add(pred, gt, batch["source"].tolist())
        if with_world:
            for i, (si, pi) in enumerate(zip(batch["scene"], batch["ped"])):
                world[(si, pi)] = to_world(pred[i], batch["origin"][i], batch["angle"][i])
    return acc.summary(), world


def per_sample_errors(models, scenes, device, tta=True, max_nbrs=16):
    """Per-trajectory (ADE, FDE, source) arrays in dataset order - for paired model comparisons."""
    ds = TrajDataset(scenes, max_nbrs)
    ades, fdes, srcs = [], [], []
    for batch in DataLoader(ds, 256, shuffle=False, collate_fn=collate):
        pred = predict_batch(models, batch, device, tta)
        d = np.linalg.norm(pred - batch["fut"].numpy(), axis=-1)
        ades.append(d.mean(1)); fdes.append(d[:, -1]); srcs.append(batch["source"].numpy())
    return np.concatenate(ades), np.concatenate(fdes), np.concatenate(srcs)


def write_submission(root, scenes, world, out_dir):
    """Re-write each test file with '?' replaced by the predicted coordinates (same row order/format)."""
    os.makedirs(out_dir, exist_ok=True)
    pred = {}  # (file, ped_id_str, frame) -> (c3, c4)
    for si, sc in enumerate(scenes):
        for pi, pid in enumerate(sc.ids):
            for k in range(T_PRED):
                pred[(sc.file, pid, float(sc.frames[pi, T_OBS + k]))] = world[(si, pi)][k]
    files = {sc.file: sc.source for sc in scenes}
    n_filled = 0
    for fname, src in files.items():
        src_path = os.path.join(root, "test", src, fname)
        out_path = os.path.join(out_dir, src + "_" + fname)
        with open(src_path) as fin, open(out_path, "w") as fout:
            for line in fin:
                p = line.split()
                if len(p) == 4 and p[2] == "?":
                    x, y = pred[(fname, p[1], float(p[0]))]
                    line = f"{p[0]} {p[1]} {x:.3f} {y:.3f}\n"
                    n_filled += 1
                fout.write(line)
    return n_filled


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--protocol", default="official", choices=["official", "holdout"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--max-files-per-source", type=int, default=None)
    ap.add_argument("--tta", action="store_true", help="average with the mirrored prediction")
    ap.add_argument("--write-submission", default=None, help="output dir; predicts the hidden-GT test split")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    loaded = [load_model(c, device) for c in a.ckpt]
    models, max_nbrs = [m for m, _ in loaded], loaded[0][1]
    if a.write_submission:
        scenes = load_scenes(a.data_root, "test", max_files_per_source=a.max_files_per_source)
        _, world = run(models, scenes, device, a.tta, max_nbrs, with_world=True)
        n = write_submission(a.data_root, scenes, world, a.write_submission)
        print(f"wrote {n} predicted rows to {a.write_submission}")
    else:
        _, ev = build_protocol(a.data_root, a.protocol, [h for h in a.holdout.split(",") if h], a.max_files_per_source)
        summary, _ = run(models, ev, device, a.tta, max_nbrs)
        print(format_table(summary, f"== {len(models)} model(s), tta={a.tta}"))


if __name__ == "__main__":
    main()
