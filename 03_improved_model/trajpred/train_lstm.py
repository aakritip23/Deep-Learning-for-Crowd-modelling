"""Train / score the re-implemented reference models on our split.
   python -m trajpred.train_lstm --data-root <data> --out-dir runs/social_lstm            (Social LSTM)
   python -m trajpred.train_lstm --data-root <data> --out-dir runs/vanilla_lstm --no-social (Vanilla LSTM)
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_protocol, observed_velocity_rms
from .metrics import format_table
from .reference_lstm import ReferenceLSTM, SceneDataset, evaluate_reference, scene_collate


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", default="official", choices=["official", "holdout"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--max-files-per-source", type=int, default=None)
    ap.add_argument("--max-train-scenes", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32, help="scenes per batch")
    ap.add_argument("--optim", default="rmsprop", choices=["rmsprop", "adam"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--decay", type=float, default=0.95, help="per-epoch lr decay (professor's config)")
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--rnn", type=int, default=128)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--nbr-size", type=float, default=4.0, help="social neighbourhood side length in metres")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--no-social", action="store_true")
    ap.add_argument("--mode", default="abs", choices=["abs", "vel"],
                    help="abs = original (predict positions); vel = refined (predict per-step displacements)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    return ap.parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    train_scenes, eval_scenes = build_protocol(a.data_root, a.protocol, [h for h in a.holdout.split(",") if h],
                                               a.max_files_per_source)
    if a.max_train_scenes:
        train_scenes = train_scenes[: a.max_train_scenes]
    print(f"device {device} | train scenes {len(train_scenes)} | eval scenes {len(eval_scenes)}")
    tl = DataLoader(SceneDataset(train_scenes), a.batch_size, shuffle=True, collate_fn=scene_collate)
    el = DataLoader(SceneDataset(eval_scenes), 64, shuffle=False, collate_fn=scene_collate)
    a.vel_scale = observed_velocity_rms(train_scenes)
    model = ReferenceLSTM(not a.no_social, a.rnn, a.emb, a.grid, a.nbr_size, a.dropout,
                          mode=a.mode, vel_scale=a.vel_scale).to(device)
    print("params", sum(p.numel() for p in model.parameters()))
    opt = (torch.optim.RMSprop(model.parameters(), lr=a.lr) if a.optim == "rmsprop"
           else torch.optim.Adam(model.parameters(), lr=a.lr))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, a.decay)
    json.dump(vars(a), open(os.path.join(a.out_dir, "config.json"), "w"), indent=2)
    log = open(os.path.join(a.out_dir, "log.csv"), "w", newline="")
    w = csv.writer(log); w.writerow(["epoch", "train_nll", "val_ADE", "val_FDE", "val_Avg", "seconds"])
    best, bad = float("inf"), 0
    for epoch in range(a.epochs):
        model.train(); t0 = time.time(); run = n = 0
        for b in tl:
            pos, mask, known = (b[k].to(device) for k in ("pos", "mask", "known"))
            loss = model.forward_train(pos, mask, known)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step(); run += loss.item(); n += 1
        sched.step()
        v = evaluate_reference(model, el, device)["ALL"]
        print(f"epoch {epoch + 1}/{a.epochs} nll {run / n:.3f} | val ADE {v['ADE']:.3f} FDE {v['FDE']:.3f} "
              f"Avg {v['Avg']:.3f} | {time.time() - t0:.0f}s", flush=True)
        w.writerow([epoch + 1, run / n, v["ADE"], v["FDE"], v["Avg"], time.time() - t0]); log.flush()
        if v["Avg"] < best:
            best, bad = v["Avg"], 0
            torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(a.out_dir, "best.pt"))
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                print("early stopping"); break
    model.load_state_dict(torch.load(os.path.join(a.out_dir, "best.pt"), map_location=device)["model"])
    summary = evaluate_reference(model, el, device)
    print(format_table(summary, "\n== best model"))
    json.dump({"model": summary}, open(os.path.join(a.out_dir, "final_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
