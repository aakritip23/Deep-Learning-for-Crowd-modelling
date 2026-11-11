"""Train / evaluate the Social Transformer.   python -m trajpred.train --data-root <data> --out-dir runs/exp1

Device-agnostic (--device auto = cuda if available else cpu), seeded, checkpoints every epoch and
resumes automatically from <out-dir>/last.pt.
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

from .baselines import constant_velocity_mean
from .data import TrajDataset, augment, build_protocol, collate, observed_velocity_rms
from .metrics import MetricAccumulator, format_table
from .models import TrajPredictor, displacement_loss


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", default="official", choices=["official", "holdout", "full"],
                    help="full = final fit on train+validation (validation is then in-sample, monitoring only)")
    ap.add_argument("--select", default="best", choices=["best", "last"],
                    help="weights stored in best.pt: lowest validation Avg, or the last epoch (use with --protocol full)")
    ap.add_argument("--holdout", default="", help="comma separated video names for protocol=holdout")
    ap.add_argument("--max-files-per-source", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-train-samples", type=int, default=None, help="smoke tests only")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-nbrs", type=int, default=16)
    ap.add_argument("--no-social", action="store_true", help="individual Transformer (ablation)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--source-emb", action="store_true", help="add a learned embedding of the source dataset")
    ap.add_argument("--fde-weight", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15, help="early stopping on validation Avg (0 = off)")
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc, cv = MetricAccumulator(), MetricAccumulator()
    for batch in loader:
        pred = model(batch["obs"].to(device), batch["nbr"].to(device), batch["nbr_mask"].to(device),
                     batch["source"].to(device))
        gt = batch["fut"].numpy()
        src = batch["source"].tolist()
        acc.add(pred.cpu().numpy(), gt, src)
        cv.add(constant_velocity_mean(batch["obs"].numpy()), gt, src)
    return acc.summary(), cv.summary()


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    print("device:", device)

    train_scenes, eval_scenes = build_protocol(args.data_root, args.protocol,
                                               [h for h in args.holdout.split(",") if h],
                                               args.max_files_per_source)
    train_ds = TrajDataset(train_scenes, args.max_nbrs)
    eval_ds = TrajDataset(eval_scenes, args.max_nbrs)
    if args.max_train_samples:
        train_ds.index = train_ds.index[: args.max_train_samples]
    print(f"train samples {len(train_ds)} | eval samples {len(eval_ds)}")
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size)
    eval_loader = DataLoader(eval_ds, 256, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    vel_scale = observed_velocity_rms(train_scenes)
    model = TrajPredictor(vel_scale, args.d_model, args.nhead, args.nlayers, args.ff, args.dropout,
                          social=not args.no_social, source_emb=args.source_emb).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | vel_scale {vel_scale:.4f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warm = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))

    json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)
    start_epoch, best, bad = 0, float("inf"), 0
    last_path = os.path.join(args.out_dir, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"resumed from epoch {start_epoch}")

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    log_path = os.path.join(args.out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["epoch", "train_loss", "val_ADE", "val_FDE", "val_Avg", "cv_Avg", "seconds"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if not args.no_augment:
                batch = augment(batch, rng)
            pred = model(batch["obs"], batch["nbr"], batch["nbr_mask"], batch["source"])
            loss = displacement_loss(pred, batch["fut"], args.fde_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item(); nb += 1
        val, cv = evaluate(model, eval_loader, device)
        a = val["ALL"]
        print(f"epoch {epoch + 1}/{args.epochs} loss {run / nb:.4f} | val ADE {a['ADE']:.3f} FDE {a['FDE']:.3f} "
              f"Avg {a['Avg']:.3f} (const-vel Avg {cv['ALL']['Avg']:.3f}) | {time.time() - t0:.0f}s")
        writer.writerow([epoch + 1, run / nb, a["ADE"], a["FDE"], a["Avg"], cv["ALL"]["Avg"], time.time() - t0])
        log.flush()
        improved = a["Avg"] < best
        if improved:
            best, bad = a["Avg"], 0
        else:
            bad += 1
        if improved or args.select == "last":
            torch.save({"model": model.state_dict(), "args": vars(args), "vel_scale": vel_scale}, os.path.join(args.out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad}, last_path)
        if args.patience and args.select == "best" and bad >= args.patience:
            print("early stopping")
            break

    ck = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    val, cv = evaluate(model, eval_loader, device)
    print(format_table(val, "\n== best model"))
    print(format_table(cv, "\n== constant velocity (reference)"))
    json.dump({"model": val, "const_vel": cv}, open(os.path.join(args.out_dir, "final_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
