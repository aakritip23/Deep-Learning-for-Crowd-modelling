"""Figures for the project report (matplotlib, headless).  All functions save a PNG and return its path."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .baselines import constant_velocity_mean  # noqa: E402
from .data import SOURCES, T_OBS, TrajDataset, build_protocol, collate  # noqa: E402

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f"]


def _read_log(run_dir):
    path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def plot_curves(runs, out_png, title="Validation Avg during training", ymax=None):
    """runs: {label: run_dir}.  Plots validation Avg (lower is better) per epoch; dashed line = constant velocity.
    ymax defaults to 1.5 x the constant-velocity score; runs that stay above it are listed as '(above range)'."""
    series, cv = {}, None
    for label, d in runs.items():
        rows = _read_log(d)
        if not rows:
            continue
        series[label] = ([int(r["epoch"]) for r in rows], [float(r["val_Avg"]) for r in rows])
        if "cv_Avg" in rows[0]:
            cv = float(rows[0]["cv_Avg"])
    if ymax is None:
        ymax = 1.5 * cv if cv else max(max(v) for _, v in series.values()) * 1.05
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (label, (ep, val)) in enumerate(series.items()):
        off = min(val) > ymax
        ax.plot(ep, val, marker="o", ms=3, lw=1.8, color=PALETTE[i % len(PALETTE)],
                label=label + (f" (above range, best {min(val):.2f})" if off else ""))
    if cv is not None:
        ax.axhline(cv, color="k", ls="--", lw=1, label=f"constant velocity ({cv:.3f})")
    ax.set_ylim(top=ymax)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Avg = (ADE+FDE)/2  [m]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_comparison(tables, out_png, highlight="Social Transformer"):
    """tables: {panel title: [(model name, Avg, ADE, FDE), ...]} -> horizontal bars, one panel per protocol."""
    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 0.55 * max(len(v) for v in tables.values()) + 1.6), squeeze=False)
    for ax, (title, rows) in zip(axes[0], tables.items()):
        rows = sorted(rows, key=lambda r: r[1], reverse=True)
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = ["#d62728" if highlight in nme else "#7f7f7f" if "onstant" in nme else "#1f77b4" for nme in names]
        ax.barh(names, vals, color=colors)
        for y, r in enumerate(rows):
            ax.text(r[1] + 0.01, y, f"{r[1]:.3f}  (ADE {r[2]:.2f} / FDE {r[3]:.2f})", va="center", fontsize=8)
        ax.set_xlim(0, max(vals) * 1.8)
        ax.set_xlabel("Avg = (ADE + FDE) / 2  [m]   (lower is better)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_per_source(per_source, out_png, title="Avg per source dataset"):
    """per_source: {model: {source: Avg}} (grouped bars)."""
    models = list(per_source)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [per_source[m].get(s, np.nan) for s in SOURCES]
        ax.bar(np.arange(len(SOURCES)) + i * w, vals, w, label=m, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(np.arange(len(SOURCES)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels(SOURCES)
    ax.set_ylabel("Avg  [m]  (lower is better)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_examples(ckpt, data_root, out_png, protocol="official", holdout=(), n=12, seed=0, max_files_per_source=None,
                  device="cpu"):
    """Predicted vs true trajectories (target-centred canonical frame, metres) for random evaluation pedestrians.

    blue = observed, green = ground truth, red = model, grey dashed = constant velocity, thin blue = neighbours' observed tracks.
    Half of the panels are pedestrians that have neighbours (where the social module can act)."""
    import torch

    from .predict import load_model, predict_batch

    model, max_nbrs = load_model(ckpt, torch.device(device))
    _, scenes = build_protocol(data_root, protocol, list(holdout), max_files_per_source)
    ds = TrajDataset(scenes, max_nbrs)
    rng = np.random.default_rng(seed)
    with_nb = [i for i in range(len(ds)) if len(scenes[ds.index[i][0]].ids) > 1]
    without = [i for i in range(len(ds)) if len(scenes[ds.index[i][0]].ids) == 1]
    pick = list(rng.choice(with_nb, min(n // 2 + n % 2, len(with_nb)), replace=False)) if with_nb else []
    pick += list(rng.choice(without, min(n - len(pick), len(without)), replace=False)) if without else []
    batch = collate([ds[i] for i in pick])
    pred = predict_batch([model], batch, torch.device(device), tta=True)
    cv = constant_velocity_mean(batch["obs"].numpy())
    cols = 4
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(pick):
            ax.axis("off")
            continue
        obs, gt = batch["obs"][k].numpy(), batch["fut"][k].numpy()
        msk = batch["nbr_mask"][k].numpy()
        for nb in batch["nbr"][k].numpy()[msk]:
            ax.plot(nb[:, 0], nb[:, 1], color="#1f77b4", lw=0.8, alpha=0.4)
            ax.plot(nb[-1, 0], nb[-1, 1], "o", ms=3, color="#1f77b4", alpha=0.4)
        ax.plot(obs[:, 0], obs[:, 1], "-o", ms=3, color="#1f77b4", lw=2, label="observed")
        ax.plot(np.r_[obs[-1:, 0], gt[:, 0]], np.r_[obs[-1:, 1], gt[:, 1]], "-o", ms=3, color="#2ca02c", lw=2, label="ground truth")
        ax.plot(np.r_[obs[-1:, 0], cv[k][:, 0]], np.r_[obs[-1:, 1], cv[k][:, 1]], "--", color="#7f7f7f", lw=1.5, label="const. velocity")
        ax.plot(np.r_[obs[-1:, 0], pred[k][:, 0]], np.r_[obs[-1:, 1], pred[k][:, 1]], "-o", ms=3, color="#d62728", lw=2, label="model")
        e_m = np.linalg.norm(pred[k] - gt, axis=-1)
        e_c = np.linalg.norm(cv[k] - gt, axis=-1)
        src = SOURCES[batch["source"][k]] if 0 <= batch["source"][k] < len(SOURCES) else "?"
        ax.set_title(f"{src} | {int(msk.sum())} nbr | ADE model {e_m.mean():.2f} / CV {e_c.mean():.2f}", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=6, loc="best")
    fig.suptitle("Predicted vs true trajectories (target-centred frame, metres)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png
