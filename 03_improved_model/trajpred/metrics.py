"""TrajNet metrics.  ADE = MAD (mean over 12 steps), FDE = FAD (last step), Avg = (ADE + FDE) / 2 = ranking score."""
from collections import defaultdict

import numpy as np

from .data import SOURCES


def displacement_errors(pred, gt):
    """pred, gt: [B, T, 2] numpy -> (ade [B], fde [B])."""
    d = np.linalg.norm(pred - gt, axis=-1)
    return d.mean(axis=1), d[:, -1]


class MetricAccumulator:
    """Accumulates errors overall and per source dataset."""

    def __init__(self):
        self.ade = defaultdict(float)
        self.fde = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, pred, gt, sources):
        ade, fde = displacement_errors(pred, gt)
        for a, f, s in zip(ade, fde, sources):
            name = SOURCES[s] if 0 <= s < len(SOURCES) else "other"
            for key in (name, "ALL"):
                self.ade[key] += float(a)
                self.fde[key] += float(f)
                self.n[key] += 1

    def summary(self):
        out = {}
        for key, n in self.n.items():
            ade, fde = self.ade[key] / n, self.fde[key] / n
            out[key] = {"n": n, "ADE": ade, "FDE": fde, "Avg": 0.5 * (ade + fde)}
        return out


def format_table(summary, title=""):
    rows = [title] if title else []
    rows.append(f"{'source':10s}{'n':>8s}{'ADE':>9s}{'FDE':>9s}{'Avg':>9s}")
    for key in [*SOURCES, "other", "ALL"]:
        if key in summary:
            r = summary[key]
            rows.append(f"{key:10s}{r['n']:8d}{r['ADE']:9.3f}{r['FDE']:9.3f}{r['Avg']:9.3f}")
    return "\n".join(rows)
