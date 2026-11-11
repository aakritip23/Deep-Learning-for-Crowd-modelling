"""Score the non-learned baselines (numpy only) on a split / protocol.

  python evaluate_baselines.py --data-root ../01_original_social_lstm/social-lstm/data
  python evaluate_baselines.py --data-root ... --protocol holdout --holdout students003,crowds_zara02
"""
import argparse

import numpy as np

from trajpred.baselines import BASELINES
from trajpred.data import TrajDataset, build_protocol
from trajpred.metrics import MetricAccumulator, format_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--protocol", default="official", choices=["official", "holdout"])
    ap.add_argument("--holdout", default="", help="comma separated video names (protocol=holdout)")
    ap.add_argument("--max-files-per-source", type=int, default=None)
    args = ap.parse_args()

    _, eval_scenes = build_protocol(args.data_root, args.protocol,
                                    [h for h in args.holdout.split(",") if h], args.max_files_per_source)
    ds = TrajDataset(eval_scenes, max_nbrs=0)
    samples = [ds[i] for i in range(len(ds))]
    obs = np.stack([s["obs"] for s in samples]).astype(np.float64)
    fut = np.stack([s["fut"] for s in samples]).astype(np.float64)
    src = [s["source"] for s in samples]
    print(f"{len(ds)} evaluation trajectories, protocol={args.protocol}")
    for name, fn in BASELINES.items():
        acc = MetricAccumulator()
        acc.add(fn(obs), fut, src)
        print(format_table(acc.summary(), f"\n== {name}"))


if __name__ == "__main__":
    main()
