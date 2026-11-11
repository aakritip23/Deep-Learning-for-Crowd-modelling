"""Validate a TrajNet submission directory written by `trajpred.predict --write-submission`.

  python check_submission.py --data-root <data> --submission <dir>

Checks, for every test file: the submission exists, has the same number of rows, observed rows are byte-identical,
every '?' was replaced by two finite numbers, no row was changed that did not contain '?', and the predicted points
are plausible (within a generous distance of the last observed point).
"""
import argparse
import glob
import math
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--max-files-per-source", type=int, default=None, help="only for smoke runs")
    a = ap.parse_args()

    problems, n_files, n_filled, worst = [], 0, 0, 0.0
    per_source = {}
    for src_path in sorted(glob.glob(os.path.join(a.data_root, "test", "*", "*.txt"))):
        src = os.path.basename(os.path.dirname(src_path))
        name = os.path.basename(src_path)
        out_path = os.path.join(a.submission, f"{src}_{name}")
        if not os.path.exists(out_path):
            if a.max_files_per_source is None:
                problems.append(f"missing {out_path}")
            continue
        per_source[src] = per_source.get(src, 0) + 1
        orig = open(src_path).read().splitlines()
        new = open(out_path).read().splitlines()
        n_files += 1
        if len(orig) != len(new):
            problems.append(f"{name}: {len(orig)} rows in the test file but {len(new)} in the submission")
            continue
        last_obs = {}
        for i, (o, n) in enumerate(zip(orig, new)):
            po, pn = o.split(), n.split()
            if len(po) != 4:
                continue
            if "?" not in po:
                if o != n:
                    problems.append(f"{name} line {i + 1}: observed row was modified")
                last_obs[po[1]] = (float(po[2]), float(po[3]))
                continue
            n_filled += 1
            if len(pn) != 4 or pn[0] != po[0] or pn[1] != po[1]:
                problems.append(f"{name} line {i + 1}: frame/ped id changed or malformed row: {n!r}")
                continue
            try:
                x, y = float(pn[2]), float(pn[3])
            except ValueError:
                problems.append(f"{name} line {i + 1}: non-numeric prediction {n!r}")
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                problems.append(f"{name} line {i + 1}: non-finite prediction")
                continue
            if po[1] in last_obs:
                worst = max(worst, math.hypot(x - last_obs[po[1]][0], y - last_obs[po[1]][1]))
    print(f"checked {n_files} files {per_source}, {n_filled} predicted rows, "
          f"largest distance from last observed point: {worst:.1f} m")
    if worst > 60:
        problems.append(f"implausible prediction: {worst:.1f} m away from the last observed point")
    if problems:
        print(f"FAILED ({len(problems)} problems), first ones:")
        print("\n".join(problems[:10]))
        sys.exit(1)
    print("submission OK")


if __name__ == "__main__":
    main()
