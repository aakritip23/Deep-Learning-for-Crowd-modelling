"""Collect finished runs into one comparison table (markdown).

  python summarize_runs.py runs/official runs/holdout
Reads <dir>/<run>/final_metrics.json (written at the end of training) and falls back to the best row of log.csv
for runs that are still going (marked 'running').  The constant-velocity reference is taken from any Transformer run.
"""
import csv
import json
import math
import os
import re
import sys

SRC = ["biwi", "crowds", "mot", "stanford"]


def best_from_log(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None, 0
    b = min(rows, key=lambda r: float(r["val_Avg"]))
    return {"ADE": float(b["val_ADE"]), "FDE": float(b["val_FDE"]), "Avg": float(b["val_Avg"])}, len(rows)


def collect(d):
    res, ref = {}, None
    for name in sorted(os.listdir(d)):
        run = os.path.join(d, name)
        if not os.path.isdir(run):
            continue
        fm = os.path.join(run, "final_metrics.json")
        if os.path.exists(fm):
            j = json.load(open(fm))
            res[name] = (j["model"], "done")
            ref = ref or j.get("const_vel")
        elif os.path.exists(os.path.join(run, "log.csv")):
            b, n = best_from_log(os.path.join(run, "log.csv"))
            if b:
                res[name] = ({"ALL": b}, f"running ({n} ep)")
    return res, ref


def group_seeds(res):
    """name_s0, name_s1, ... -> one row 'name (n seeds)' with mean over seeds (and std of Avg)."""
    groups = {}
    for name, (m, st) in res.items():
        g = re.sub(r"_s\d+$", "", name)
        groups.setdefault(g, []).append((m, st))
    out = {}
    for g, items in groups.items():
        if len(items) == 1 and g == next(iter(n for n in res if re.sub(r"_s\d+$", "", n) == g)):
            out[g] = items[0]
            continue
        keys = set().union(*(m.keys() for m, _ in items))
        agg = {}
        for k in keys:
            vals = [m[k] for m, _ in items if k in m]
            agg[k] = {f: sum(v[f] for v in vals) / len(vals) for f in ("ADE", "FDE", "Avg")}
            agg[k]["n"] = vals[0].get("n", 0)
            avgs = [v["Avg"] for v in vals]
            agg[k]["std"] = math.sqrt(sum((x - agg[k]["Avg"]) ** 2 for x in avgs) / max(1, len(avgs) - 1))
        out[f"{g} (mean of {len(items)} seeds)"] = (agg, "done" if all(s == "done" for _, s in items) else "running")
    return out


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    seeds = "--group-seeds" in sys.argv
    for d in args or ["runs/official"]:
        res, ref = collect(d)
        if seeds:
            res = group_seeds(res)
        print(f"\n### {d}\n")
        print("| model | status | ADE | FDE | **Avg** | " + " | ".join(f"{s} Avg" for s in SRC) + " |")
        print("|---|---|---|---|---|" + "---|" * len(SRC))
        rows = [("constant velocity (mean)", "reference", ref)] if ref else []
        rows += [(n, st, m) for n, (m, st) in res.items()]
        for name, status, m in sorted(rows, key=lambda r: r[2]["ALL"]["Avg"] if r[2] else 9e9):
            a = m["ALL"]
            per = " | ".join(f"{m[s]['Avg']:.3f}" if s in m else "-" for s in SRC)
            std = f" +/- {a['std']:.3f}" if "std" in a else ""
            print(f"| {name} | {status} | {a['ADE']:.3f} | {a['FDE']:.3f} | **{a['Avg']:.3f}**{std} | {per} |")


if __name__ == "__main__":
    main()
