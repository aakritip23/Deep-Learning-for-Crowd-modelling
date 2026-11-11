#!/usr/bin/env python
"""One-command pipeline for the finalised Social Transformer:
   train -> evaluate -> (optional) compare with the baselines -> figures -> TrajNet submission.

  python colab_run.py --data-root <folder with train/ validation/ test/> --out results_colab --mode quick

Modes
  smoke : 1 epoch on a tiny subset - only checks that the whole pipeline runs (~2-5 min on CPU)
  quick : 8 epochs (Transformers) - fast test of the results; Social Transformer reaches Avg ~1.04 vs ~1.01 for the full schedule
          (use --tf-epochs 4 for the absolute minimum: Avg ~1.07)
  full  : 30 epochs - reproduces the numbers in RESULTS.md

Re-running the same command resumes: finished runs are skipped, interrupted Transformer runs continue from last.pt.
Everything is written below --out:  runs/ (checkpoints+logs), results/ (tables, figures, paired tests), submission/.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
HOLD = "biwi_hotel,crowds_zara02,students003,PETS09-S2L1,bookstore_3,hyang_5,gates_3"
SOURCES = ["biwi", "crowds", "mot", "stanford"]

# epochs per run type.  tf = Social/Individual Transformer; lstm_prof = original-formulation Social LSTM (its validation
# error is best after ~3 epochs and then diverges, so a few epochs are enough); lstm_vel = refined velocity LSTM.
MODES = {
    "smoke": dict(tf=1, lstm_prof=1, lstm_vel=1),
    "quick": dict(tf=8, lstm_prof=4, lstm_vel=8),
    "full": dict(tf=30, lstm_prof=8, lstm_vel=20),
}
LABELS = {
    "social_tf": "Social Transformer (final)",
    "indiv_tf": "Individual Transformer (ablation)",
    "social_lstm_vel": "Social LSTM, refined (velocity)",
    "social_lstm_prof": "Social LSTM, professor's settings (re-impl.)",
}
# reference numbers from the full 30-epoch CPU study (RESULTS.md), Avg = (ADE+FDE)/2
REFERENCE = {
    ("official", "social_tf"): 1.008, ("official", "indiv_tf"): 1.054, ("official", "social_lstm_vel"): 1.105,
    ("official", "social_lstm_prof"): 2.193, ("official", "const_vel"): 1.206,
    ("holdout", "social_tf"): 1.020, ("holdout", "indiv_tf"): 1.044, ("holdout", "const_vel"): 1.169,
}


class Pipeline:
    def __init__(self, a):
        self.a = a
        self.data = os.path.abspath(a.data_root)
        self.out = os.path.abspath(a.out)
        self.runs = os.path.join(self.out, "runs")
        self.res = os.path.join(self.out, "results")
        os.makedirs(self.res, exist_ok=True)
        self.ep = dict(MODES[a.mode])
        if a.tf_epochs:
            self.ep["tf"] = a.tf_epochs
        self.smoke = a.mode == "smoke"
        self.sub = ["--max-files-per-source", "2"] if self.smoke else []
        self.timings = {}

    # ------------------------------------------------------------------ helpers
    def run(self, cmd, check=True, capture=False):
        print("\n$ " + " ".join(cmd), flush=True)
        return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=capture, text=True)

    def run_dir(self, protocol, name):
        return os.path.join(self.runs, protocol, name)

    def done(self, protocol, name):
        return os.path.exists(os.path.join(self.run_dir(protocol, name), "final_metrics.json"))

    def timed(self, key, fn):
        t0 = time.time()
        out = fn()
        self.timings[key] = round(time.time() - t0, 1)
        return out

    def proto_args(self, protocol):
        return {"official": [], "holdout": ["--protocol", "holdout", "--holdout", HOLD],
                "full": ["--protocol", "full", "--select", "last"]}[protocol]

    # ------------------------------------------------------------------ training
    def train_tf(self, name, protocol, extra=(), required=True):
        d = self.run_dir(protocol, name)
        if self.done(protocol, name):
            print(f"[skip] {protocol}/{name} already finished")
            return True
        cmd = [PY, "-u", "-m", "trajpred.train", "--data-root", self.data, "--out-dir", d, "--epochs",
               str(self.ep["tf"]), "--seed", str(self.a.seed)] + self.proto_args(protocol) + list(extra) + self.sub
        if self.smoke:
            cmd += ["--max-train-samples", "600", "--batch-size", "64"]
        return self._try(lambda: self.timed(f"{protocol}/{name}", lambda: self.run(cmd)), f"{protocol}/{name}", required)

    def train_lstm(self, name, protocol, kind, required=False):
        d = self.run_dir(protocol, name)
        if self.done(protocol, name):
            print(f"[skip] {protocol}/{name} already finished")
            return True
        cmd = [PY, "-u", "-m", "trajpred.train_lstm", "--data-root", self.data, "--out-dir", d, "--seed", str(self.a.seed)]
        cmd += self.proto_args(protocol) + self.sub
        if kind == "prof":   # the professor's configuration: predict positions, RMSprop, dropout 0.5
            cmd += ["--epochs", str(self.ep["lstm_prof"])]
        else:                # refined: predict per-step velocities, Adam, dropout 0.1
            cmd += ["--epochs", str(self.ep["lstm_vel"]), "--mode", "vel", "--optim", "adam", "--lr", "1e-3",
                    "--dropout", "0.1", "--patience", "6"]
        if self.smoke:
            cmd += ["--max-train-scenes", "300"]
        return self._try(lambda: self.timed(f"{protocol}/{name}", lambda: self.run(cmd)), f"{protocol}/{name}", required)

    def _try(self, fn, label, required):
        try:
            fn()
            return True
        except subprocess.CalledProcessError as e:
            if required:
                raise
            print(f"\n!!! WARNING: optional run {label} failed (exit {e.returncode}) - continuing without it.\n", flush=True)
            return False

    # ------------------------------------------------------------------ tables
    def metrics(self, protocol, name):
        p = os.path.join(self.run_dir(protocol, name), "final_metrics.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def table(self, protocol):
        """-> (markdown, rows) for one protocol; rows = [(label, metrics-by-source)]."""
        rows, cv = [], None
        for name in LABELS:
            m = self.metrics(protocol, name)
            if m is None:
                continue
            rows.append((LABELS[name], name, m["model"]))
            cv = cv or m.get("const_vel")
        if cv is None:
            return "", []
        rows.append(("Constant velocity (reference)", "const_vel", cv))
        rows.sort(key=lambda r: r[2]["ALL"]["Avg"])
        cv_avg = cv["ALL"]["Avg"]
        n = cv["ALL"]["n"]
        md = [f"**{'Official split (train -> validation)' if protocol == 'official' else 'Held-out videos (7 unseen videos)'}**"
              f" - {n} trajectories, metres, lower is better\n",
              "| Model | ADE | FDE | **Avg** | vs const. vel. | " + " | ".join(f"{s} Avg" for s in SOURCES) + " | full-study Avg |",
              "|---|---|---|---|---|" + "---|" * len(SOURCES) + "---|"]
        for label, key, m in rows:
            a = m["ALL"]
            per = " | ".join(f"{m[s]['Avg']:.3f}" if s in m else "-" for s in SOURCES)
            ref = REFERENCE.get((protocol, key))
            md.append(f"| {label} | {a['ADE']:.3f} | {a['FDE']:.3f} | **{a['Avg']:.3f}** | {a['Avg'] / cv_avg:.2f}x | {per} | "
                      f"{'%.3f' % ref if ref else '-'} |")
        return "\n".join(md), rows

    # ------------------------------------------------------------------ steps
    def step_env(self):
        print("=" * 100 + f"\nSocial Transformer pipeline | mode={self.a.mode} | epochs={self.ep}\n" + "=" * 100)
        try:
            import torch
            dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (slow - use a GPU runtime)"
            print(f"python {sys.version.split()[0]} | torch {torch.__version__} | device: {dev}")
        except ImportError:
            sys.exit("PyTorch is not installed (pip install -r requirements.txt from the repository root)")
        for split in ("train", "validation", "test"):
            n = len(glob.glob(os.path.join(self.data, split, "*", "*.txt")))
            if n == 0:
                sys.exit(f"ERROR: no .txt files in {self.data}/{split}/<source>/ - check --data-root")
            print(f"data/{split}: {n} files")

    def step_tests(self):
        if not self.a.skip_tests:
            self.run([PY, "tests/test_basic.py", "--data-root", self.data])

    def step_train(self):
        self.train_tf("social_tf", "official", required=True)          # the finalised model
        if not self.a.no_holdout:
            self.train_tf("social_tf", "holdout", required=True)
        if not self.a.no_comparison:
            self.train_tf("indiv_tf", "official", ["--no-social"], required=False)
            if not self.a.no_holdout:
                self.train_tf("indiv_tf", "holdout", ["--no-social"], required=False)
            self.train_lstm("social_lstm_prof", "official", "prof")
            self.train_lstm("social_lstm_vel", "official", "vel")
        if self.a.final_fit:
            self.train_tf("social_tf", "full", required=True)

    def step_tables(self):
        parts, summary = [], {"mode": self.a.mode, "epochs": self.ep, "timings_s": self.timings}
        for protocol in ("official", "holdout"):
            md, rows = self.table(protocol)
            if not md:
                continue
            open(os.path.join(self.res, f"results_{protocol}.md"), "w", encoding="utf-8").write(md + "\n")
            parts.append(md)
            summary[protocol] = {key: m for _, key, m in rows}
            print("\n" + md)
        json.dump(summary, open(os.path.join(self.res, "results.json"), "w"), indent=1)
        return parts

    def step_paired(self):
        """Paired bootstrap comparisons (B - A, negative = B better)."""
        pairs = [("official", "indiv_tf", "social_tf", "Individual TF -> Social TF"),
                 ("official", "social_lstm_vel", "social_tf", "Refined Social LSTM -> Social TF"),
                 ("holdout", "indiv_tf", "social_tf", "Individual TF -> Social TF")]
        lines = []
        for protocol, a, b, title in pairs:
            ca = os.path.join(self.run_dir(protocol, a), "best.pt")
            cb = os.path.join(self.run_dir(protocol, b), "best.pt")
            if not (os.path.exists(ca) and os.path.exists(cb)):
                continue
            cmd = [PY, "compare_models.py", "--data-root", self.data, "--a", ca, "--b", cb] + self.sub
            if protocol == "holdout":
                cmd += ["--protocol", "holdout", "--holdout", HOLD]
            r = self.run(cmd, check=False, capture=True)
            head = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else f"failed: {r.stderr[-300:]}"
            lines.append(f"{protocol:8s} {title:36s} {head}")
        txt = "\n".join(lines)
        if txt:
            txt = ("Paired bootstrap over trajectories: diff = B - A in Avg (negative = B better), 95% CI\n" + txt)
            open(os.path.join(self.res, "paired_tests.txt"), "w", encoding="utf-8").write(txt + "\n")
            print("\n" + txt)
        return txt

    def step_figures(self):
        from trajpred import viz
        figs = []
        curves = {LABELS[n]: self.run_dir("official", n) for n in LABELS if os.path.exists(self.run_dir("official", n))}
        if curves:
            figs.append(viz.plot_curves(curves, os.path.join(self.res, "fig_curves_official.png"),
                                        "Validation Avg during training (official split)"))
        tables = {}
        for protocol, title in (("official", "official split"), ("holdout", "held-out videos")):
            _, rows = self.table(protocol)
            if rows:
                tables[title] = [(lab, m["ALL"]["Avg"], m["ALL"]["ADE"], m["ALL"]["FDE"]) for lab, _, m in rows]
        if tables:
            figs.append(viz.plot_comparison(tables, os.path.join(self.res, "fig_comparison.png")))
        _, rows = self.table("official")
        keep = [r for r in rows if r[1] in ("const_vel", "indiv_tf", "social_lstm_vel", "social_tf")]
        if keep:
            figs.append(viz.plot_per_source({lab: {s: m[s]["Avg"] for s in SOURCES if s in m} for lab, _, m in keep},
                                            os.path.join(self.res, "fig_per_source_official.png"),
                                            "Avg per source dataset (official split)"))
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = os.path.join(self.run_dir("official", "social_tf"), "best.pt")
        if os.path.exists(ck):
            figs.append(viz.plot_examples(ck, self.data, os.path.join(self.res, "fig_examples_official.png"), n=12, seed=1,
                                          max_files_per_source=2 if self.smoke else None, device=dev))
        ck = os.path.join(self.run_dir("holdout", "social_tf"), "best.pt")
        if os.path.exists(ck):
            figs.append(viz.plot_examples(ck, self.data, os.path.join(self.res, "fig_examples_holdout.png"), protocol="holdout",
                                          holdout=HOLD.split(","), n=12, seed=2,
                                          max_files_per_source=2 if self.smoke else None, device=dev))
        print("\nfigures:\n  " + "\n  ".join(figs))
        return figs

    def step_submission(self):
        name, protocol = ("social_tf", "full") if self.a.final_fit else ("social_tf", "official")
        ck = os.path.join(self.run_dir(protocol, name), "best.pt")
        sub = os.path.join(self.out, "submission")
        self.run([PY, "-m", "trajpred.predict", "--data-root", self.data, "--ckpt", ck, "--tta",
                  "--write-submission", sub] + self.sub)
        self.run([PY, "check_submission.py", "--data-root", self.data, "--submission", sub] + self.sub)
        zp = os.path.join(self.out, "submission.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(glob.glob(os.path.join(sub, "*.txt"))):
                z.write(f, os.path.basename(f))
        print(f"submission model: {protocol}/{name} (+ flip TTA) -> {zp}")
        return zp

    def step_bundle(self, parts, paired):
        """RESULTS_SUMMARY.md (paste-ready) and one zip with everything worth downloading."""
        md = [f"# Social Transformer - results (mode: {self.a.mode}, epochs: {self.ep})\n",
              "Metrics are TrajNet ADE / FDE / Avg = (ADE+FDE)/2 in metres (lower is better). "
              "'vs const. vel.' = Avg divided by the constant-velocity Avg. "
              "'full-study Avg' = the 30-epoch CPU study in RESULTS.md (a short run is expected to be slightly worse).\n"]
        md += parts
        if paired:
            md += ["\n```\n" + paired + "\n```"]
        md += ["\nFigures: fig_comparison.png, fig_curves_official.png, fig_per_source_official.png, "
               "fig_examples_official.png, fig_examples_holdout.png (folder results/)."]
        path = os.path.join(self.out, "RESULTS_SUMMARY.md")
        open(path, "w", encoding="utf-8").write("\n\n".join(md) + "\n")
        zp = os.path.join(self.out, "results_bundle.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(path, "RESULTS_SUMMARY.md")
            for f in glob.glob(os.path.join(self.res, "*")):
                z.write(f, "results/" + os.path.basename(f))
            sz = os.path.join(self.out, "submission.zip")
            if os.path.exists(sz):
                z.write(sz, "submission.zip")
            for f in glob.glob(os.path.join(self.runs, "*", "*", "*")):
                if os.path.basename(f) in ("best.pt", "log.csv", "final_metrics.json", "config.json"):
                    z.write(f, os.path.relpath(f, self.out))
        print(f"\nsummary: {path}\ndownload bundle: {zp} ({os.path.getsize(zp) // 1024} KB)")
        return zp

    def go(self):
        self.step_env()
        self.step_tests()
        self.step_train()
        parts = self.step_tables()
        paired = self.step_paired()
        self.step_figures()
        self.step_submission()
        self.step_bundle(parts, paired)
        print("\nDONE in %.1f min. Timings (s): %s" % (sum(self.timings.values()) / 60, self.timings))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder containing train/ validation/ test/")
    ap.add_argument("--out", default="results_colab")
    ap.add_argument("--mode", default="quick", choices=list(MODES))
    ap.add_argument("--tf-epochs", type=int, default=None, help="override the Transformer epochs of the chosen mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-comparison", action="store_true", help="only the final model (skip individual TF and LSTM baselines)")
    ap.add_argument("--no-holdout", action="store_true", help="skip the held-out-videos protocol")
    ap.add_argument("--final-fit", action="store_true", help="also fit on train+validation and use that model for the submission")
    ap.add_argument("--skip-tests", action="store_true")
    Pipeline(ap.parse_args(argv)).go()


if __name__ == "__main__":
    main()
