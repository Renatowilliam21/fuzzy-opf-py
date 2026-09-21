"""Paired Wilcoxon signed-rank test comparing OPF vs. Fuzzy-OPF test
accuracy, using the results already saved by run_experiment.py (same
train/val/test split per run for both methods, so pairing by run number
is a fair paired comparison, not an assumption).

Gives formal statistical significance to "Fuzzy-OPF >= OPF" beyond just
comparing mean +/- std, and reports an effect size (matched-pairs r =
|z|/sqrt(N)) since statistical significance alone doesn't say whether a
gain is practically meaningful -- relevant here, since several datasets in
this project showed tiny absolute gains (e.g. Data1: 0.9922 -> 0.9931).

Usage:
    python experiments/wilcoxon_test.py --csv results/data1/20260916T133940Z.csv
    python experiments/wilcoxon_test.py --csv results/thyroid/*.csv   (accepts a glob)
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_paired_accuracies(csv_path: str) -> tuple[list[float], list[float]]:
    """Reads a run_experiment.py-style CSV and returns (opf_accs,
    fuzzy_accs), paired by run number (both lists in matching run order)."""
    opf_by_run: dict[int, float] = {}
    fuzzy_by_run: dict[int, float] = {}

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            run_id = int(row["run"])
            acc = float(row["test_accuracy"])
            if row["method"] == "opf":
                opf_by_run[run_id] = acc
            elif row["method"] == "fuzzy-opf":
                fuzzy_by_run[run_id] = acc

    common_runs = sorted(set(opf_by_run) & set(fuzzy_by_run))
    if len(common_runs) < len(opf_by_run) or len(common_runs) < len(fuzzy_by_run):
        print(f"  WARNING: {csv_path} has unpaired rows "
              f"(opf: {len(opf_by_run)}, fuzzy-opf: {len(fuzzy_by_run)}); "
              f"using only the {len(common_runs)} runs present in both.")

    opf_accs = [opf_by_run[r] for r in common_runs]
    fuzzy_accs = [fuzzy_by_run[r] for r in common_runs]
    return opf_accs, fuzzy_accs


def run_test(csv_path: str) -> None:
    opf_accs, fuzzy_accs = load_paired_accuracies(csv_path)
    n = len(opf_accs)

    print(f"\n=== {csv_path} ===")
    print(f"n_pairs={n}  opf_mean={np.mean(opf_accs):.4f}  fuzzy_mean={np.mean(fuzzy_accs):.4f}  "
          f"diff_mean={np.mean(fuzzy_accs) - np.mean(opf_accs):+.4f}")

    diffs = np.array(fuzzy_accs) - np.array(opf_accs)
    n_wins = int(np.sum(diffs > 0))
    n_losses = int(np.sum(diffs < 0))
    n_ties = int(np.sum(diffs == 0))
    print(f"Fuzzy-OPF better: {n_wins}/{n}  |  OPF better: {n_losses}/{n}  |  tied: {n_ties}/{n}")

    if n_ties == n:
        print("All pairs tied (zero variance) -- Wilcoxon is undefined here, skipping.")
        return

    try:
        result = wilcoxon(fuzzy_accs, opf_accs, method="asymptotic", alternative="two-sided")
        r = abs(result.zstatistic) / np.sqrt(n)
        size = "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"

        print(f"Wilcoxon signed-rank: statistic={result.statistic:.1f}  z={result.zstatistic:.3f}  "
              f"p={result.pvalue:.4f}  {'(significant at alpha=0.05)' if result.pvalue < 0.05 else '(NOT significant at alpha=0.05)'}")
        print(f"Effect size r={r:.3f} ({size})")
    except ValueError as exc:
        print(f"Wilcoxon could not be computed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to a results CSV, or a glob pattern (quoted).")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.csv)) if any(c in args.csv for c in "*?[") else [args.csv]
    if not paths:
        raise SystemExit(f"No files matched: {args.csv}")

    for path in paths:
        run_test(path)
