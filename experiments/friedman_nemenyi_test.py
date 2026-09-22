"""Friedman test (are the 5 methods' mean accuracies significantly
different across datasets?) followed by Nemenyi post-hoc (which specific
pairs differ?) -- the standard non-parametric methodology (Demsar 2006)
for comparing multiple classifiers across multiple datasets, using each
dataset's mean accuracy per method as one "block".

Uses Statys (https://github.com/gugarosa/statys) instead of a hand-rolled
scipy/scikit-posthocs implementation -- from the same author/group behind
opfython and opytimizer, already used throughout this project, so this
keeps the statistical tooling consistent with the rest of the toolchain.
Statys also gives a real critical-difference diagram (plot_critical_difference,
the standard Demsar-style figure), which our first hand-rolled version
didn't produce.

Builds directly on compare_baselines.py's output CSVs (one row per
run/method, already opf_accuracy-corrected -- see BACKLOG.md) -- takes the
mean accuracy per method per dataset as input, exactly the summary table
already assembled by hand in this project's investigation.

Usage:
    python experiments/friedman_nemenyi_test.py --config experiments/configs/friedman_all8.yaml

Config format:
    datasets:
      boat: results/boat/baselines_20260922T013658Z.csv
      cone-torus: results/cone-torus/baselines_20260922T013813Z.csv
      ...
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from statys import friedman, nemenyi, plot_critical_difference

REPO_ROOT = Path(__file__).resolve().parents[1]


def mean_accuracy_by_method(csv_path: str) -> dict[str, float]:
    """Reads a compare_baselines.py-style CSV and returns {method: mean
    test_accuracy over all its rows}."""
    accs_by_method: dict[str, list[float]] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            accs_by_method[row["method"]].append(float(row["test_accuracy"]))
    return {method: float(np.mean(accs)) for method, accs in accs_by_method.items()}


def run(config_path: str) -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    dataset_files: dict[str, str] = config["datasets"]

    # Build the (n_datasets x n_methods) matrix.
    per_dataset = {name: mean_accuracy_by_method(REPO_ROOT / path) for name, path in dataset_files.items()}
    methods = sorted({m for scores in per_dataset.values() for m in scores})
    datasets = sorted(per_dataset)

    matrix = np.array([[per_dataset[d][m] for m in methods] for d in datasets])

    print(f"{len(datasets)} datasets x {len(methods)} methods\n")
    print("Mean accuracy matrix:")
    header = "dataset".ljust(16) + "".join(m.ljust(16) for m in methods)
    print(header)
    for d, row in zip(datasets, matrix):
        print(d.ljust(16) + "".join(f"{v:.4f}".ljust(16) for v in row))

    # Friedman test (Statys): rows = blocks (datasets), columns = treatments
    # (methods). Returns ((chi_square, df), (F, (df1, df2))) -- the F
    # statistic is the Iman-Davenport correction, generally preferred over
    # the chi-square form for small numbers of treatments/blocks.
    (chi_square, df), (f_stat, (df1, df2)) = friedman(matrix)
    print(f"\nFriedman (Statys): chi-square={chi_square:.4f} (df={df})  "
          f"Iman-Davenport F={f_stat:.4f} (df1={df1}, df2={df2})")

    # Statys ranks SMALLER values as rank 1 by default; for accuracy
    # (higher is better), negate the matrix so the highest accuracy gets
    # rank 1, per the library's own documented convention.
    ranks, critical_difference = nemenyi(-matrix, alpha=0.05)
    avg_ranks = ranks.mean(axis=0) if ranks.ndim == 2 else ranks

    print(f"\nAverage ranks (1 = best), critical difference = {critical_difference:.4f}:")
    for m, r in sorted(zip(methods, avg_ranks), key=lambda x: x[1]):
        print(f"  {m}: {r:.2f}")

    print("\nPairs differing by more than the critical difference (significant):")
    found_any = False
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j and abs(avg_ranks[i] - avg_ranks[j]) > critical_difference:
                print(f"  {m1} vs {m2}: |rank diff|={abs(avg_ranks[i] - avg_ranks[j]):.2f} > CD={critical_difference:.2f}")
                found_any = True
    if not found_any:
        print(f"  (none -- every pair's rank difference is within the critical difference of {critical_difference:.2f})")

    out_dir = REPO_ROOT / "results" / "friedman_nemenyi"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fig_path = out_dir / f"critical_difference_{timestamp}.pdf"

    plot_critical_difference(avg_ranks, critical_difference, labels=methods, output=str(fig_path))
    print(f"\nCritical difference diagram saved to: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
