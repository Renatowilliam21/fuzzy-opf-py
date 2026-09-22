"""Friedman test (are the 5 methods' mean accuracies significantly
different across datasets?) followed by Nemenyi post-hoc (which specific
pairs differ?) -- the standard non-parametric methodology (Demsar 2006)
for comparing multiple classifiers across multiple datasets, using each
dataset's mean accuracy per method as one "block".

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
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare

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

    # Average ranks (Demsar-style; rank 1 = best per dataset, i.e. highest
    # accuracy). pandas' rank() on the negated row gives rank 1 to the
    # largest original value.
    ranks = np.array([pd.Series(-row).rank(ascending=True).values for row in matrix])
    avg_ranks = ranks.mean(axis=0)

    print("\nAverage ranks (1 = best):")
    for m, r in sorted(zip(methods, avg_ranks), key=lambda x: x[1]):
        print(f"  {m}: {r:.2f}")

    # Friedman test: are the methods' accuracies (columns) significantly
    # different across datasets (blocks/rows)?
    stat, p = friedmanchisquare(*[matrix[:, i] for i in range(len(methods))])
    print(f"\nFriedman test: statistic={stat:.4f}  p={p:.4f}  "
          f"{'(significant at alpha=0.05)' if p < 0.05 else '(NOT significant at alpha=0.05)'}")

    if p >= 0.05:
        print("\nFriedman not significant -- post-hoc Nemenyi is not warranted "
              "(no evidence the methods differ overall); skipping.")
        return

    try:
        import scikit_posthocs as sp
    except ImportError:
        print("\nscikit-posthocs not installed -- run: pip install scikit-posthocs")
        return

    nemenyi = sp.posthoc_nemenyi_friedman(matrix)
    nemenyi.columns = methods
    nemenyi.index = methods

    print("\nNemenyi post-hoc pairwise p-values:")
    print(nemenyi.round(4).to_string())

    print("\nSignificant pairs (p < 0.05):")
    found_any = False
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j and nemenyi.iloc[i, j] < 0.05:
                print(f"  {m1} vs {m2}: p={nemenyi.iloc[i, j]:.4f}")
                found_any = True
    if not found_any:
        print("  (none -- Friedman found an overall difference, but no individual "
              "pair reaches significance at alpha=0.05 with this sample size)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
