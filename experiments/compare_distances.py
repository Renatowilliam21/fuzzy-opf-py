"""Compares Fuzzy-OPF's accuracy across different distance metrics, with
hyperparameters (k_max, sigma) held fixed -- isolates the effect of the
distance metric itself from hyperparameter search.

opfython registers ~47 distance metrics (see
opfython.math.distance.DISTANCES); this script runs a representative subset
by default (configurable via YAML), covering common families: Euclidean
variants, L1 (manhattan), Chebyshev, cosine/angular, and histogram/chi-square
style metrics.

Usage:
    python experiments/compare_distances.py --config experiments/configs/distances_boat.yaml

Writes results/<dataset>/distances_<timestamp>.csv with columns:
    distance, k_max, sigma, test_accuracy, fit_seconds
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzy_opf import FuzzyOPF, load_dataset, stratified_split
from fuzzy_opf.datasets import standardize

from opfython.math.general import opf_accuracy
from opfython.stream.splitter import split

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DISTANCES = [
    "log_squared_euclidean",  # FuzzyOPF's own default
    "euclidean",
    "squared_euclidean",
    "manhattan",
    "chebyshev",
    "cosine",
    "chi_squared",
    "bray_curtis",
    "canberra",
    "gaussian",
]


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())

    dataset_path = REPO_ROOT / config["dataset"]
    dataset_name = dataset_path.stem
    seed = config.get("seed", 0)
    k_max = config.get("k_max", 20)
    sigma = config.get("sigma", 0.6)
    search_best_k = config.get("search_best_k", False)
    membership_side = config.get("membership_side", "target")
    normalize = config.get("normalize", False)
    stratified = config.get("stratified", False)
    distances = config.get("distances", DEFAULT_DISTANCES)

    X, y = load_dataset(dataset_path)
    splitter = stratified_split if stratified else split
    X_train, X_rest, y_train, y_rest = splitter(X, y, percentage=0.6, random_state=seed)
    X_val, X_test, y_val, y_test = splitter(X_rest, y_rest, percentage=0.5, random_state=seed)

    if normalize:
        X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    print(f"n_train={X_train.shape[0]} n_test={X_test.shape[0]}, "
          f"k_max={k_max} sigma={sigma}, {len(distances)} distances")

    rows = []
    for distance in distances:
        t0 = time.time()
        try:
            model = FuzzyOPF(
                k_max=k_max, sigma=sigma, search_best_k=search_best_k,
                membership_side=membership_side, distance=distance,
            )
            model.fit(X_train, y_train)
            acc = opf_accuracy(y_test, model.predict(X_test))
            elapsed = time.time() - t0
            print(f"{distance:28s}: test_acc={acc:.4f} ({elapsed:.1f}s)")
            rows.append([distance, k_max, sigma, f"{acc:.4f}", f"{elapsed:.1f}"])
        except Exception as exc:
            # Some metrics assume non-negative / normalized inputs (e.g.
            # histogram-style chi_squared, bray_curtis) and can raise on
            # arbitrary real-valued features -- record the failure instead
            # of aborting the whole comparison.
            print(f"{distance:28s}: FAILED ({exc})")
            rows.append([distance, k_max, sigma, "FAILED", "-"])

    out_dir = REPO_ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"distances_{timestamp}.csv"

    with open(out_path, "w") as f:
        f.write("distance,k_max,sigma,test_accuracy,fit_seconds\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
