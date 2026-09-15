"""Sweeps sigma over a fixed grid (k_max held constant) to map the real
shape of the accuracy-vs-sigma curve -- cheap because the (expensive)
clustering step only depends on k_max, so it is computed ONCE and reused
across every sigma value in the sweep.

Motivated by GA/PSO converging to the same sigma~0.92 on Thyroid while
random search found a consistently better sigma~1.11 -- this sweep checks
directly whether GA/PSO are stuck in a local optimum, without relying on
another search algorithm's behaviour to infer the landscape's shape.

Usage:
    python experiments/sweep_sigma.py --config experiments/configs/sweep_thyroid.yaml

Writes results/<dataset>/sigma_sweep_<timestamp>.csv with columns:
    sigma, val_accuracy, test_accuracy, fit_seconds
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzy_opf import FuzzyOPF, load_dataset, stratified_split
from fuzzy_opf.datasets import standardize

from opfython.math.general import opf_accuracy
from opfython.models.unsupervised import UnsupervisedOPF
from opfython.stream.splitter import split

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())

    dataset_path = REPO_ROOT / config["dataset"]
    dataset_name = dataset_path.stem
    seed = config.get("seed", 0)
    k_max = config.get("k_max", 20)
    search_best_k = config.get("search_best_k", False)
    membership_side = config.get("membership_side", "target")
    normalize = config.get("normalize", False)
    stratified = config.get("stratified", False)

    sigma_start, sigma_end, sigma_step = config.get("sigma_range", [0.2, 1.2, 0.1])
    sigmas = list(np.round(np.arange(sigma_start, sigma_end + 1e-9, sigma_step), 4))

    X, y = load_dataset(dataset_path)
    splitter = stratified_split if stratified else split
    X_train, X_rest, y_train, y_rest = splitter(X, y, percentage=0.6, random_state=seed)
    X_val, X_test, y_val, y_test = splitter(X_rest, y_rest, percentage=0.5, random_state=seed)

    if normalize:
        X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    print(f"n_train={X_train.shape[0]} n_val={X_val.shape[0]} n_test={X_test.shape[0]}, "
          f"k_max={k_max} (fixed), {len(sigmas)} sigma values: {sigmas}")

    # Clustering depends only on k_max -- compute once, reuse for every
    # sigma value below (this is what makes the sweep cheap).
    t0 = time.time()
    min_k = 1 if search_best_k else k_max
    cluster_model = UnsupervisedOPF(min_k=min_k, max_k=k_max, distance="log_squared_euclidean")
    cluster_model.fit(X_train, y_train)
    print(f"Clustering (k_max={k_max}) done once in {time.time() - t0:.1f}s, reused below.\n")

    rows = []
    for sigma in sigmas:
        t0 = time.time()
        model = FuzzyOPF(k_max=k_max, sigma=sigma, search_best_k=search_best_k, membership_side=membership_side)
        model.fit(X_train, y_train, precomputed_cluster_model=cluster_model)

        val_acc = opf_accuracy(y_val, model.predict(X_val))
        test_acc = opf_accuracy(y_test, model.predict(X_test))
        elapsed = time.time() - t0

        print(f"sigma={sigma:.2f}: val_acc={val_acc:.4f} test_acc={test_acc:.4f} ({elapsed:.1f}s)")
        rows.append([sigma, f"{val_acc:.4f}", f"{test_acc:.4f}", f"{elapsed:.1f}"])

    out_dir = REPO_ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"sigma_sweep_{timestamp}.csv"

    with open(out_path, "w") as f:
        f.write("sigma,val_accuracy,test_accuracy,fit_seconds\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    best = max(rows, key=lambda r: float(r[1]))
    print(f"\nBest by val_accuracy: sigma={best[0]} (val={best[1]}, test={best[2]})")
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
