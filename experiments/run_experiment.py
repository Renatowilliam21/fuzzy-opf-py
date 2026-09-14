"""Run the Fuzzy-OPF experimental protocol used in the paper:
20 runs, 60% train / 20% validation / 20% test, hyperparameters tuned on the
validation split, accuracy reported on the test split.

Usage:
    python experiments/run_experiment.py --config experiments/configs/boat.yaml

Each run appends a row to results/<dataset>/<timestamp>.csv with columns:
    run, method, k_max, sigma, val_accuracy, test_accuracy, fit_seconds
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

from opfython.math.general import opf_accuracy
from opfython.models.supervised import SupervisedOPF
from opfython.stream.splitter import split

from fuzzy_opf import FuzzyOPF, genetic_search, load_dataset
from fuzzy_opf.datasets import standardize

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())

    dataset_path = REPO_ROOT / config["dataset"]
    dataset_name = dataset_path.stem
    n_runs = config.get("n_runs", 20)
    tuning_cfg = config.get("tuning", {"method": "fixed", "k_max": 10, "sigma": 0.6})
    search_best_k = config.get("search_best_k", True)
    membership_side = config.get("membership_side", "target")
    base_seed = config.get("seed", 0)
    prune_cfg = config.get("prune")  # None (default) disables pruning entirely
    normalize = config.get("normalize", False)

    X, y = load_dataset(dataset_path)

    rows = []
    for run_id in range(n_runs):
        seed = base_seed + run_id

        X_train, X_rest, y_train, y_rest = split(X, y, percentage=0.6, random_state=seed)
        X_val, X_test, y_val, y_test = split(X_rest, y_rest, percentage=0.5, random_state=seed)

        if normalize:
            X_train, X_val, X_test = standardize(X_train, X_val, X_test)

        # --- Optional pruning: shrink the training set once per run, before
        # both the OPF baseline and the (possibly GA-driven) Fuzzy-OPF reuse
        # it. This is where the speedup actually pays off for large datasets
        # (e.g. Thyroid) -- pruning is O(n^2) itself, so doing it once per
        # run and then training many times on the smaller set is what makes
        # it worthwhile, as opposed to pruning inside every GA evaluation.
        if prune_cfg is not None:
            t0 = time.time()
            pruner = FuzzyOPF(
                k_max=prune_cfg.get("k_max", 20),
                sigma=prune_cfg.get("sigma", 0.6),
                search_best_k=search_best_k,
                membership_side=membership_side,
            )
            val_acc = pruner.prune_best(
                X_train, y_train, X_val, y_val,
                n_iterations=prune_cfg.get("n_iterations", 5),
            )
            prune_time = time.time() - t0
            n_before = X_train.shape[0]
            X_train = X_train[: pruner.subgraph.n_nodes]
            y_train = y_train[: pruner.subgraph.n_nodes]
            print(f"[{dataset_name}] run {run_id + 1}/{n_runs}: pruned "
                  f"{n_before} -> {X_train.shape[0]} samples "
                  f"(val_acc={val_acc:.4f}, {prune_time:.1f}s)")

        # --- Standard OPF (baseline) ---
        t0 = time.time()
        opf = SupervisedOPF()
        opf.fit(X_train, y_train)
        opf_time = time.time() - t0
        opf_acc = opf_accuracy(y_test, opf.predict(X_test))
        rows.append([run_id, "opf", "-", "-", "-", opf_acc, opf_time])

        # --- Fuzzy-OPF ---
        if tuning_cfg["method"] == "ga":
            result = genetic_search(
                X_train, y_train, X_val, y_val,
                k_max_bounds=tuple(tuning_cfg.get("k_max_bounds", (1, 150))),
                n_agents=tuning_cfg.get("n_agents", 15),
                n_iterations=tuning_cfg.get("n_iterations", 30),
                search_best_k=search_best_k,
                membership_side=membership_side,
                seed=seed,
            )
            k_max, sigma, val_acc = result.k_max, result.sigma, result.accuracy
        else:
            k_max, sigma = tuning_cfg["k_max"], tuning_cfg["sigma"]
            val_acc = "-"

        t0 = time.time()
        fuzzy = FuzzyOPF(k_max=k_max, sigma=sigma, search_best_k=search_best_k, membership_side=membership_side)
        fuzzy.fit(X_train, y_train)
        fuzzy_time = time.time() - t0
        fuzzy_acc = opf_accuracy(y_test, fuzzy.predict(X_test))
        rows.append([run_id, "fuzzy-opf", k_max, f"{sigma:.3f}", val_acc, fuzzy_acc, fuzzy_time])

        print(f"[{dataset_name}] run {run_id + 1}/{n_runs}: "
              f"OPF={opf_acc:.4f}  Fuzzy-OPF={fuzzy_acc:.4f}  (k={k_max}, sigma={sigma})")

    out_dir = REPO_ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{timestamp}.csv"

    header = "run,method,k_max,sigma,val_accuracy,test_accuracy,fit_seconds\n"
    with open(out_path, "w") as f:
        f.write(header)
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    _print_summary(rows)
    print(f"\nSaved: {out_path}")
    return out_path


def _print_summary(rows: list) -> None:
    for method in ("opf", "fuzzy-opf"):
        accs = np.array([r[5] for r in rows if r[1] == method], dtype=float)
        print(f"{method:>10}: mean acc = {accs.mean():.4f} +/- {accs.std():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    run(args.config)
