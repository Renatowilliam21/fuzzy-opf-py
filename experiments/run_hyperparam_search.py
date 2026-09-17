"""Compares hyperparameter search methods for Fuzzy-OPF under a matched
evaluation budget: GA, PSO, and random search (the "negative control" --
any method that doesn't beat it isn't earning its complexity).

Same train/val/test split for every method (fair comparison), optional
pruning applied once before all of them (so the comparison reflects search
quality, not who got lucky with a smaller/larger training set).

Usage:
    python experiments/run_hyperparam_search.py --config experiments/configs/hpsearch_thyroid.yaml

Writes results/<dataset>/hyperparam_search_<timestamp>.csv with columns:
    method, k_max, sigma, val_accuracy, test_accuracy, wall_seconds, n_evaluations
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import fuzzy_opf FIRST: it disables opfython's crash-prone per-module file
# logging (see fuzzy_opf/__init__.py) before any of the opfython imports
# below get a chance to trigger it.
from fuzzy_opf import FuzzyOPF, genetic_search, load_dataset, pso_search, random_search
from fuzzy_opf.datasets import standardize, stratified_split, apply_balance

from opfython.math.general import opf_accuracy
from opfython.stream.splitter import split

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())

    dataset_path = REPO_ROOT / config["dataset"]
    dataset_name = dataset_path.stem
    seed = config.get("seed", 0)
    search_best_k = config.get("search_best_k", False)
    membership_side = config.get("membership_side", "target")
    k_max_bounds = tuple(config.get("k_max_bounds", (1, 100)))
    budget = config.get("budget", 30)
    prune_cfg = config.get("prune")
    normalize = config.get("normalize", False)
    stratified = config.get("stratified", False)
    balance = config.get("balance")

    X, y = load_dataset(dataset_path)
    splitter = stratified_split if stratified else split
    X_train, X_rest, y_train, y_rest = splitter(X, y, percentage=0.6, random_state=seed)
    X_val, X_test, y_val, y_test = splitter(X_rest, y_rest, percentage=0.5, random_state=seed)

    if balance is not None:
        X_train, y_train = apply_balance(X_train, y_train, balance, random_state=seed)

    if normalize:
        X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    if prune_cfg is not None:
        t0 = time.time()
        pruner = FuzzyOPF(
            k_max=prune_cfg.get("k_max", 20),
            sigma=prune_cfg.get("sigma", 0.6),
            search_best_k=search_best_k,
            membership_side=membership_side,
        )
        pruner.prune_best(X_train, y_train, X_val, y_val, n_iterations=prune_cfg.get("n_iterations", 5))
        n_before = X_train.shape[0]
        X_train = np.array([node.features for node in pruner.subgraph.nodes])
        y_train = np.array([node.label for node in pruner.subgraph.nodes])
        print(f"Pruned training set: {n_before} -> {X_train.shape[0]} samples "
              f"({time.time() - t0:.1f}s)")

    print(f"n_train={X_train.shape[0]} n_val={X_val.shape[0]} n_test={X_test.shape[0]}, "
          f"budget={budget} evaluations per method")

    common = dict(
        k_max_bounds=k_max_bounds, search_best_k=search_best_k,
        membership_side=membership_side, seed=seed,
    )

    # n_agents x n_iterations must equal `budget` for a fair comparison
    # against random_search's n_evaluations. Defaults to a small population
    # (3 agents, override via config "n_agents") so nearly all of the
    # budget goes into ITERATIONS instead -- the opposite of the original
    # heuristic here, which capped n_agents at min(10, budget) and left
    # n_iterations=1 for any budget <= 10. With n_iterations=1, GA/PSO
    # never actually evolve (no crossover/mutation/velocity update ever
    # applies): they just evaluate an initial random population once and
    # stop, which is only a thin disguise for random search -- confirmed
    # empirically on Thyroid, where a second seed reversed which "method"
    # looked better, because both were really just sampling luck. Override
    # "n_agents" in the config to test whether a LARGER population escapes
    # local optima the default (3 agents) gets stuck in -- also confirmed
    # on Thyroid via a sigma sweep (see BACKLOG.md).
    n_agents = config.get("n_agents", 3)
    n_iterations = max(1, budget // n_agents)
    actual_budget = n_agents * n_iterations

    # Shared across all three methods below: when k_max is fixed (or its
    # range is narrow), a k_max already clustered by GA is reused by PSO
    # and Random instead of reclustered from scratch.
    cluster_cache: dict = {}

    methods = {
        "ga": lambda: genetic_search(X_train, y_train, X_val, y_val, n_agents=n_agents, n_iterations=n_iterations, cluster_cache=cluster_cache, **common),
        "pso": lambda: pso_search(X_train, y_train, X_val, y_val, n_agents=n_agents, n_iterations=n_iterations, cluster_cache=cluster_cache, **common),
        "random": lambda: random_search(X_train, y_train, X_val, y_val, n_evaluations=actual_budget, cluster_cache=cluster_cache, **common),
    }

    rows = []
    for name, fn in methods.items():
        t0 = time.time()
        result = fn()
        wall_seconds = time.time() - t0

        model = FuzzyOPF(
            k_max=result.k_max, sigma=result.sigma,
            search_best_k=search_best_k, membership_side=membership_side,
        )
        model.fit(X_train, y_train)
        test_acc = opf_accuracy(y_test, model.predict(X_test))

        print(f"{name:8s}: k_max={result.k_max:4d} sigma={result.sigma:.3f} "
              f"val_acc={result.accuracy:.4f} test_acc={test_acc:.4f} "
              f"({wall_seconds:.1f}s, {result.n_evaluations} real evaluations, "
              f"{wall_seconds / max(result.n_evaluations, 1):.1f}s/eval)")

        rows.append([name, result.k_max, f"{result.sigma:.4f}", f"{result.accuracy:.4f}",
                     f"{test_acc:.4f}", f"{wall_seconds:.1f}", result.n_evaluations])

    out_dir = REPO_ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"hyperparam_search_{timestamp}.csv"

    with open(out_path, "w") as f:
        f.write("method,k_max,sigma,val_accuracy,test_accuracy,wall_seconds,n_evaluations\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
