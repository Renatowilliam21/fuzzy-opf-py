"""Compares Fuzzy-OPF against external, modern baselines outside the OPF
family -- SVM (RBF kernel), Random Forest, k-NN (distance-weighted), and
XGBoost -- addressing a specific gap this project had: every prior
comparison was Fuzzy-OPF vs. plain OPF, never against classifiers from
other families. Each baseline gets its own light hyperparameter search
(GridSearchCV, 3-fold CV on the training split) per run, so this is not a
strawman comparison against untuned defaults.

Usage:
    python experiments/compare_baselines.py --config experiments/configs/baselines_boat.yaml

Writes results/<dataset>/baselines_<timestamp>.csv with columns:
    run, method, test_accuracy, fit_seconds, best_params
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzy_opf import FuzzyOPF, genetic_search, load_dataset, stratified_split
from fuzzy_opf.datasets import standardize

from opfython.math.general import opf_accuracy
from opfython.stream.splitter import split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]

PARAM_GRIDS = {
    "svm_rbf": (
        SVC,
        {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1]},
    ),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    ),
    "knn": (
        KNeighborsClassifier,
        {"n_neighbors": [3, 5, 7, 9, 15], "weights": ["distance"]},
    ),
    "xgboost": (
        XGBClassifier,
        {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
    ),
}


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())

    dataset_path = REPO_ROOT / config["dataset"]
    dataset_name = dataset_path.stem
    n_runs = config.get("n_runs", 20)
    base_seed = config.get("seed", 0)
    normalize = config.get("normalize", False)
    stratified = config.get("stratified", False)
    methods = config.get("methods", list(PARAM_GRIDS.keys()))

    # Fuzzy-OPF's own hyperparameters -- fixed (not re-searched here) to
    # whatever this project already validated for the dataset, matching
    # the fixed-hyperparameter protocol used for the Wilcoxon test on
    # Thyroid (see BACKLOG.md) rather than paying for a fresh GA search on
    # every one of the n_runs splits.
    fuzzy_k_max = config.get("fuzzy_k_max", 20)
    fuzzy_sigma = config.get("fuzzy_sigma", 0.6)
    search_best_k = config.get("search_best_k", False)
    membership_side = config.get("membership_side", "target")

    X, y = load_dataset(dataset_path)
    splitter = stratified_split if stratified else split

    rows = []
    for run_id in range(n_runs):
        seed = base_seed + run_id
        X_train, X_rest, y_train, y_rest = splitter(X, y, percentage=0.6, random_state=seed)
        X_val, X_test, y_val, y_test = splitter(X_rest, y_rest, percentage=0.5, random_state=seed)

        if normalize:
            X_train, X_val, X_test = standardize(X_train, X_val, X_test)

        # --- Fuzzy-OPF (fixed hyperparameters) ---
        t0 = time.time()
        fuzzy = FuzzyOPF(k_max=fuzzy_k_max, sigma=fuzzy_sigma, search_best_k=search_best_k, membership_side=membership_side)
        fuzzy.fit(X_train, y_train)
        fuzzy_acc = opf_accuracy(y_test, fuzzy.predict(X_test))
        fuzzy_time = time.time() - t0
        print(f"[run {run_id}] fuzzy-opf: acc={fuzzy_acc:.4f} ({fuzzy_time:.1f}s)")
        rows.append([run_id, "fuzzy-opf", f"{fuzzy_acc:.4f}", f"{fuzzy_time:.1f}", "-"])

        # --- External baselines, each with its own GridSearchCV ---
        for name in methods:
            estimator_cls, grid = PARAM_GRIDS[name]
            t0 = time.time()
            search = GridSearchCV(estimator_cls(), grid, cv=3, n_jobs=-1)
            search.fit(X_train, y_train)
            preds = search.predict(X_test)
            # opf_accuracy (not plain accuracy) for a fair, apples-to-apples
            # comparison against Fuzzy-OPF above -- see BACKLOG.md: using
            # plain accuracy here while Fuzzy-OPF used opf_accuracy (a
            # class-balanced measure) produced a spurious ~20pp gap on
            # Thyroid that had nothing to do with either classifier's real
            # performance, only with comparing two different metrics.
            acc = opf_accuracy(y_test, preds)
            elapsed = time.time() - t0
            print(f"[run {run_id}] {name}: acc={acc:.4f} ({elapsed:.1f}s) best_params={search.best_params_}")
            rows.append([run_id, name, f"{acc:.4f}", f"{elapsed:.1f}", str(search.best_params_).replace(",", ";")])

    out_dir = REPO_ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"baselines_{timestamp}.csv"

    with open(out_path, "w") as f:
        f.write("run,method,test_accuracy,fit_seconds,best_params\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
