"""Exports a dataset (using the SAME seed=0 60/20/20 split used throughout
this project's experiments) into LibOPF's native ASCII text format, for a
direct comparison against the original C reference implementation.

Two corrections vs. naively reusing our own thyroid.txt/data.txt files:
  1. LibOPF's text format has a HEADER line ("n_samples n_classes
     n_features") that our own format does not use.
  2. LibOPF's convention is 1-indexed labels (1, 2, 3, ...), confirmed by
     decompiling one of LibOPF's own bundled datasets (boat.dat) back to
     text via `opf2txt` and inspecting it -- our own converters use
     0-indexed labels (0, 1, 2, ...). Feeding 0-indexed labels directly
     would make LibOPF's own opf_Accuracy (which loops `for i=1;
     i<=nlabels`) silently skip whichever class is labeled 0 entirely.

Usage:
    python experiments/export_to_libopf.py --dataset data/raw/thyroid.txt \
        --normalize --out-dir /tmp/thyroid_libopf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzy_opf import load_dataset, stratified_split
from fuzzy_opf.datasets import standardize

from opfython.stream.splitter import split


def write_libopf_text(path: Path, X, y) -> None:
    n_samples, n_features = X.shape
    n_classes = len(set(y.tolist()))
    with open(path, "w") as f:
        f.write(f"{n_samples} {n_classes} {n_features}\n")
        for i in range(n_samples):
            label = int(y[i]) + 1  # 0-indexed -> 1-indexed, see module docstring
            features = " ".join(f"{v:.6f}" for v in X[i])
            f.write(f"{i} {label} {features}\n")


def run(dataset_path: str, out_dir: str, seed: int, normalize: bool, stratified: bool) -> None:
    X, y = load_dataset(dataset_path)
    splitter = stratified_split if stratified else split
    X_train, X_rest, y_train, y_rest = splitter(X, y, percentage=0.6, random_state=seed)
    X_val, X_test, y_val, y_test = splitter(X_rest, y_rest, percentage=0.5, random_state=seed)

    if normalize:
        X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_libopf_text(out_dir / "train.txt", X_train, y_train)
    write_libopf_text(out_dir / "test.txt", X_test, y_test)

    print(f"n_train={X_train.shape[0]}  n_test={X_test.shape[0]}  "
          f"n_features={X_train.shape[1]}  n_classes={len(set(y_train.tolist()))}")
    print(f"Written: {out_dir / 'train.txt'}")
    print(f"Written: {out_dir / 'test.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--stratified", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.out_dir, args.seed, args.normalize, args.stratified)
