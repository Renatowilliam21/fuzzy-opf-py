# Copyright (c) 2026 Renato W. R. de Souza.
# Licensed under the Apache License, Version 2.0.

"""Dataset loading helpers for the Fuzzy-OPF experiments.

Supports the same three cases the original C project dealt with:
  * ``.dat``  -> original LibOPF binary format (as in ``data/*.dat`` in the
                 LibOPF_fuzzy repo). Converted to text on the fly via
                 ``opfython.utils.converter.opf2txt`` and cached next to it.
  * ``.txt``  -> whitespace-separated OPF text format (id label f1 f2 ...).
  * ``.csv``  -> comma-separated, same column layout.

Usage:
    >>> X, y = load_dataset("data/raw/boat.dat")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from opfython.stream import loader, parser
from opfython.utils import converter


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an OPF-formatted dataset (.dat, .txt or .csv) into (X, y)."""
    path = Path(path)

    if path.suffix == ".dat":
        txt_path = path.with_suffix(".txt")
        if not txt_path.exists():
            converter.opf2txt(path, txt_path)
        data = loader.load_txt(txt_path)
    elif path.suffix == ".txt":
        data = loader.load_txt(path)
    elif path.suffix == ".csv":
        data = loader.load_csv(path)
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix} (expected .dat, .txt or .csv)")

    if data is None:
        raise FileNotFoundError(f"Could not load dataset at {path}")

    X, y = parser.parse_loader(data)
    if X is None:
        raise ValueError(f"Could not parse dataset at {path}")

    return X, y


def standardize(X_train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Z-score standardization: fit mean/std on X_train, apply to all arrays.

    Matters most for datasets mixing features on very different scales
    (e.g. Thyroid: 15 binary attributes alongside 6 continuous ones) --
    opfython's distance metrics (log_squared_euclidean by default) are
    scale-sensitive, so a handful of large-magnitude continuous features
    can dominate the distance and drown out the binary ones entirely.

    Statistics are computed on X_train only and applied to every array
    passed in (val/test), avoiding leakage from val/test into the
    normalization itself.

    Args:
        X_train: Training features; mean/std are computed from this.
        *others: Any number of additional feature arrays (X_val, X_test,
            ...) to standardize with X_train's statistics.

    Returns:
        A tuple (X_train_scaled, *others_scaled), same order as given,
        X_train always first.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0  # constant columns (e.g. an always-0/1 binary
    # attribute with no variance in this particular split) would divide by
    # zero; leave them unscaled (subtracting the mean still centers them).

    scaled = [(X_train - mean) / std]
    for X in others:
        scaled.append((X - mean) / std)
    return tuple(scaled)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    percentage: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like ``opfython.stream.splitter.split``, but keeps each class's
    proportion the same in both output splits.

    ``opfython``'s own ``split()`` is a plain random permutation with no
    stratification. For a severely imbalanced dataset (e.g. Thyroid: ~92%
    of one class, ~1.4%/2.9% the other two), an unlucky draw can shift a
    minority class's representation between train/val/test noticeably --
    with only ~100-200 minority samples total, a few percentage points of
    imbalance in the draw is a real fraction of the class. Splitting each
    class separately at the same ``percentage`` and concatenating removes
    that source of noise.

    Args:
        X, y: Full dataset.
        percentage: Fraction to keep in the first split (e.g. 0.6 for a
            60/40 split). Same convention as opfython's split().
        random_state: Seed for reproducibility.

    Returns:
        (X_1, X_2, Y_1, Y_2), same order/convention as opfython's split().
    """
    rng = np.random.default_rng(random_state)

    idx_1, idx_2 = [], []
    for label in np.unique(y):
        class_idx = np.flatnonzero(y == label)
        rng.shuffle(class_idx)
        cut = round(len(class_idx) * percentage)
        idx_1.extend(class_idx[:cut])
        idx_2.extend(class_idx[cut:])

    idx_1 = np.array(idx_1)
    idx_2 = np.array(idx_2)
    rng.shuffle(idx_1)  # undo the class-grouped order within each split
    rng.shuffle(idx_2)

    return X[idx_1], X[idx_2], y[idx_1], y[idx_2]
