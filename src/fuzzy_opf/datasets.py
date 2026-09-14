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
