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


def oversample_minority_classes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int | None = None,
    strategy: str = "balance",
) -> tuple[np.ndarray, np.ndarray]:
    """Random oversampling (with replacement) of minority classes.

    Motivated by Thyroid's severe imbalance (~92%/3%/5%), where the
    minority class had only 34% recall despite ~93% instance-level
    accuracy overall: OPF-family classifiers let minority classes get
    "out-competed" for graph territory during training when they are
    heavily outnumbered. Duplicating minority-class training samples (with
    replacement) gives them proportionally more influence in the
    competition process, at the cost of a larger (and thus slower) training
    set -- apply AFTER a train/val/test split, never to val/test (that
    would let duplicated points leak between splits and inflate reported
    accuracy).

    This is plain random oversampling, not SMOTE (no synthetic
    interpolation) -- simplest possible baseline; revisit with SMOTE
    (e.g. via imbalanced-learn) if this proves insufficient.

    Args:
        X_train, y_train: Training split to rebalance.
        random_state: Seed for reproducibility.
        strategy: "balance" duplicates every non-majority class up to the
            majority class's count (fully balanced). A float in (0, 1]
            instead targets that fraction of the majority count (e.g. 0.5
            brings minorities up to half the majority's size -- softer
            rebalancing, smaller resulting training set).

    Returns:
        (X_resampled, y_resampled), shuffled (not grouped by class).
    """
    rng = np.random.default_rng(random_state)

    counts = {label: np.sum(y_train == label) for label in np.unique(y_train)}
    majority_count = max(counts.values())
    target = majority_count if strategy == "balance" else round(majority_count * float(strategy))

    idx_out = []
    for label, count in counts.items():
        class_idx = np.flatnonzero(y_train == label)
        idx_out.extend(class_idx)  # keep all originals
        n_extra = target - count
        if n_extra > 0:
            idx_out.extend(rng.choice(class_idx, size=n_extra, replace=True))

    idx_out = np.array(idx_out)
    rng.shuffle(idx_out)

    return X_train[idx_out], y_train[idx_out]


def smote_oversample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    random_state: int | None = None,
    strategy: str = "balance",
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE: oversample minority classes with synthetic (interpolated)
    points instead of exact duplicates.

    Motivated by ``oversample_minority_classes`` (plain duplication) making
    things *worse* on Thyroid: OPF competes by graph topology, and a
    duplicate sits at the exact same position as its original, so it
    changes nothing about who conquers nearby territory. SMOTE instead
    places each new point strictly BETWEEN a minority sample and one of its
    same-class nearest neighbors -- a genuinely new position that can shift
    the local graph topology near the class boundary, which plain
    duplication cannot do.

    For each synthetic point: pick a random original minority-class sample
    x_i, find its k nearest neighbors within the SAME class (excluding
    itself), pick one of them (x_nn) at random, and interpolate:
    x_new = x_i + lambda * (x_nn - x_i), lambda ~ Uniform(0, 1).

    Args:
        X_train, y_train: Training split to rebalance (never apply to
            val/test -- same rule as oversample_minority_classes).
        k_neighbors: Neighborhood size for interpolation. Automatically
            reduced for classes with fewer than k_neighbors+1 samples.
        random_state: Seed for reproducibility.
        strategy: Same as oversample_minority_classes: "balance" (fully
            balanced) or a float in (0, 1] (fraction of the majority
            count).

    Returns:
        (X_resampled, y_resampled): originals plus synthetic points,
        shuffled (not grouped by class).
    """
    rng = np.random.default_rng(random_state)

    counts = {label: np.sum(y_train == label) for label in np.unique(y_train)}
    majority_count = max(counts.values())
    target = majority_count if strategy == "balance" else round(majority_count * float(strategy))

    X_out = [X_train]
    y_out = [y_train]

    for label, count in counts.items():
        n_extra = target - count
        if n_extra <= 0:
            continue

        class_idx = np.flatnonzero(y_train == label)
        X_class = X_train[class_idx]

        if count == 1:
            # Nothing to interpolate with -- fall back to duplicating the
            # single available point (matches oversample_minority_classes'
            # behaviour for this degenerate case).
            synthetic = np.repeat(X_class, n_extra, axis=0)
        else:
            k = min(k_neighbors, count - 1)
            # Brute-force pairwise distances within the class -- classes
            # needing oversampling are, by definition, small, so this is
            # cheap even without a KD-tree.
            dists = np.linalg.norm(X_class[:, None, :] - X_class[None, :, :], axis=-1)
            np.fill_diagonal(dists, np.inf)
            neighbor_idx = np.argsort(dists, axis=1)[:, :k]

            base_choices = rng.integers(0, count, size=n_extra)
            neighbor_choices = neighbor_idx[base_choices, rng.integers(0, k, size=n_extra)]
            lambdas = rng.uniform(0.0, 1.0, size=(n_extra, 1))

            synthetic = X_class[base_choices] + lambdas * (X_class[neighbor_choices] - X_class[base_choices])

        X_out.append(synthetic)
        y_out.append(np.full(n_extra, label))

    X_resampled = np.vstack(X_out)
    y_resampled = np.concatenate(y_out)

    shuffle_idx = rng.permutation(len(y_resampled))
    return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


def apply_balance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    balance_config,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatches to oversample_minority_classes or smote_oversample based
    on a YAML-friendly config value, for use by the experiment scripts.

    Accepts either:
      - a bare strategy ("balance" or a float like 0.5): plain duplication
        via oversample_minority_classes (kept for backward compatibility
        with existing configs; SMOTE is the one that actually helped, see
        BACKLOG.md, so prefer the dict form below for new configs).
      - a dict: {"method": "oversample" | "smote", "strategy": ..., "k_neighbors": ...}
    """
    if isinstance(balance_config, dict):
        method = balance_config.get("method", "smote")
        strategy = balance_config.get("strategy", "balance")
        k_neighbors = balance_config.get("k_neighbors", 5)
    else:
        method = "oversample"
        strategy = balance_config

    if method == "smote":
        return smote_oversample(X_train, y_train, k_neighbors=k_neighbors, random_state=random_state, strategy=strategy)
    return oversample_minority_classes(X_train, y_train, random_state=random_state, strategy=strategy)


def stratified_kfold_indices(y: np.ndarray, n_splits: int, random_state: int | None = None):
    """Yields (train_idx, val_idx) for each of n_splits stratified folds.

    Each class's samples are shuffled and split into n_splits nearly-equal
    chunks independently, so every fold's validation slice keeps roughly
    the same class proportions as the full dataset -- same rationale as
    stratified_split, extended to k folds instead of one 60/40 split.

    Motivated by the Breast Tissue finding: a single ~20-sample validation
    split was too granular to distinguish between sigma values (accuracy
    was flat across the entire [0.2, 1.2] range). k-fold CV uses every
    sample as validation data exactly once, giving a more stable signal
    for hyperparameter selection on small datasets, at the cost of k
    trainings instead of 1 per candidate (k_max, sigma) evaluated.

    Args:
        y: Labels (used only to determine class membership; X is indexed
            by the caller using the same indices).
        n_splits: Number of folds (k).
        random_state: Seed for reproducibility.

    Yields:
        (train_idx, val_idx) index arrays, n_splits times.
    """
    rng = np.random.default_rng(random_state)

    fold_assignment = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        class_idx = np.flatnonzero(y == label)
        rng.shuffle(class_idx)
        # np.array_split handles counts not evenly divisible by n_splits.
        for fold, chunk in enumerate(np.array_split(class_idx, n_splits)):
            fold_assignment[chunk] = fold

    all_idx = np.arange(len(y))
    for fold in range(n_splits):
        val_idx = all_idx[fold_assignment == fold]
        train_idx = all_idx[fold_assignment != fold]
        yield train_idx, val_idx
