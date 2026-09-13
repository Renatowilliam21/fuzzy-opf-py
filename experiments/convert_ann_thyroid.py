"""Converts the UCI "ann-thyroid" files (ann-train.data + ann-test.data) into
the OPF text format expected by fuzzy_opf.load_dataset (id label f1 ... fn,
labels zero-based and sequential).

The UCI files are whitespace-separated, one sample per line, with the class
label as the LAST column (1-indexed: 1, 2 or 3) -- the opposite convention
from the OPF text format (id, label, features), so this can't just be
renamed/reused directly like boat.dat/cone-torus.dat were.

The paper re-splits the full dataset 60/20/20 for its own protocol rather
than using UCI's original train/test split, so this script concatenates
both files into a single dataset before writing it out.

Usage:
    python experiments/convert_ann_thyroid.py \
        --train "data/thyroid+disease/ann-train.data" \
        --test "data/thyroid+disease/ann-test.data" \
        --out data/raw/thyroid.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert(train_path: str, test_path: str, out_path: str) -> None:
    train = np.loadtxt(train_path)
    test = np.loadtxt(test_path)
    data = np.vstack([train, test])

    features = data[:, :-1]
    raw_labels = data[:, -1].astype(int)

    # Map whatever original label values appear (e.g. 1, 2, 3) to zero-based,
    # sequential integers, as required by opfython's parser.
    unique_labels = np.sort(np.unique(raw_labels))
    label_map = {orig: new for new, orig in enumerate(unique_labels)}
    labels = np.array([label_map[v] for v in raw_labels])

    ids = np.arange(1, len(labels) + 1).reshape(-1, 1)
    out = np.hstack([ids, labels.reshape(-1, 1), features])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_features = features.shape[1]
    np.savetxt(out_path, out, fmt="%d %d" + " %.6f" * n_features)

    print(f"{len(labels)} samples, {n_features} features, "
          f"{len(unique_labels)} classes (original labels {unique_labels.tolist()} "
          f"-> 0..{len(unique_labels) - 1}).")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to ann-train.data")
    parser.add_argument("--test", required=True, help="Path to ann-test.data")
    parser.add_argument("--out", default="data/raw/thyroid.txt")
    args = parser.parse_args()
    convert(args.train, args.test, args.out)
