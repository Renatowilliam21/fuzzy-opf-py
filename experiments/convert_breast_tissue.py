"""Converts the UCI "Breast Tissue" dataset (BreastTissue.xls, sheet "Data")
into the OPF text format expected by fuzzy_opf.load_dataset.

The file has two sheets ("Description" and "Data"); the "Data" sheet has
columns: Case #, Class, I0, PA500, HFS, DA, Area, A/DA, Max IP, DR, P.
"Class" is a categorical string (car, fad, mas, gla, con, adi -- 6 classes);
"Case #" is dropped (just a row index, not a feature).

Usage:
    python experiments/convert_breast_tissue.py \
        --input data/raw_sources/breast-tissue/BreastTissue.xls \
        --out data/raw/breast-tissue.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert(input_path: str, out_path: str, sheet_name: str = "Data") -> None:
    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    except ImportError as exc:
        raise ImportError(
            "Reading this .xls file needs the 'xlrd' package (legacy Excel "
            "format). Install it with: pip install xlrd"
        ) from exc

    # Drop the row-index column if present; keep "Class" + all numeric
    # feature columns as they are.
    for col in ("Case #", "Case#", "Unnamed: 0"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if "Class" not in df.columns:
        raise ValueError(
            f"Expected a 'Class' column in sheet '{sheet_name}', found: {list(df.columns)}"
        )

    raw_labels = df["Class"].astype(str).to_numpy()
    features = df.drop(columns=["Class"]).to_numpy(dtype=float)

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
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="data/raw/breast-tissue.txt")
    parser.add_argument("--sheet", default="Data", help="Excel sheet name (default: 'Data')")
    args = parser.parse_args()
    convert(args.input, args.out, args.sheet)
