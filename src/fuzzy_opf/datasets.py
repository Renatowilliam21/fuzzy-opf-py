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
