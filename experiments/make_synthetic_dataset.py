"""Generates a small synthetic dataset in OPF text format (id label f1 f2 ...)
for smoke-testing the pipeline without needing the original LibOPF datasets
(Boat, Cone-Torus, Thyroid, etc. -- see data/README.md on how to get those).

Usage:
    python experiments/make_synthetic_dataset.py
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(n_samples: int = 300, n_features: int = 4, n_classes: int = 3, seed: int = 0) -> None:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )

    out_path = REPO_ROOT / "data" / "raw" / "synthetic_quicktest.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = np.arange(1, n_samples + 1).reshape(-1, 1)
    labels = y.reshape(-1, 1)
    data = np.hstack([ids, labels, X])
    np.savetxt(out_path, data, fmt="%d %d" + " %.6f" * n_features)

    print(f"Synthetic dataset written to {out_path} ({n_samples} samples, {n_classes} classes).")


if __name__ == "__main__":
    main()
