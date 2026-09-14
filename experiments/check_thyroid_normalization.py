"""Quick sanity check: does normalizing features fix Thyroid's suspiciously
low accuracy (0.69, worse than the trivial 92%-majority-class baseline)?

Single fit each way, no pruning, no search -- just to confirm the
hypothesis before repeating a multi-hour run.

Usage:
    python experiments/check_thyroid_normalization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opfython.math.general import opf_accuracy
from opfython.stream.splitter import split

from fuzzy_opf import FuzzyOPF, load_dataset
from fuzzy_opf.datasets import standardize

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    X, y = load_dataset(REPO_ROOT / "data/raw/thyroid.txt")
    X_train, X_rest, y_train, y_rest = split(X, y, percentage=0.6, random_state=0)
    X_val, X_test, y_val, y_test = split(X_rest, y_rest, percentage=0.5, random_state=0)

    print("Without normalization:")
    m1 = FuzzyOPF(k_max=20, sigma=0.6, search_best_k=False)
    m1.fit(X_train, y_train)
    acc1 = opf_accuracy(y_test, m1.predict(X_test))
    print(f"  test_accuracy = {acc1:.4f}")

    print("With normalization:")
    X_train_n, X_val_n, X_test_n = standardize(X_train, X_val, X_test)
    m2 = FuzzyOPF(k_max=20, sigma=0.6, search_best_k=False)
    m2.fit(X_train_n, y_train)
    acc2 = opf_accuracy(y_test, m2.predict(X_test_n))
    print(f"  test_accuracy = {acc2:.4f}")

    print(f"\nTrivial majority-class baseline should be ~0.92 for this dataset.")
    print(f"Difference: {acc2 - acc1:+.4f}")


if __name__ == "__main__":
    main()
