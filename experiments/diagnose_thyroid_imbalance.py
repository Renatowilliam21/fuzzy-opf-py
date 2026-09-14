"""Diagnoses whether Thyroid's low accuracy (even after normalization) comes
from class imbalance: OPF-family classifiers are documented to let minority
classes "conquer" disproportionate territory in the competition graph
relative to their true prevalence, which shows up as majority-class samples
being misclassified as minority classes.

Usage:
    python experiments/diagnose_thyroid_imbalance.py
"""

import sys
from collections import Counter
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
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    print("Class counts:")
    print("  train:", sorted(Counter(y_train.tolist()).items()))
    print("  test: ", sorted(Counter(y_test.tolist()).items()))

    model = FuzzyOPF(k_max=20, sigma=0.6, search_best_k=False)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = opf_accuracy(y_test, preds)
    print(f"\nOverall test accuracy: {acc:.4f}")

    classes = sorted(set(y_test.tolist()))
    print("\nPer-class recall (of samples truly in class X, % predicted as class X):")
    for c in classes:
        idx = [i for i, t in enumerate(y_test) if t == c]
        correct = sum(1 for i in idx if preds[i] == c)
        print(f"  class {c}: {correct}/{len(idx)} = {correct/len(idx):.4f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "        " + "".join(f"pred={c:<6}" for c in classes)
    print(header)
    for true_c in classes:
        row = []
        idx = [i for i, t in enumerate(y_test) if t == true_c]
        for pred_c in classes:
            count = sum(1 for i in idx if preds[i] == pred_c)
            row.append(f"{count:<10}")
        print(f"true={true_c:<3} " + "".join(row))


if __name__ == "__main__":
    main()
