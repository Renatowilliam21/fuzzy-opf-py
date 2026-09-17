"""Compares per-class recall and confusion matrices WITH and WITHOUT
oversampling the minority classes in training -- direct evidence of
whether rebalancing actually helps the minority class ("1") that had only
34% recall in the original diagnosis (see diagnose_thyroid_imbalance.py).

Usage:
    python experiments/compare_balance.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzy_opf import FuzzyOPF, load_dataset, oversample_minority_classes, smote_oversample
from fuzzy_opf.datasets import standardize

from opfython.math.general import opf_accuracy
from opfython.stream.splitter import split

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate(X_train, y_train, X_test, y_test, k_max=20, sigma=0.6, label=""):
    model = FuzzyOPF(k_max=k_max, sigma=sigma, search_best_k=False)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = opf_accuracy(y_test, preds)
    classes = sorted(set(y_test.tolist()))

    print(f"\n=== {label} ===")
    print(f"n_train={X_train.shape[0]} (class counts: {sorted(Counter(y_train.tolist()).items())})")
    print(f"Overall test accuracy (OPF-balanced metric): {acc:.4f}")

    print("Per-class recall:")
    for c in classes:
        idx = [i for i, t in enumerate(y_test) if t == c]
        correct = sum(1 for i in idx if preds[i] == c)
        print(f"  class {c}: {correct}/{len(idx)} = {correct/len(idx):.4f}")

    print("Confusion matrix (rows=true, cols=predicted):")
    header = "        " + "".join(f"pred={c:<6}" for c in classes)
    print(header)
    for true_c in classes:
        row = []
        idx = [i for i, t in enumerate(y_test) if t == true_c]
        for pred_c in classes:
            count = sum(1 for i in idx if preds[i] == pred_c)
            row.append(f"{count:<10}")
        print(f"true={true_c:<3} " + "".join(row))

    return acc


def main():
    X, y = load_dataset(REPO_ROOT / "data/raw/thyroid.txt")
    X_train, X_rest, y_train, y_rest = split(X, y, percentage=0.6, random_state=0)
    X_val, X_test, y_val, y_test = split(X_rest, y_rest, percentage=0.5, random_state=0)
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    acc_before = evaluate(X_train, y_train, X_test, y_test, label="WITHOUT balancing (baseline)")

    X_train_dup, y_train_dup = oversample_minority_classes(X_train, y_train, random_state=0)
    acc_dup = evaluate(X_train_dup, y_train_dup, X_test, y_test, label="Plain oversampling (exact duplicates)")

    X_train_smote, y_train_smote = smote_oversample(X_train, y_train, random_state=0)
    acc_smote = evaluate(X_train_smote, y_train_smote, X_test, y_test, label="SMOTE (synthetic, interpolated)")

    print(f"\nOverall accuracy: baseline={acc_before:.4f}  "
          f"plain_oversample={acc_dup:.4f} ({acc_dup - acc_before:+.4f})  "
          f"smote={acc_smote:.4f} ({acc_smote - acc_before:+.4f})")


if __name__ == "__main__":
    main()
