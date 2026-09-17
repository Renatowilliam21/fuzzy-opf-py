"""Smoke tests for FuzzyOPF and the GA-based hyperparameter search.

Run with: pytest -q
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from fuzzy_opf import FuzzyOPF, genetic_search


@pytest.fixture()
def toy_dataset():
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    return X, y


def test_fit_predict_shapes(toy_dataset):
    X, y = toy_dataset
    X_train, y_train = X[:80], y[:80]
    X_test = X[80:]

    model = FuzzyOPF(k_max=5, sigma=0.6, search_best_k=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(X_test)
    assert set(preds).issubset(set(y_train))


def test_sigma_out_of_range_raises():
    with pytest.raises(ValueError):
        FuzzyOPF(sigma=2.0)


def test_invalid_membership_side_raises():
    with pytest.raises(ValueError):
        FuzzyOPF(membership_side="oops")


def test_membership_bounds(toy_dataset):
    X, y = toy_dataset
    model = FuzzyOPF(k_max=5, sigma=0.6, search_best_k=True)
    model.fit(X, y)

    memberships = [node.membership for node in model.subgraph.nodes]
    assert all(0.0 <= m <= 1.0 + 1e-9 for m in memberships)


def test_membership_side_target_vs_source_differ_or_match(toy_dataset):
    X, y = toy_dataset
    X_train, y_train = X[:80], y[:80]
    X_test = X[80:]

    target = FuzzyOPF(k_max=5, sigma=0.5, membership_side="target").fit(X_train, y_train)
    source = FuzzyOPF(k_max=5, sigma=0.5, membership_side="source").fit(X_train, y_train)

    # Both must run and produce valid predictions; they are not required to
    # match (that's exactly the discrepancy documented in README.md).
    assert len(target.predict(X_test)) == len(X_test)
    assert len(source.predict(X_test)) == len(X_test)


def test_sigma_one_is_close_to_standard_opf(toy_dataset):
    from opfython.math.general import opf_accuracy
    from opfython.models.supervised import SupervisedOPF

    X, y = toy_dataset
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]

    opf = SupervisedOPF()
    opf.fit(X_train, y_train)
    opf_acc = opf_accuracy(y_test, opf.predict(X_test))

    fuzzy = FuzzyOPF(k_max=5, sigma=1.0, search_best_k=True)
    fuzzy.fit(X_train, y_train)
    fuzzy_acc = opf_accuracy(y_test, fuzzy.predict(X_test))

    # sigma=1 -> membership degenerates to 1.0 for every node -> Fuzzy-OPF
    # should behave like standard OPF (paper, Sec. V-A).
    assert abs(float(opf_acc) - float(fuzzy_acc)) < 0.15


def test_genetic_search_returns_valid_result(toy_dataset):
    X, y = toy_dataset
    X_train, y_train = X[:60], y[:60]
    X_val, y_val = X[60:90], y[60:90]

    result = genetic_search(
        X_train, y_train, X_val, y_val,
        k_max_bounds=(1, 10), n_agents=4, n_iterations=3, seed=0,
    )

    assert 1 <= result.k_max <= 10
    assert 0.2 <= result.sigma <= 1.2
    assert 0.0 <= result.accuracy <= 1.0


def test_genetic_search_does_not_leak_global_rng_state(toy_dataset):
    """A seeded genetic_search() call must not change np.random's state for
    whatever code runs after it (see fuzzy_opf.tuning._seeded_global_rng)."""
    X, y = toy_dataset
    X_train, y_train = X[:60], y[:60]
    X_val, y_val = X[60:90], y[60:90]

    np.random.seed(123)
    before = np.random.get_state()

    genetic_search(
        X_train, y_train, X_val, y_val,
        k_max_bounds=(1, 10), n_agents=4, n_iterations=3, seed=0,
    )

    after = np.random.get_state()
    assert before[1].tolist() == after[1].tolist()


def test_no_predecessor_cycles_after_fit(toy_dataset):
    """Regression test: multiplying cost by membership (<= 1) can make a
    node's fuzzy cost fall below an already-finalized node's cost. Without
    the `heap.color[q] != BLACK` guard, this could rewrite an already-black
    node's `pred`, creating a cycle that hangs `mark_nodes`/`prune` forever
    (see model.py's `_grow_fuzzy_minimax_forest`)."""
    X, y = toy_dataset
    model = FuzzyOPF(k_max=5, sigma=0.3, search_best_k=True)  # low sigma stresses this path
    model.fit(X, y)

    n = model.subgraph.n_nodes
    for start in range(n):
        i = start
        steps = 0
        while model.subgraph.nodes[i].pred != -1:
            i = model.subgraph.nodes[i].pred
            steps += 1
            assert steps <= n, f"predecessor cycle detected starting at node {start}"


def test_predict_marks_nodes_and_does_not_hang(toy_dataset):
    """predict() must terminate promptly and mark relevant nodes (needed by
    prune()) -- this used to hang indefinitely before the fix above."""
    X, y = toy_dataset
    X_train, y_train = X[:80], y[:80]
    X_test = X[80:]

    model = FuzzyOPF(k_max=5, sigma=0.3, search_best_k=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(X_test)
    assert any(node.relevant for node in model.subgraph.nodes)


def test_prune_reduces_or_keeps_training_set(toy_dataset):
    X, y = toy_dataset
    X_train, y_train = X[:60].copy(), y[:60].copy()
    X_val, y_val = X[60:90].copy(), y[60:90].copy()

    model = FuzzyOPF(k_max=5, sigma=0.6, search_best_k=True)
    model.prune(X_train, y_train, X_val, y_val, n_iterations=2)

    assert model.subgraph.n_nodes <= 60
    assert model.subgraph.trained


def test_single_class_dataset_uses_prototype_fallback():
    """_find_prototypes has a fallback for datasets with only one class
    (no MST edge ever crosses a class boundary, so no prototype would
    otherwise be found) -- exercise it explicitly."""
    rng = np.random.default_rng(0)
    X = rng.random((30, 4))
    y = np.zeros(30, dtype=int)

    model = FuzzyOPF(k_max=5, sigma=0.6, search_best_k=False)
    model.fit(X, y)
    preds = model.predict(X[:5])

    assert all(p == 0 for p in preds)
    assert any(node.status == 1 for node in model.subgraph.nodes)  # c.PROTOTYPE == 1


def test_stratified_split_preserves_class_proportions():
    from fuzzy_opf.datasets import stratified_split

    rng = np.random.default_rng(0)
    n = 600
    y = rng.choice([0, 1, 2], size=n, p=[0.05, 0.1, 0.85])
    X = rng.random((n, 4))

    X1, X2, y1, y2 = stratified_split(X, y, percentage=0.6, random_state=0)

    assert X1.shape[0] == len(y1)
    assert X2.shape[0] == len(y2)
    assert len(y1) + len(y2) == n

    original = np.bincount(y) / n
    split1 = np.bincount(y1) / len(y1)
    split2 = np.bincount(y2) / len(y2)

    assert np.allclose(original, split1, atol=0.02)
    assert np.allclose(original, split2, atol=0.02)


def test_oversample_minority_classes():
    from fuzzy_opf.datasets import oversample_minority_classes

    rng = np.random.default_rng(0)
    n = 500
    y = rng.choice([0, 1, 2], size=n, p=[0.05, 0.1, 0.85])
    X = rng.random((n, 4))

    X_bal, y_bal = oversample_minority_classes(X, y, random_state=0)
    counts = np.bincount(y_bal)
    assert counts[0] == counts[1] == counts[2]  # fully balanced
    assert X_bal.shape[0] == y_bal.shape[0]

    # Originals are all kept (oversampling adds, never removes).
    assert X_bal.shape[0] >= X.shape[0]

    X_soft, y_soft = oversample_minority_classes(X, y, random_state=0, strategy=0.3)
    majority_count = np.bincount(y)[2]
    assert np.bincount(y_soft)[0] == round(majority_count * 0.3)


def test_smote_oversample():
    from fuzzy_opf.datasets import smote_oversample

    rng = np.random.default_rng(0)
    n = 500
    y = rng.choice([0, 1, 2], size=n, p=[0.05, 0.1, 0.85])
    X = rng.random((n, 4))

    X_bal, y_bal = smote_oversample(X, y, random_state=0)
    counts = np.bincount(y_bal)
    assert counts[0] == counts[1] == counts[2]
    assert X_bal.shape[0] == y_bal.shape[0]
    assert X_bal.shape[0] >= X.shape[0]

    # Every original row must still be present somewhere in the output.
    for row in X:
        assert np.any(np.all(np.isclose(row, X_bal), axis=1))

    # Synthetic points must be genuinely new positions, not exact copies of
    # any original row (the whole point of SMOTE vs. plain duplication).
    original_set = {tuple(row) for row in X}
    n_new = X_bal.shape[0] - X.shape[0]
    exact_dup_count = sum(1 for row in X_bal if tuple(row) in original_set)
    # exact_dup_count includes the n original rows themselves (kept as-is);
    # anything beyond that would mean a synthetic point exactly duplicated one.
    assert exact_dup_count == X.shape[0], (
        f"expected exactly {X.shape[0]} exact matches (the originals), got {exact_dup_count}"
    )
