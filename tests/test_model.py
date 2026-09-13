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
