"""Smoke tests for FuzzyOPF and the GA-based hyperparameter search.

Run with: pytest -q
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from pathlib import Path

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


def test_stratified_kfold_indices():
    from fuzzy_opf.datasets import stratified_kfold_indices

    rng = np.random.default_rng(0)
    n = 106
    y = rng.choice([0, 1, 2], size=n, p=[0.2, 0.3, 0.5])

    folds = list(stratified_kfold_indices(y, n_splits=5, random_state=0))
    assert len(folds) == 5

    all_val = []
    for train_idx, val_idx in folds:
        assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
        all_val.extend(val_idx.tolist())

    # Every sample used as validation exactly once across all folds.
    assert sorted(all_val) == list(range(n))


def test_bayesian_search_returns_valid_result(toy_dataset):
    from fuzzy_opf import bayesian_search

    X, y = toy_dataset
    X_train, y_train = X[:60], y[:60]
    X_val, y_val = X[60:90], y[60:90]

    result = bayesian_search(
        X_train, y_train, X_val, y_val,
        k_max_bounds=(1, 10), n_trials=5, seed=0,
    )

    assert 1 <= result.k_max <= 10
    assert 0.2 <= result.sigma <= 1.2
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_evaluations == 5  # exact, unlike GA/PSO's inflation


def test_nsga2_search_returns_pareto_front(toy_dataset):
    from fuzzy_opf import nsga2_search

    X, y = toy_dataset
    X_train, y_train = X[:60], y[:60]
    X_val, y_val = X[60:90], y[60:90]

    result = nsga2_search(
        X_train, y_train, X_val, y_val,
        k_max_bounds=(1, 10), n_agents=6, n_iterations=3, search_best_k=False, seed=0,
    )

    assert len(result.points) >= 1
    for k_max, sigma, acc in result.points:
        assert 1 <= k_max <= 10
        assert 0.2 <= sigma <= 1.2
        assert 0.0 <= acc <= 1.0

    # Points must be sorted by k_max ascending (as documented).
    k_maxes = [p[0] for p in result.points]
    assert k_maxes == sorted(k_maxes)

    # No point should be dominated by another (Pareto-front property): for
    # any two points, it must NOT be that one has both <= k_max AND >= acc
    # than the other (with at least one strict).
    for i, (k1, _, a1) in enumerate(result.points):
        for j, (k2, _, a2) in enumerate(result.points):
            if i == j:
                continue
            dominates = (k2 <= k1 and a2 >= a1) and (k2 < k1 or a2 > a1)
            assert not dominates, f"point {i} is dominated by point {j}"


def test_opf_us_undersample():
    from fuzzy_opf.datasets import opf_us_undersample

    rng = np.random.default_rng(0)
    n = 200
    y = rng.choice([0, 1, 2], size=n, p=[0.15, 0.25, 0.6])
    X = rng.random((n, 5))
    X_train, y_train = X[:140], y[:140]
    X_val, y_val = X[140:], y[140:]

    X_us, y_us = opf_us_undersample(X_train, y_train, X_val, y_val, k_max=10, sigma=0.6)

    counts = np.bincount(y_us)
    assert counts[0] == counts[1] == counts[2]
    assert X_us.shape[0] == y_us.shape[0]
    # Undersampling only removes -- never adds or duplicates.
    assert X_us.shape[0] <= X_train.shape[0]
    for row in X_us:
        assert np.any(np.all(row == X_train, axis=1))


def test_invalid_membership_kind_raises():
    with pytest.raises(ValueError):
        FuzzyOPF(membership_kind="oops")


def test_membership_kinds_satisfy_boundary_conditions(toy_dataset):
    """Every membership_kind must give sigma at the lowest-density sample
    and 1.0 at the highest-density one (same boundary conditions as the
    paper's own Eq. 5) -- otherwise comparisons between shapes would be
    confounded by unequal ranges, not just curve shape."""
    X, y = toy_dataset
    sigma = 0.6

    for kind in ["linear", "quadratic", "cubic", "sigmoid"]:
        model = FuzzyOPF(k_max=5, sigma=sigma, search_best_k=False, membership_kind=kind)
        model.fit(X, y)
        memberships = [node.membership for node in model.subgraph.nodes]

        assert np.isclose(min(memberships), sigma, atol=1e-6), kind
        assert np.isclose(max(memberships), 1.0, atol=1e-6), kind


def test_ensemble_requires_at_least_one_config():
    from fuzzy_opf import EnsembleFuzzyOPF
    with pytest.raises(ValueError):
        EnsembleFuzzyOPF([])


def test_ensemble_single_member_matches_that_model(toy_dataset):
    from fuzzy_opf import EnsembleFuzzyOPF

    X, y = toy_dataset
    X_train, y_train = X[:80], y[:80]
    X_test = X[80:]

    cfg = {"k_max": 5, "sigma": 0.6, "search_best_k": False}
    single = FuzzyOPF(**cfg)
    single.fit(X_train, y_train)

    ensemble = EnsembleFuzzyOPF([cfg])
    ensemble.fit(X_train, y_train)

    assert ensemble.predict(X_test) == single.predict(X_test)


def test_ensemble_majority_vote(toy_dataset):
    from fuzzy_opf import EnsembleFuzzyOPF

    X, y = toy_dataset
    X_train, y_train = X[:80], y[:80]
    X_test = X[80:]

    ensemble = EnsembleFuzzyOPF([
        {"k_max": 5, "sigma": 0.4, "search_best_k": False},
        {"k_max": 10, "sigma": 0.7, "search_best_k": False},
        {"k_max": 15, "sigma": 1.0, "search_best_k": False},
    ])
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)

    assert len(preds) == len(X_test)
    assert set(preds).issubset(set(y_train))


def test_ensemble_from_pareto_front(toy_dataset):
    from fuzzy_opf import EnsembleFuzzyOPF, nsga2_search

    X, y = toy_dataset
    X_train, y_train = X[:60], y[:60]
    X_val, y_val = X[60:90], y[60:90]
    X_test = X[90:]

    pareto = nsga2_search(
        X_train, y_train, X_val, y_val,
        k_max_bounds=(1, 10), n_agents=6, n_iterations=3, search_best_k=False, seed=0,
    )
    ensemble = EnsembleFuzzyOPF.from_pareto_front(pareto, search_best_k=False)

    assert len(ensemble.members) == len(pareto.points)
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
    assert len(preds) == len(X_test)


def test_wilcoxon_load_paired_accuracies(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from wilcoxon_test import load_paired_accuracies

    csv_content = (
        "run,method,k_max,sigma,val_accuracy,test_accuracy,fit_seconds\n"
        "0,opf,-,-,-,0.80,1.0\n"
        "0,fuzzy-opf,20,0.6,0.8,0.82,2.0\n"
        "1,opf,-,-,-,0.85,1.0\n"
        "1,fuzzy-opf,20,0.6,0.8,0.84,2.0\n"
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(csv_content)

    opf_accs, fuzzy_accs = load_paired_accuracies(str(csv_path))
    assert opf_accs == [0.80, 0.85]
    assert fuzzy_accs == [0.82, 0.84]
