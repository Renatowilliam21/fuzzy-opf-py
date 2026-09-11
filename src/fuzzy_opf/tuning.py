# Copyright (c) 2026 Renato W. R. de Souza.
# Licensed under the Apache License, Version 2.0.

"""Hyperparameter search for Fuzzy-OPF using opytimizer.

Replaces the exhaustive grid search hardcoded in ``fuzzy_validation_opf.c``
(16 values of k_max x 6 values of sigma = 96 full retrainings, each one
redoing the O(n^2)/O(n^3) clustering step from scratch even though
clustering does not depend on sigma at all).

Two things are addressed here:
  1. A meta-heuristic (e.g. Genetic Algorithm) explores (k_max, sigma)
     instead of the full Cartesian grid, which is exactly what was
     suggested for future work in the paper's conclusion.
  2. Clustering results are cached per k_max, so agents that land on an
     already-seen k_max only pay for the (cheap) membership + fuzzy
     training step, not for re-clustering.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.single_objective.evolutionary.ga import GA
from opytimizer.spaces.search import SearchSpace

from opfython.math.general import opf_accuracy
from opfython.models.unsupervised import UnsupervisedOPF

from .model import FuzzyOPF

# sigma bounds follow the paper's [0.2, 1.2] range.
SIGMA_BOUNDS = (0.2, 1.2)


@dataclass
class TuningResult:
    k_max: int
    sigma: float
    accuracy: float
    history: object


@contextmanager
def _seeded_global_rng(seed: int | None):
    """Seed NumPy's global RNG for the block, then restore the prior state.

    ``opytimizer`` calls ``np.random.rand()`` directly (module-level, not an
    injectable ``Generator``), so reproducing a GA run means seeding the
    global RNG. Restoring the previous state afterwards keeps that
    reproducibility from leaking into unrelated code that runs later in the
    same process (e.g. another dataset split in the calling script).
    """
    if seed is None:
        yield
        return

    prior_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(prior_state)


def _make_fitness(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int],
    search_best_k: bool,
    membership_side: str,
    distance: str,
) -> Callable[[np.ndarray], float]:
    """Builds the objective evaluated by the meta-heuristic (minimization)."""

    k_lo, k_hi = k_max_bounds

    # Cache the *unsupervised* step per k_max: independent of sigma, so an
    # agent revisiting a k_max already tried by another agent/generation
    # skips the expensive clustering step entirely.
    @lru_cache(maxsize=None)
    def _cached_cluster_model(k_max: int) -> UnsupervisedOPF:
        min_k = 1 if search_best_k else k_max
        cluster_model = UnsupervisedOPF(min_k=min_k, max_k=k_max, distance=distance)
        cluster_model.fit(X_train, Y_train)
        return cluster_model

    def fitness(x: np.ndarray) -> float:
        k_max = int(np.clip(round(x[0, 0]), k_lo, k_hi))
        sigma = float(np.clip(x[1, 0], *SIGMA_BOUNDS))

        cluster_model = _cached_cluster_model(k_max)

        model = FuzzyOPF(
            k_max=k_max,
            sigma=sigma,
            search_best_k=search_best_k,
            membership_side=membership_side,
            distance=distance,
        )
        # Reuses FuzzyOPF.fit's actual training code path (no duplicated
        # logic here) while skipping the redundant re-clustering.
        model.fit(X_train, Y_train, precomputed_cluster_model=cluster_model)

        preds = model.predict(X_val)
        acc = opf_accuracy(Y_val, preds)

        # opytimizer minimizes by default -> minimize the error.
        return 1.0 - acc

    return fitness


def genetic_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int] = (1, 150),
    n_agents: int = 15,
    n_iterations: int = 30,
    search_best_k: bool = True,
    membership_side: str = "target",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
) -> TuningResult:
    """Finds (k_max, sigma) with a Genetic Algorithm instead of grid search.

    Args:
        X_train, Y_train: Training split.
        X_val, Y_val: Validation split used as the fitness signal (same role
            as the "evaluating" set in the original scripts).
        k_max_bounds: Search bounds for k_max (paper: 1..150).
        n_agents: GA population size.
        n_iterations: Number of generations.
        search_best_k: Passed through to FuzzyOPF (min-cut k* search).
        membership_side: "target" (paper's Eq. 6) or "source" (legacy C).
        distance: Distance metric name registered in opfython, used for
            both the clustering and the supervised step.
        seed: Optional RNG seed for reproducibility. Only affects this call
            (the global NumPy RNG state is restored afterwards).

    Returns:
        TuningResult with the best (k_max, sigma) and validation accuracy.
    """

    n_variables = 2  # [k_max, sigma]
    lower_bound = [k_max_bounds[0], SIGMA_BOUNDS[0]]
    upper_bound = [k_max_bounds[1], SIGMA_BOUNDS[1]]

    with _seeded_global_rng(seed):
        # SearchSpace() randomizes the initial population on construction,
        # so it must be seeded too -- otherwise `seed` would only cover
        # task.start() and the run would not be fully reproducible.
        space = SearchSpace(
            n_agents=n_agents,
            n_variables=n_variables,
            n_objectives=1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        optimizer = GA()
        function = Function(
            _make_fitness(
                X_train, Y_train, X_val, Y_val, k_max_bounds, search_best_k, membership_side, distance
            )
        )

        task = Opytimizer(space, optimizer, function, save_agents=False)
        history = task.start(n_iterations=n_iterations)

    best_agent = space.best_agent
    best_k_max = int(np.clip(round(best_agent.position[0, 0]), *k_max_bounds))
    best_sigma = float(np.clip(best_agent.position[1, 0], *SIGMA_BOUNDS))
    best_acc = 1.0 - float(best_agent.fit)

    return TuningResult(k_max=best_k_max, sigma=best_sigma, accuracy=best_acc, history=history)
