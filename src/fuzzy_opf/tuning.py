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
from typing import Callable

import numpy as np
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.optimizers.single_objective.evolutionary.ga import GA
from opytimizer.optimizers.single_objective.misc.cem import CEM
from opytimizer.optimizers.single_objective.swarm.pso import PSO
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
    n_evaluations: int = 0
    """Actual number of fitness evaluations performed. NOT the same as
    n_agents * n_iterations: population-based methods (GA, PSO, ...)
    evaluate the initial population AND every subsequent generation, so
    the real count is typically higher than that nominal product -- this
    field is the ground truth, measured by wrapping the fitness function
    with a counter, not inferred from the search configuration."""


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
    cluster_cache: dict | None = None,
) -> Callable[[np.ndarray], float]:
    """Builds the objective evaluated by the meta-heuristic (minimization).

    Args:
        cluster_cache: Optional dict shared across multiple search() calls
            (e.g. genetic_search, pso_search, random_search run back-to-back
            in the same comparison) so a k_max already clustered by one
            method is reused by the next instead of recomputed. Pass the
            same dict to each call to share it; omit for a private,
            call-local cache (the previous behaviour).
    """

    k_lo, k_hi = k_max_bounds
    cache = {} if cluster_cache is None else cluster_cache

    # Cache the *unsupervised* step per k_max: independent of sigma, so an
    # agent revisiting a k_max already tried by another agent/generation
    # (or by a different search method entirely, if `cluster_cache` is
    # shared) skips the expensive clustering step entirely.
    def _cached_cluster_model(k_max: int) -> UnsupervisedOPF:
        if k_max not in cache:
            min_k = 1 if search_best_k else k_max
            cluster_model = UnsupervisedOPF(min_k=min_k, max_k=k_max, distance=distance)
            cluster_model.fit(X_train, Y_train)
            cache[k_max] = cluster_model
        return cache[k_max]

    def fitness(x: np.ndarray) -> float:
        fitness.n_calls += 1
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

    # Ground-truth evaluation counter: population-based optimizers call
    # this once per agent for the initial population AND again after every
    # generation's update, so the real count is typically higher than the
    # nominal n_agents * n_iterations "budget" -- read after the search
    # completes (see _run_metaheuristic_search / random_search).
    fitness.n_calls = 0

    return fitness


def _run_metaheuristic_search(
    optimizer: Optimizer,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int],
    n_agents: int,
    n_iterations: int,
    search_best_k: bool,
    membership_side: str,
    distance: str,
    seed: int | None,
    cluster_cache: dict | None = None,
) -> TuningResult:
    """Shared core behind genetic_search/pso_search/cem_search.

    Every population-based search in this module (GA, PSO, CEM, ...) only
    differs in which ``opytimizer.core.Optimizer`` instance explores the
    same (k_max, sigma) space against the same fitness function -- so this
    is the one place that builds the SearchSpace/Function/Opytimizer trio
    and seeds/restores the global RNG. Adding a new metaheuristic is just a
    new one-line wrapper calling this with a different ``optimizer``.
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
        fitness_fn = _make_fitness(
            X_train, Y_train, X_val, Y_val, k_max_bounds, search_best_k, membership_side, distance,
            cluster_cache=cluster_cache,
        )
        function = Function(fitness_fn)

        task = Opytimizer(space, optimizer, function, save_agents=False)
        history = task.start(n_iterations=n_iterations)

    best_agent = space.best_agent
    best_k_max = int(np.clip(round(best_agent.position[0, 0]), *k_max_bounds))
    best_sigma = float(np.clip(best_agent.position[1, 0], *SIGMA_BOUNDS))
    best_acc = 1.0 - float(best_agent.fit)

    return TuningResult(
        k_max=best_k_max, sigma=best_sigma, accuracy=best_acc, history=history,
        n_evaluations=fitness_fn.n_calls,
    )


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
    cluster_cache: dict | None = None,
) -> TuningResult:
    """Finds (k_max, sigma) with a Genetic Algorithm instead of grid search.

    Args:
        X_train, Y_train: Training split.
        X_val, Y_val: Validation split used as the fitness signal (same role
            as the "evaluating" set in the original scripts).
        k_max_bounds: Search bounds for k_max (paper: 1..150).
        n_agents: Population size.
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
    return _run_metaheuristic_search(
        GA(), X_train, Y_train, X_val, Y_val, k_max_bounds,
        n_agents, n_iterations, search_best_k, membership_side, distance, seed,
        cluster_cache=cluster_cache,
    )


def pso_search(
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
    cluster_cache: dict | None = None,
) -> TuningResult:
    """Same search as genetic_search, but driven by Particle Swarm
    Optimization instead of a Genetic Algorithm. Same signature/semantics;
    see genetic_search's docstring for argument details."""
    return _run_metaheuristic_search(
        PSO(), X_train, Y_train, X_val, Y_val, k_max_bounds,
        n_agents, n_iterations, search_best_k, membership_side, distance, seed,
        cluster_cache=cluster_cache,
    )


def cem_search(
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
    cluster_cache: dict | None = None,
) -> TuningResult:
    """Same search as genetic_search, but driven by the Cross-Entropy
    Method: instead of evolving a population via crossover/mutation (GA) or
    velocity (PSO), it fits a probability distribution over the
    best-performing agents each iteration and resamples from it -- usually
    converging in fewer evaluations for low-dimensional continuous spaces
    like this one (k_max, sigma). Same signature/semantics; see
    genetic_search's docstring for argument details.

    KNOWN ISSUE (as of opytimizer 4.0.0 + numpy>=2.0): `CEM.compile()`
    assigns a shape-(1,) array into a scalar slot of `self.mean`/`self.std`,
    which numpy>=2.0 rejects (`TypeError: only 0-dimensional arrays can be
    converted to Python scalars`). This is a bug in opytimizer itself, not
    in this wrapper -- confirmed by reproducing the failing assignment in
    isolation, outside of fuzzy_opf entirely. Calling this function will
    currently raise that error; kept here so it starts working automatically
    once opytimizer patches it, without any change needed on our side.
    """
    return _run_metaheuristic_search(
        CEM(), X_train, Y_train, X_val, Y_val, k_max_bounds,
        n_agents, n_iterations, search_best_k, membership_side, distance, seed,
        cluster_cache=cluster_cache,
    )


def random_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int] = (1, 150),
    n_evaluations: int = 450,
    search_best_k: bool = True,
    membership_side: str = "target",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
) -> TuningResult:
    """Uniform random sampling of (k_max, sigma) -- the "negative control"
    for the comparison: any metaheuristic that fails to beat this on a
    matched evaluation budget is not earning its complexity.

    Args:
        n_evaluations: Total number of (k_max, sigma) points sampled --
            the random-search equivalent of `n_agents * n_iterations`.
        Other args: same as genetic_search.

    Returns:
        TuningResult with the best (k_max, sigma) found and its history as
        a plain list of (k_max, sigma, val_accuracy) tuples, one per
        evaluation, in sampling order (for convergence-curve plotting).
    """
    rng = np.random.default_rng(seed)
    k_lo, k_hi = k_max_bounds

    fitness = _make_fitness(
        X_train, Y_train, X_val, Y_val, k_max_bounds, search_best_k, membership_side, distance,
        cluster_cache=cluster_cache,
    )

    trace = []
    best_k_max, best_sigma, best_acc = None, None, -np.inf
    for _ in range(n_evaluations):
        k_max = int(rng.integers(k_lo, k_hi + 1))
        sigma = float(rng.uniform(*SIGMA_BOUNDS))

        x = np.array([[k_max], [sigma]], dtype=float)
        acc = 1.0 - fitness(x)
        trace.append((k_max, sigma, acc))

        if acc > best_acc:
            best_k_max, best_sigma, best_acc = k_max, sigma, acc

    return TuningResult(
        k_max=best_k_max, sigma=best_sigma, accuracy=best_acc, history=trace,
        n_evaluations=fitness.n_calls,
    )
