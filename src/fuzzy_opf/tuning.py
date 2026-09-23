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
from opytimizer.optimizers.single_objective.evolutionary.de import DE
from opytimizer.optimizers.single_objective.population.gwo import GWO
from opytimizer.optimizers.multi_objective.evolutionary.nsga2 import NSGA2
from opytimizer.spaces.search import SearchSpace
from opytimizer.core.stopping import MaxIterations

from opfython.math.general import opf_accuracy
from opfython.models.unsupervised import UnsupervisedOPF

from .model import FuzzyOPF
from .datasets import stratified_kfold_indices

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


@dataclass
class ParetoResult:
    """Result of a multi-objective search (nsga2_search): a whole Pareto
    front instead of a single best point, since accuracy and
    computational cost trade off against each other -- there is no single
    "best" (k_max, sigma) once cost matters too, only a frontier of
    non-dominated choices.
    """
    points: list  # list of (k_max, sigma, accuracy) tuples, one per
    # non-dominated point on the front, sorted by k_max ascending.
    n_evaluations: int
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
    membership_kind: str,
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
            membership_kind=membership_kind,
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


def _make_cv_fitness(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    cv_folds: int,
    k_max_bounds: tuple[int, int],
    search_best_k: bool,
    membership_side: str,
    membership_kind: str,
    distance: str,
) -> Callable[[np.ndarray], float]:
    """Like _make_fitness, but scores each candidate (k_max, sigma) by mean
    accuracy over stratified k-fold cross-validation instead of a single
    held-out validation split.

    Motivated by the Breast Tissue finding: a single ~20-sample validation
    split was too granular to distinguish between sigma values at all
    (accuracy was completely flat across [0.2, 1.2]), which meant any
    search method was really just guessing among ties. k-fold CV uses
    every sample in the pool as validation data exactly once, giving a
    much more stable signal for hyperparameter selection -- worthwhile for
    small datasets, where the extra cost (cv_folds trainings per candidate
    instead of 1) is still cheap in absolute terms.

    NOTE: no clustering cache here (unlike _make_fitness). Each fold
    trains on a different sample subset, so a clustering fit on one fold
    isn't valid for another -- caching would need a (k_max, fold) key
    instead of just k_max. Not implemented for now since this path targets
    small, already-cheap datasets; revisit if used on something bigger.

    Args:
        X_pool, Y_pool: The full pool to run k-fold CV over (typically
            train+val combined -- keep a separate held-out test set
            outside this pool for the final, honest evaluation).
        cv_folds: Number of folds (k).
        Other args: same as _make_fitness.
    """
    k_lo, k_hi = k_max_bounds

    def fitness(x: np.ndarray) -> float:
        fitness.n_calls += 1
        k_max = int(np.clip(round(x[0, 0]), k_lo, k_hi))
        sigma = float(np.clip(x[1, 0], *SIGMA_BOUNDS))

        accs = []
        for train_idx, val_idx in stratified_kfold_indices(Y_pool, cv_folds, random_state=0):
            model = FuzzyOPF(
                k_max=k_max, sigma=sigma, search_best_k=search_best_k,
                membership_side=membership_side, membership_kind=membership_kind, distance=distance,
            )
            model.fit(X_pool[train_idx], Y_pool[train_idx])
            accs.append(opf_accuracy(Y_pool[val_idx], model.predict(X_pool[val_idx])))

        return 1.0 - float(np.mean(accs))

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
    membership_kind: str,
    distance: str,
    seed: int | None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
) -> TuningResult:
    """Shared core behind genetic_search/pso_search/cem_search.

    Every population-based search in this module (GA, PSO, CEM, ...) only
    differs in which ``opytimizer.core.Optimizer`` instance explores the
    same (k_max, sigma) space against the same fitness function -- so this
    is the one place that builds the SearchSpace/Function/Opytimizer trio
    and seeds/restores the global RNG. Adding a new metaheuristic is just a
    new one-line wrapper calling this with a different ``optimizer``.

    If cv_folds is given, X_val/Y_val are ignored entirely and X_train/
    Y_train is treated as the FULL pool to run stratified k-fold CV over
    (see _make_cv_fitness) -- pass your combined train+val data as
    X_train/Y_train in that case, with a separate held-out test set kept
    outside of this call.
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
        if cv_folds is not None:
            fitness_fn = _make_cv_fitness(
                X_train, Y_train, cv_folds, k_max_bounds, search_best_k, membership_side, membership_kind, distance,
            )
        else:
            fitness_fn = _make_fitness(
                X_train, Y_train, X_val, Y_val, k_max_bounds, search_best_k, membership_side, membership_kind, distance,
                cluster_cache=cluster_cache,
            )
        function = Function(fitness_fn)

        task = Opytimizer(space, optimizer, function, save_agents=False)
        history = task.start(stopping_criteria=MaxIterations(n_iterations))

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
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
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
        n_agents, n_iterations, search_best_k, membership_side, membership_kind, distance, seed,
        cluster_cache=cluster_cache, cv_folds=cv_folds,
    )


def de_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int] = (1, 150),
    n_agents: int = 15,
    n_iterations: int = 30,
    search_best_k: bool = True,
    membership_side: str = "target",
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
) -> TuningResult:
    """Finds (k_max, sigma) with Differential Evolution instead of grid
    search. Same interface and semantics as genetic_search/pso_search (see
    their docstrings) -- only the underlying opytimizer.Optimizer differs.

    NOTE: unlike GA/PSO, DE's mutation step (opytimizer's
    DE.update, eq. 1-4) samples 3 DISTINCT agents excluding the current
    one for every update, so it structurally requires n_agents >= 4 --
    with fewer, opytimizer crashes deep inside numpy with a cryptic
    "Cannot take a larger sample than population" error instead of a
    clear message. Validated here instead, since this project's smaller
    search-budget presets (e.g. n_agents=3, chosen for GA/PSO to still
    behave reasonably on a small budget) are BELOW DE's minimum.
    """
    if n_agents < 4:
        raise ValueError(
            f"Differential Evolution requires n_agents >= 4 (its mutation step samples "
            f"3 distinct agents excluding the current one each update), got n_agents={n_agents}. "
            f"This is a structural requirement of DE itself, not specific to this project -- "
            f"raise n_agents to at least 4 when using de_search (directly, or via "
            f"run_hyperparam_search.py's config)."
        )
    return _run_metaheuristic_search(
        DE(), X_train, Y_train, X_val, Y_val, k_max_bounds,
        n_agents, n_iterations, search_best_k, membership_side, membership_kind, distance, seed,
        cluster_cache=cluster_cache, cv_folds=cv_folds,
    )


def gwo_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int] = (1, 150),
    n_agents: int = 15,
    n_iterations: int = 30,
    search_best_k: bool = True,
    membership_side: str = "target",
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
) -> TuningResult:
    """Finds (k_max, sigma) with the Grey Wolf Optimizer instead of grid
    search. Same interface and semantics as genetic_search/pso_search (see
    their docstrings) -- only the underlying opytimizer.Optimizer differs.
    """
    return _run_metaheuristic_search(
        GWO(), X_train, Y_train, X_val, Y_val, k_max_bounds,
        n_agents, n_iterations, search_best_k, membership_side, membership_kind, distance, seed,
        cluster_cache=cluster_cache, cv_folds=cv_folds,
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
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
) -> TuningResult:
    """Same search as genetic_search, but driven by Particle Swarm
    Optimization instead of a Genetic Algorithm. Same signature/semantics;
    see genetic_search's docstring for argument details."""
    return _run_metaheuristic_search(
        PSO(), X_train, Y_train, X_val, Y_val, k_max_bounds,
        n_agents, n_iterations, search_best_k, membership_side, membership_kind, distance, seed,
        cluster_cache=cluster_cache, cv_folds=cv_folds,
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
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
) -> TuningResult:
    """Same search as genetic_search, but driven by the Cross-Entropy
    Method: instead of evolving a population via crossover/mutation (GA) or
    velocity (PSO), it fits a probability distribution over the
    best-performing agents each iteration and resamples from it -- usually
    converging in fewer evaluations for low-dimensional continuous spaces
    like this one (k_max, sigma). Same signature/semantics; see
    genetic_search's docstring for argument details.

    FIXED as of opytimizer>=5.0.0 (2026-09-23): previously (opytimizer
    4.0.0/4.1.0 + numpy>=2.0), `CEM.compile()` assigned a shape-(1,) array
    into a scalar slot of `self.mean`/`self.std`, which numpy>=2.0 rejects
    (`TypeError: only 0-dimensional arrays can be converted to Python
    scalars`) -- a bug in opytimizer itself (confirmed by reproducing the
    failing assignment in isolation, outside of fuzzy_opf entirely, and
    reported upstream: https://github.com/recogna-lab/opytimizer/issues/10).
    Confirmed working again with opytimizer 5.0.1.
    """
    return _run_metaheuristic_search(
        CEM(), X_train, Y_train, X_val, Y_val, k_max_bounds,
        n_agents, n_iterations, search_best_k, membership_side, membership_kind, distance, seed,
        cluster_cache=cluster_cache, cv_folds=cv_folds,
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
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
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

    if cv_folds is not None:
        fitness = _make_cv_fitness(
            X_train, Y_train, cv_folds, k_max_bounds, search_best_k, membership_side, membership_kind, distance,
        )
    else:
        fitness = _make_fitness(
            X_train, Y_train, X_val, Y_val, k_max_bounds, search_best_k, membership_side, membership_kind, distance,
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


def bayesian_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int] = (1, 150),
    n_trials: int = 30,
    search_best_k: bool = True,
    membership_side: str = "target",
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
    cv_folds: int | None = None,
) -> TuningResult:
    """Bayesian Optimization (Tree-structured Parzen Estimator, via Optuna)
    for (k_max, sigma).

    The natural candidate for the "sample-efficient, model-based" slot in
    the search-method comparison, after CEM turned out to be broken in the
    installed opytimizer version (see cem_search's docstring) -- outside
    the Recogna/opytimizer ecosystem (Optuna is a separate, independent
    dependency), but this is exactly the regime (few dimensions, expensive
    evaluations) where Bayesian methods are expected to need fewer
    evaluations than population-based search (GA/PSO) or random sampling
    to find a good point, by building a probabilistic model of the
    objective and choosing where to sample next instead of exploring
    blindly.

    Unlike genetic_search/pso_search, n_trials maps EXACTLY to
    n_evaluations (Optuna calls the objective once per trial, no
    population-inflation like GA/PSO's generational evaluation -- see
    TuningResult.n_evaluations' docstring), so budget comparisons against
    the other methods are direct here, no real/nominal distinction needed.

    Args:
        Same as genetic_search, except n_trials replaces n_agents/n_iterations
        (there's no population here, just a sequence of trials).

    Returns:
        TuningResult with the best (k_max, sigma) and validation accuracy.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    k_lo, k_hi = k_max_bounds

    if cv_folds is not None:
        raw_fitness = _make_cv_fitness(
            X_train, Y_train, cv_folds, k_max_bounds, search_best_k, membership_side, membership_kind, distance,
        )
    else:
        raw_fitness = _make_fitness(
            X_train, Y_train, X_val, Y_val, k_max_bounds, search_best_k, membership_side, membership_kind, distance,
            cluster_cache=cluster_cache,
        )

    def objective(trial: "optuna.Trial") -> float:
        k_max = trial.suggest_int("k_max", k_lo, k_hi)
        sigma = trial.suggest_float("sigma", *SIGMA_BOUNDS)
        x = np.array([[k_max], [sigma]], dtype=float)
        return raw_fitness(x)

    sampler = optuna.samplers.TPESampler(seed=seed)
    with _seeded_global_rng(seed):
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_k_max = int(study.best_params["k_max"])
    best_sigma = float(study.best_params["sigma"])
    best_acc = 1.0 - float(study.best_value)

    return TuningResult(
        k_max=best_k_max, sigma=best_sigma, accuracy=best_acc, history=study,
        n_evaluations=raw_fitness.n_calls,
    )


def nsga2_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    k_max_bounds: tuple[int, int] = (1, 150),
    n_agents: int = 20,
    n_iterations: int = 30,
    search_best_k: bool = True,
    membership_side: str = "target",
    membership_kind: str = "quadratic",
    distance: str = "log_squared_euclidean",
    seed: int | None = None,
    cluster_cache: dict | None = None,
) -> ParetoResult:
    """Multi-objective search (NSGA-II) trading off accuracy against
    computational cost, instead of picking one "best" (k_max, sigma).

    Every other search function in this module optimizes accuracy alone,
    implicitly treating k_max as "however big it needs to be" -- but the
    paper's own "Computational Burden" discussion, and everything this
    project measured about Thyroid (clustering cost scaling with k_max,
    hours-long runs), says cost is a real second objective, not
    negligible. NSGA-II returns the whole Pareto front of (k_max, sigma)
    choices: for each point on it, no other point achieves both equal-or-
    better accuracy AND equal-or-lower k_max -- so picking a smaller k_max
    from the front is a genuine, quantified trade-off, not a guess.

    Objectives (both minimized): (1) validation error (1 - accuracy); (2)
    k_max itself, as a deterministic, reproducible proxy for computational
    cost -- wall-clock time was considered but rejected: this project's
    own experiments (see BACKLOG.md) measured 2-3x run-to-run variance in
    wall time under system/OS noise unrelated to the actual algorithm,
    which would make it a noisy, non-reproducible objective.

    Both objectives are evaluated from a SINGLE FuzzyOPF fit per agent
    (memoized by exact (k_max, sigma), since opytimizer's Function calls
    each objective callable separately for the same position) -- not
    trained twice.

    Args:
        Same as genetic_search. n_agents/n_iterations behave as in a
        normal GA (NSGA-II is evolutionary): population size and number of
        generations, not agents-vs-iterations budget-split like
        run_hyperparam_search.py's single-objective heuristic.

    Returns:
        ParetoResult with every non-dominated (k_max, sigma, accuracy)
        found, sorted by k_max ascending (so points[0] is the cheapest,
        points[-1] the most accurate).
    """
    k_lo, k_hi = k_max_bounds
    cache = {} if cluster_cache is None else cluster_cache
    position_cache: dict[tuple[int, float], float] = {}
    n_calls = [0]

    def _cached_cluster_model(k_max: int) -> UnsupervisedOPF:
        if k_max not in cache:
            min_k = 1 if search_best_k else k_max
            cluster_model = UnsupervisedOPF(min_k=min_k, max_k=k_max, distance=distance)
            cluster_model.fit(X_train, Y_train)
            cache[k_max] = cluster_model
        return cache[k_max]

    def _evaluate(x: np.ndarray) -> float:
        k_max = int(np.clip(round(x[0, 0]), k_lo, k_hi))
        sigma = float(np.clip(x[1, 0], *SIGMA_BOUNDS))
        key = (k_max, round(sigma, 6))

        if key not in position_cache:
            n_calls[0] += 1
            cluster_model = _cached_cluster_model(k_max)
            model = FuzzyOPF(
                k_max=k_max, sigma=sigma, search_best_k=search_best_k,
                membership_side=membership_side, membership_kind=membership_kind, distance=distance,
            )
            model.fit(X_train, Y_train, precomputed_cluster_model=cluster_model)
            acc = opf_accuracy(Y_val, model.predict(X_val))
            position_cache[key] = acc

        return position_cache[key]

    def objective_error(x: np.ndarray) -> float:
        return 1.0 - _evaluate(x)

    def objective_cost(x: np.ndarray) -> float:
        return float(int(np.clip(round(x[0, 0]), k_lo, k_hi)))

    n_variables = 2
    lower_bound = [k_lo, SIGMA_BOUNDS[0]]
    upper_bound = [k_hi, SIGMA_BOUNDS[1]]

    with _seeded_global_rng(seed):
        space = SearchSpace(
            n_agents=n_agents, n_variables=n_variables, n_objectives=2,
            lower_bound=lower_bound, upper_bound=upper_bound,
        )
        # opytimizer 5.x's Function no longer auto-wraps a list of
        # callables for multi-objective (its `pointer` setter now
        # requires a single callable, see BACKLOG.md) -- wrap the two
        # objectives into one function that returns both values instead.
        def combined_objective(x: np.ndarray) -> list[float]:
            return [objective_error(x), objective_cost(x)]

        function = Function(combined_objective)
        optimizer = NSGA2()

        task = Opytimizer(space, optimizer, function, save_agents=False)
        history = task.start(stopping_criteria=MaxIterations(n_iterations))

    pareto_points = set()
    for agent, rank in zip(space.agents, optimizer.rank):
        if rank == 0:  # non-dominated
            k_max = int(np.clip(round(agent.position[0, 0]), k_lo, k_hi))
            sigma = float(np.clip(agent.position[1, 0], *SIGMA_BOUNDS))
            acc = 1.0 - float(agent.fit[0])
            pareto_points.add((k_max, sigma, acc))

    points = sorted(pareto_points, key=lambda p: p[0])

    return ParetoResult(points=points, n_evaluations=n_calls[0], history=history)
