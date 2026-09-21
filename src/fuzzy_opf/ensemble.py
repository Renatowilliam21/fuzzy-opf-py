# Copyright (c) 2026 Renato W. R. de Souza.
# Licensed under the Apache License, Version 2.0.

"""Ensemble of Fuzzy-OPF classifiers, combined by majority voting.

Motivated by the last item on this project's improvement list: instead of
committing to a single (k_max, sigma) found by hyperparameter search, train
several FuzzyOPF models with different combinations and let them vote. The
most natural source of diverse-but-good combinations already sitting in
this codebase is a Pareto front from ``nsga2_search`` -- every point on it
is, by definition, undominated (no other point is both cheaper AND more
accurate), so it is a principled ensemble instead of an arbitrary one.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .model import FuzzyOPF


class EnsembleFuzzyOPF:
    """Trains several FuzzyOPF models (one per given hyperparameter
    combination) and predicts by majority vote across them.

    Unlike a single FuzzyOPF, this has no single "sigma"/"k_max" -- it is
    defined entirely by its list of member configurations.
    """

    def __init__(self, configs: list[dict]) -> None:
        """
        Args:
            configs: One dict of FuzzyOPF constructor kwargs per ensemble
                member, e.g. ``[{"k_max": 20, "sigma": 0.6}, {"k_max": 8,
                "sigma": 1.1}]``. Any FuzzyOPF constructor argument may be
                set per member (including ``membership_kind``,
                ``membership_side``, ``distance``) -- members do not have
                to share hyperparameters beyond what each dict specifies;
                unset ones fall back to FuzzyOPF's own defaults.
        """
        if not configs:
            raise ValueError("`configs` must contain at least one member configuration.")

        self.configs = configs
        self.members: list[FuzzyOPF] = [FuzzyOPF(**cfg) for cfg in configs]

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "EnsembleFuzzyOPF":
        """Fits every member on the same training data."""
        for member in self.members:
            member.fit(X_train, Y_train)
        return self

    def predict(self, X_val: np.ndarray) -> list[int]:
        """Majority vote across members. Ties are broken by the first
        member in ``configs`` order (deterministic, not random) -- with an
        odd number of members and few classes, ties are rare, but this
        keeps behaviour reproducible when they do happen."""
        all_preds = [member.predict(X_val) for member in self.members]  # (n_members, n_samples)

        final = []
        for i in range(len(X_val)):
            votes = [preds[i] for preds in all_preds]
            counts = Counter(votes)
            best_count = max(counts.values())
            # Among labels with the max vote count, keep the one that was
            # cast earliest (by the first member to vote for it) -- avoids
            # Counter's arbitrary tie order depending on Python's hashing.
            winners = {label for label, c in counts.items() if c == best_count}
            final.append(next(v for v in votes if v in winners))

        return final

    @classmethod
    def from_pareto_front(cls, pareto_result, **shared_kwargs) -> "EnsembleFuzzyOPF":
        """Builds an ensemble directly from an ``nsga2_search`` result:
        one member per point on the Pareto front.

        Args:
            pareto_result: A ``ParetoResult`` (see ``fuzzy_opf.tuning.
                nsga2_search``).
            **shared_kwargs: Extra FuzzyOPF kwargs applied to every member
                (e.g. ``membership_kind="sigmoid"``, ``distance=...``) --
                only k_max/sigma come from the front itself.
        """
        configs = [
            {"k_max": k_max, "sigma": sigma, **shared_kwargs}
            for k_max, sigma, _accuracy in pareto_result.points
        ]
        return cls(configs)
