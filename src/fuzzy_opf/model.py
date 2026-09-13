# Copyright (c) 2026 Renato W. R. de Souza.
# Licensed under the Apache License, Version 2.0.

"""Fuzzy Optimum-Path Forest (Fuzzy-OPF).

Port of the original C implementation (LibOPF_fuzzy) to Python, built on top
of ``opfython`` (SupervisedOPF / UnsupervisedOPF), following:

    R. W. R. de Souza, J. V. C. de Oliveira, L. A. Passos, W. Ding,
    J. P. Papa and V. H. C. de Albuquerque. "A Novel Approach for
    Optimum-Path Forest Classification Using Fuzzy Logic."
    IEEE Transactions on Fuzzy Systems (2019). doi:10.1109/TFUZZ.2019.2949014

Relative to the original C code, this port:
  * reuses ``opfython``'s density/clustering routines instead of
    re-implementing and re-reading the dataset twice (see the original
    ``fuzzy.c``, which calls ``ReadSubgraph`` on the same file twice);
  * applies the membership F_Theta(u) of the *candidate* node u in the
    cost update, matching Eq. (6) / Algorithm 3 of the paper exactly.
    The original C code (``OPF.c::opf_OPF_Fuzzy_Training``) multiplies by
    the membership of the *source* node p instead; both behaviours are kept
    here (``membership_side="target"`` reproduces the paper,
    ``membership_side="source"`` reproduces the legacy C behaviour) so
    results can be compared/reproduced;
  * validates sigma against the range used in the paper ([0.2, 1.2]) and
    guards against rho_max == rho_min (constant density -> would divide by
    zero in the original C code);
  * exposes k_max/sigma as normal constructor parameters so they can be
    driven by an external hyperparameter search (grid, random or
    meta-heuristic, see ``fuzzy_opf.tuning``) instead of a hardcoded
    brute-force loop like ``fuzzy_validation_opf.c``.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

import opfython.utils.constants as c
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph
from opfython.models.supervised import SupervisedOPF
from opfython.models.unsupervised import UnsupervisedOPF
from opfython.utils.logging import get_logger

logger = get_logger(__name__)

MembershipSide = Literal["target", "source"]


class FuzzyOPF(OPF):
    """Supervised OPF classifier weighted by an unsupervised membership degree."""

    def __init__(
        self,
        k_max: int = 10,
        sigma: float = 0.6,
        search_best_k: bool = True,
        membership_side: MembershipSide = "target",
        distance: str = "log_squared_euclidean",
    ) -> None:
        """
        Args:
            k_max: Upper bound for the k-nearest-neighbour graph used to
                estimate the unsupervised density (Eq. 3 in the paper).
            sigma: Lower bound of the membership function (Eq. 5). The
                paper restricts it to [0.2, 1.2]; sigma == 1 makes
                Fuzzy-OPF degenerate to standard OPF.
            search_best_k: If True, runs the minimum-cut search
                (equivalent to ``opf_BestkMinCut``) over k in [1, k_max]
                to pick k*. If False, uses k_max directly as the fixed
                neighbourhood size (equivalent to the plain ``fuzzy.c``
                driver, which skips the min-cut search).
            membership_side: See module docstring.
            distance: Distance metric name registered in opfython.
        """
        if not 0.0 < sigma <= 1.5:
            raise ValueError(f"`sigma` looks out of range (paper uses [0.2, 1.2]), got {sigma}.")
        if k_max < 1:
            raise ValueError(f"`k_max` must be >= 1, got {k_max}.")
        if membership_side not in ("target", "source"):
            raise ValueError(
                f"`membership_side` must be 'target' or 'source', got {membership_side!r}."
            )

        logger.info("Overriding class: OPF -> FuzzyOPF.")
        super().__init__(distance, pre_computed_distance=None)

        self.k_max = k_max
        self.sigma = sigma
        self.search_best_k = search_best_k
        self.membership_side = membership_side

        self._cluster_model: UnsupervisedOPF | None = None

    # ------------------------------------------------------------------ #
    # Membership (Eq. 5)
    # ------------------------------------------------------------------ #
    def _compute_membership(self, subgraph: Subgraph) -> np.ndarray:
        rho = np.asarray([node.density for node in subgraph.nodes], dtype=float)

        # NOTE: subgraph.min_density/max_density (like sg->mindens/maxdens in
        # the original C opf_PDF) hold the *pre-normalization* PDF stats, not
        # the range of the actual (normalized) node.density values used by
        # Eq. 5. We must recompute rho_min/rho_max over `rho` itself here,
        # exactly like the original C authors did in fuzzy.c/fuzzy_validation_opf.c
        # (fuzzy_mindens/fuzzy_maxdens, looped explicitly over node[i].dens).
        rho_min, rho_max = float(rho.min()), float(rho.max())

        spread = rho_max - rho_min
        if spread < 1e-12:
            # All samples share the same density (degenerate/constant PDF):
            # every sample is equally "typical", so membership collapses to 1.
            logger.warning(
                "rho_max == rho_min: degenerate density, membership set to 1.0 for every sample."
            )
            return np.ones_like(rho)

        return (1.0 - self.sigma) * ((rho - rho_min) / spread) ** 2 + self.sigma

    # ------------------------------------------------------------------ #
    # Training (Algorithm 3)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        precomputed_cluster_model: UnsupervisedOPF | None = None,
    ) -> "FuzzyOPF":
        """Fit the classifier.

        Args:
            X_train, Y_train: Training data.
            precomputed_cluster_model: Optional, already-fitted
                ``UnsupervisedOPF`` over the same ``X_train``/``Y_train``.
                When given, the (expensive) clustering/density step is
                skipped and this model's densities are reused directly.
                Used by ``fuzzy_opf.tuning`` to cache clustering per
                ``k_max`` across sigma values without duplicating the
                training logic here.
        """
        logger.info("Fitting Fuzzy-OPF classifier ...")
        start = time.time()

        # 1) Unsupervised step: densities via OPF clustering (Eq. 3).
        #    This replaces opf_CreateArcs + opf_PDF (+ opf_BestkMinCut).
        if precomputed_cluster_model is not None:
            cluster_model = precomputed_cluster_model
        else:
            min_k = 1 if self.search_best_k else self.k_max
            cluster_model = UnsupervisedOPF(min_k=min_k, max_k=self.k_max, distance=self.distance)
            cluster_model.fit(X_train, Y_train)
        self._cluster_model = cluster_model

        membership = self._compute_membership(cluster_model.subgraph)

        # 2) Supervised step: same graph, complete adjacency, weighted by membership.
        self.subgraph = Subgraph(X_train, Y_train)
        for node, m in zip(self.subgraph.nodes, membership):
            node.membership = float(m)

        self._find_prototypes()
        self._grow_fuzzy_minimax_forest()
        self.subgraph.trained = True

        logger.info("Classifier has been fitted.")
        logger.info("Training time: %s seconds.", time.time() - start)
        return self

    def _find_prototypes(self) -> None:
        """MST-based prototype search (Sec. II-A of the paper).

        Delegated directly to ``SupervisedOPF._find_prototypes`` instead of
        keeping a hand-copied duplicate here: it only touches attributes
        FuzzyOPF also has via the shared ``OPF`` base class (``subgraph``,
        ``distance_fn``, ``pre_computed_distance``, ``pre_distances``), so
        calling the real, tested implementation keeps this in sync with
        opfython automatically instead of risking silent drift.
        """
        SupervisedOPF._find_prototypes(self)

    def _grow_fuzzy_minimax_forest(self) -> None:
        """Competition process weighted by membership (Eq. 6 / Algorithm 3)."""
        subgraph = self.subgraph
        heap = Heap(size=subgraph.n_nodes)

        for i, node in enumerate(subgraph.nodes):
            if node.status == c.PROTOTYPE:
                node.pred = c.NIL
                node.predicted_label = node.label
                heap.cost[i] = 0
                heap.insert(i)
            else:
                heap.cost[i] = c.FLOAT_MAX

        while not heap.is_empty():
            p = heap.remove()
            node = subgraph.nodes[p]
            subgraph.idx_nodes.append(p)
            node.cost = heap.cost[p]

            # NOTE: the `heap.color[q] != c.BLACK` check is NOT optional here,
            # unlike in opfython's own (unweighted) `_grow_minimax_forest`.
            # There, `current_cost = max(cost[p], weight) >= cost[p]` always,
            # so a heap.cost[p] < heap.cost[q] with q already black (i.e.
            # heap.cost[q] already <= any not-yet-removed cost, by the
            # min-heap invariant) can never happen -- the check is provably
            # redundant. Multiplying by `membership` (<= 1) breaks that
            # guarantee: current_cost can fall below heap.cost[p] itself,
            # so an already-finalized (black) node could otherwise get its
            # `pred` rewritten, corrupting the forest into having a cycle
            # (mark_nodes/prune then loops forever walking `pred`). Matches
            # the original C's `Q->color[q] != BLACK` guard in fuzzy.c.
            for q, neighbour in enumerate(subgraph.nodes):
                if p != q and heap.color[q] != c.BLACK and heap.cost[p] < heap.cost[q]:
                    weight = self.distance_fn(node.features, neighbour.features)
                    base_cost = np.maximum(heap.cost[p], weight)

                    # Eq. 6: cst <- F_Theta(u) * max{C(q), d(q,u)}.
                    # "target" = candidate/destination node (paper); "source" =
                    # node being expanded (legacy LibOPF_fuzzy C behaviour).
                    membership = neighbour.membership if self.membership_side == "target" else node.membership
                    current_cost = membership * base_cost

                    if current_cost < heap.cost[q]:
                        neighbour.pred = p
                        neighbour.predicted_label = node.predicted_label
                        heap.update(q, current_cost)

    # ------------------------------------------------------------------ #
    # Classification (unchanged w.r.t. standard OPF: membership only acts
    # during training, exactly as stated in the paper)
    # ------------------------------------------------------------------ #
    def predict(self, X_val: np.ndarray) -> list[int]:
        """Classify samples with the pre-trained classifier.

        Delegated to ``SupervisedOPF.predict`` for the same reason as
        ``_find_prototypes``: membership (Eq. 6) only affects training, not
        classification, so plain OPF's implementation applies unchanged.
        Reusing it also gives FuzzyOPF the node-relevance marking
        (``subgraph.mark_nodes``) needed by ``prune()`` below, which our
        previous hand-copied version silently lacked.
        """
        return SupervisedOPF.predict(self, X_val)

    def prune(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        n_iterations: int = 10,
    ) -> None:
        """Iteratively remove training samples irrelevant to validation predictions.

        Port of ``opf_pruning.c`` from the original LibOPF, via delegation to
        ``SupervisedOPF.prune``: it only calls ``self.fit``/``self.predict``,
        which Python resolves to FuzzyOPF's own (membership-aware) versions
        since ``self`` is a FuzzyOPF instance -- so every pruning iteration
        retrains a real Fuzzy-OPF, not a plain OPF. Mutates ``self`` in place
        (matching the original's behaviour); re-fit from scratch if you need
        the unpruned classifier again.

        Args:
            X_train, Y_train: Training split (unchanged; pruning works on a
                copy internally).
            X_val, Y_val: Validation split used to decide which samples are
                relevant.
            n_iterations: Maximum number of prune/retrain cycles.
        """
        SupervisedOPF.prune(self, X_train, Y_train, X_val, Y_val, n_iterations)
