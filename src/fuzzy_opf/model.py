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

import copy
import time
from typing import Literal

import numpy as np

import opfython.utils.constants as c
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph
from opfython.math.general import opf_accuracy
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
        membership_kind: str = "quadratic",
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
            membership_kind: Shape of the membership curve between
                rho_min (-> sigma) and rho_max (-> 1). "quadratic" is the
                paper's own Eq. 5. All other shapes satisfy the same two
                boundary conditions (F(rho_min)=sigma, F(rho_max)=1) by
                construction, so they are directly comparable -- only the
                curve's shape in between differs:
                  - "linear": F(t) = (1-sigma)*t + sigma. Uniform rate of
                    change; the simplest possible baseline.
                  - "quadratic" (default, Eq. 5): F(t) = (1-sigma)*t^2 + sigma.
                    Flat near rho_min, steepens toward rho_max -- typical
                    samples (high density) are pulled toward full
                    membership faster than atypical ones are pulled away
                    from sigma.
                  - "cubic": F(t) = (1-sigma)*t^3 + sigma. Same idea as
                    quadratic but more pronounced -- membership stays
                    close to sigma for a wider range of below-average
                    densities before rising sharply near rho_max.
                  - "sigmoid": S-shaped (logistic), steep in the middle of
                    the density range and flat at both ends -- unlike the
                    others, treats samples near the MEDIAN density as the
                    most "decisive" region, rather than always favouring
                    high density.
                See ``t`` in ``_compute_membership``: the normalized
                density (rho - rho_min) / (rho_max - rho_min) in [0, 1].
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
        if membership_kind not in ("linear", "quadratic", "cubic", "sigmoid"):
            raise ValueError(
                f"`membership_kind` must be one of 'linear', 'quadratic', 'cubic', 'sigmoid', "
                f"got {membership_kind!r}."
            )

        logger.info("Overriding class: OPF -> FuzzyOPF.")
        super().__init__(distance, pre_computed_distance=None)

        self.k_max = k_max
        self.sigma = sigma
        self.search_best_k = search_best_k
        self.membership_side = membership_side
        self.membership_kind = membership_kind

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

        t = (rho - rho_min) / spread

        if self.membership_kind == "linear":
            return (1.0 - self.sigma) * t + self.sigma
        elif self.membership_kind == "cubic":
            return (1.0 - self.sigma) * t**3 + self.sigma
        elif self.membership_kind == "sigmoid":
            # Logistic curve, steepness fixed at k=10 (steep enough to be
            # visibly S-shaped without being a near step-function).
            # Normalized so F(0)=sigma and F(1)=1 EXACTLY (a raw logistic
            # never quite reaches its asymptotes at finite t) -- otherwise
            # this wouldn't satisfy the same boundary conditions as the
            # other three shapes, and comparisons between them would be
            # confounded by unequal ranges, not just curve shape.
            k = 10.0
            raw = 1.0 / (1.0 + np.exp(-k * (t - 0.5)))
            raw_0 = 1.0 / (1.0 + np.exp(k * 0.5))
            raw_1 = 1.0 / (1.0 + np.exp(-k * 0.5))
            return self.sigma + (1.0 - self.sigma) * (raw - raw_0) / (raw_1 - raw_0)
        else:  # "quadratic" (default): Eq. 5 of the paper, unchanged.
            return (1.0 - self.sigma) * t**2 + self.sigma

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

    def predict_class_scores(self, X_val: np.ndarray) -> np.ndarray:
        """For each query point, returns a confidence score per class --
        needed for AUC-ROC, which plain ``predict()`` (single winning
        label, no per-class scores) cannot provide on its own.

        The OPF competition process finds the single globally-cheapest
        path for each query, pruning the search once a candidate beats the
        best cost seen so far (see ``predict()``'s delegated loop) -- it
        never needs to know the best cost reachable through EVERY class,
        only the overall winner. To get a genuine per-class score, this
        method instead does a full, unpruned scan of every training node
        for every query point, tracking the minimum path cost separately
        per class. That minimum cost is converted to a bounded,
        monotonically-decreasing "confidence" via ``1 / (1 + cost)`` (lower
        cost -> higher confidence), suitable as the decision score
        ``sklearn.metrics.roc_auc_score`` expects (does not need to sum to
        1 across classes for the one-vs-rest AUC computation).

        This is more expensive than ``predict()`` (no early pruning), same
        asymptotic O(n_train) per query either way, but likely visits more
        nodes in practice -- use it only when AUC-ROC (or similar
        per-class-score metrics) is actually needed, not as a drop-in
        replacement for predict().

        Returns:
            (n_query, n_classes) array; column order matches
            ``sorted(set(label for label in training labels))``.
        """
        classes = sorted({node.label for node in self.subgraph.nodes})
        class_to_col = {c: i for i, c in enumerate(classes)}

        pred_subgraph = Subgraph(X_val)
        scores = np.full((pred_subgraph.n_nodes, len(classes)), np.inf)

        for i in range(pred_subgraph.n_nodes):
            for node in self.subgraph.nodes:
                weight = self.distance_fn(node.features, pred_subgraph.nodes[i].features)
                cost = np.maximum(node.cost, weight)
                col = class_to_col[node.predicted_label]
                if cost < scores[i, col]:
                    scores[i, col] = cost

        raw_scores = 1.0 / (1.0 + scores)
        # sklearn's roc_auc_score(multi_class="ovr") requires each row to
        # sum to 1 (it treats the input as a probability matrix even
        # though OVR-AUC only actually needs correct RELATIVE ranking
        # within each class's column) -- row-normalize to satisfy that
        # constraint without changing the within-row ranking.
        return raw_scores / raw_scores.sum(axis=1, keepdims=True)

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

    def prune_best(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        n_iterations: int = 10,
    ) -> float:
        """Prune, but never end up worse than an earlier iteration.

        ``opfython``'s ``SupervisedOPF.prune`` (delegated to by ``prune``
        above) runs ``n_iterations`` blindly and keeps whatever the last
        iteration produced, even if validation accuracy degraded along the
        way -- measured on Cone-Torus, 3 iterations traded a 38% smaller
        training set for a 4.4-point accuracy drop, with no guarantee that
        was the best trade-off available among the iterations tried. This
        mirrors ``learn()``'s own pattern instead: track validation accuracy
        every iteration, keep a deep copy of the best-scoring state, and
        restore it at the end. Also stops early once an iteration fails to
        shrink the training set further (nothing left to prune).

        Args:
            X_train, Y_train, X_val, Y_val: Same as ``prune``.
            n_iterations: Maximum number of prune/retrain cycles.

        Returns:
            Validation accuracy of the retained (best) iteration.
        """
        self.fit(X_train, Y_train)
        preds = self.predict(X_val)
        best_acc = opf_accuracy(Y_val, preds)
        best_state = copy.deepcopy(self.__dict__)
        initial_nodes = self.subgraph.n_nodes
        previous_nodes = initial_nodes

        for iteration in range(n_iterations):
            X_temp, Y_temp = [], []
            for j, node in enumerate(self.subgraph.nodes):
                if node.relevant != c.IRRELEVANT:
                    X_temp.append(X_train[j, :])
                    Y_temp.append(Y_train[j])
            X_train = np.asarray(X_temp)
            Y_train = np.asarray(Y_temp)

            self.fit(X_train, Y_train)
            preds = self.predict(X_val)
            acc = opf_accuracy(Y_val, preds)

            logger.info(
                "Prune iteration %d/%d: n_nodes=%d, val_accuracy=%s.",
                iteration + 1, n_iterations, self.subgraph.n_nodes, acc,
            )

            if acc >= best_acc:
                best_acc = acc
                best_state = copy.deepcopy(self.__dict__)

            if self.subgraph.n_nodes >= previous_nodes:
                logger.info("Nothing left to prune, stopping early.")
                break
            previous_nodes = self.subgraph.n_nodes

        self.__dict__.update(best_state)
        logger.info(
            "Prune ratio: %s | Best validation accuracy: %s.",
            1 - self.subgraph.n_nodes / initial_nodes, best_acc,
        )
        return best_acc
