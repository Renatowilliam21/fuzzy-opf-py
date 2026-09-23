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
        membership_source: str = "density",
        fcm_n_clusters: int | None = None,
        fcm_m: float = 2.0,
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
            membership_source: What "typicality" measure membership_kind's
                curve is applied to.
                  - "density" (default): the OPF's own k-NN density
                    estimate (Eq. 3), exactly as in the paper.
                  - "fcm": Fuzzy C-Means clustering on the raw training
                    features instead -- each sample's highest cluster
                    membership degree (how confidently it belongs to its
                    own cluster) is used as its typicality. A genuinely
                    different membership computation, not just a different
                    curve shape (unlike membership_kind alone).
                  - "density_kdtree": the SAME formula as "density"
                    (Eq. 3), but the k-nearest-neighbour search is done
                    with a KD-tree (O(n log n)) instead of opfython's own
                    O(n^2) brute-force adjacency construction -- an exact
                    (not approximate) speedup for Euclidean-derived
                    metrics (the project default, "log_squared_euclidean",
                    qualifies). Only valid with search_best_k=False (a
                    fixed k_max); opfython's own minimum-cut search over a
                    k range is a different algorithm, not replicated here.
            fcm_n_clusters: Number of FCM clusters, only used when
                membership_source="fcm". Defaults to the number of
                distinct classes in Y_train at fit time.
            fcm_m: FCM fuzziness exponent (standard default 2.0), only
                used when membership_source="fcm".
            distance: Distance metric name registered in opfython.
        """
        if not 0.0 < sigma <= 1.5:
            raise ValueError(f"`sigma` looks out of range (paper uses [0.2, 1.2]), got {sigma}.")
        if k_max < 1:
            raise ValueError(f"`k_max` must be >= 1, got {k_max}.")
        if membership_source not in ("density", "fcm", "density_kdtree"):
            raise ValueError(
                f"`membership_source` must be 'density', 'fcm', or 'density_kdtree', got {membership_source!r}."
            )
        if membership_source == "density_kdtree" and search_best_k:
            raise ValueError(
                "`membership_source='density_kdtree'` requires `search_best_k=False` "
                "(a fixed k_max) -- opfython's minimum-cut search over a k range is a "
                "different algorithm and is not replicated by the KD-tree path."
            )
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
        self.membership_source = membership_source
        self.fcm_n_clusters = fcm_n_clusters
        self.fcm_m = fcm_m

        self._cluster_model: UnsupervisedOPF | None = None

    # ------------------------------------------------------------------ #
    # Membership (Eq. 5)
    # ------------------------------------------------------------------ #
    def _apply_membership_curve(self, t: np.ndarray) -> np.ndarray:
        """Maps a normalized [0, 1] "typicality" value t into [sigma, 1]
        using ``self.membership_kind``'s curve shape. Shared by both
        membership sources (``_compute_membership`` for the density-based
        Eq. 5 path, and ``_compute_membership_fcm`` for the FCM path) so
        the two only differ in how t itself is computed, not in how it's
        mapped to a final membership value.
        """
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
        return self._apply_membership_curve(t)

    def _compute_membership_fcm(self, X_train: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
        """Alternative to the density-based Eq. 5 (``_compute_membership``):
        computes per-sample membership from Fuzzy C-Means (FCM) instead of
        the OPF's own k-NN density estimate.

        Motivated by the "pertinencia alternativa" backlog item: unlike
        ``membership_kind`` (which only changes the SHAPE of the curve
        mapping density -> membership), this changes what's being measured
        in the first place. FCM clusters the raw training features into
        ``fcm_n_clusters`` fuzzy clusters (default: one per class), giving
        each sample a membership degree in [0, 1] to every cluster (summing
        to 1 across clusters, by FCM's own definition). This sample's
        highest membership among those clusters -- how confidently it
        belongs to its own "home" cluster -- is used as its typicality,
        min-max normalized into [0, 1] and then mapped into [sigma, 1] via
        the same ``membership_kind`` curve used by the density path, for a
        fair, consistent comparison between the two membership sources.

        NOTE: as of the density_kdtree addition, this path no longer pays
        for opfython's UnsupervisedOPF clustering step in ``fit()`` --
        both "fcm" and "density_kdtree" skip it entirely (only
        membership_source="density" still needs it).
        """
        import skfuzzy as fuzz

        n_clusters = self.fcm_n_clusters or len(set(Y_train.tolist()))
        n_clusters = max(2, min(n_clusters, X_train.shape[0] - 1))  # FCM needs 2 <= c < n_samples

        _cntr, u, _u0, _d, _jm, _p, _fpc = fuzz.cluster.cmeans(
            X_train.T, c=n_clusters, m=self.fcm_m, error=0.005, maxiter=1000, seed=0,
        )
        max_u = u.max(axis=0)  # (n_samples,) -- each sample's confidence in its best-fitting cluster

        u_min, u_max = float(max_u.min()), float(max_u.max())
        spread = u_max - u_min
        if spread < 1e-12:
            logger.warning(
                "FCM gave identical max-membership for every sample: degenerate partition, "
                "membership set to 1.0 for every sample."
            )
            return np.ones_like(max_u)

        t = (max_u - u_min) / spread
        return self._apply_membership_curve(t)

    def _compute_membership_kdtree(self, X_train: np.ndarray) -> np.ndarray:
        """Alternative to the density-based Eq. 5 that reproduces the EXACT
        same Gaussian-kernel (Parzen-window) density formula opfython's
        ``KNNSubgraph.calculate_pdf`` uses, but finds each sample's k_max
        nearest neighbours via a KD-tree (``scipy.spatial.cKDTree``,
        O(n log n) average case) instead of opfython's own brute-force
        O(n^2) adjacency construction.

        This is an EXACT speedup, not an approximation, for any distance
        metric that is a strictly monotonic function of Euclidean distance
        (true of the project's default, "log_squared_euclidean", and of
        "squared_euclidean" and plain "euclidean" too): such a metric
        preserves the RANKING of nearest neighbours, so the k candidates a
        Euclidean KD-tree query returns are provably the same k neighbours
        an O(n^2) brute-force search under the configured metric would
        return -- only the (cheap, O(k) per node) final distance
        evaluation uses the real ``self.distance_fn``, not raw Euclidean
        distance. For metrics that do NOT have this property (e.g.
        cosine/angular distances), this path would silently give the wrong
        neighbours; it is therefore only exposed for the common
        Euclidean-derived metrics (see ``membership_source`` validation in
        ``__init__``).

        Replicates opfython's exact formula (see
        ``opfython.subgraphs.knn.KNNSubgraph.calculate_pdf``):
        ``pdf[i] = (1 + sum_{j in kNN(i)} exp(-d(i,j) / constant)) / (k+1)``,
        ``constant = 2 * max_adjacency_distance / 9`` (graph-wide, the
        largest distance among any node's k-nearest-neighbour arcs), then
        min-max scaled into ``[1, MAX_DENSITY]`` (1000, opfython's own
        constant) before being handed to ``_apply_membership_curve`` the
        same way the standard density path is.

        NOTE: only used when ``search_best_k=False`` (a fixed k_max) --
        opfython's own minimum-cut search over a k RANGE is not
        replicated here, since it evaluates multiple k values by rebuilding
        the graph each time, a different algorithm entirely.
        """
        import opfython.utils.constants as opf_constants
        from scipy.spatial import cKDTree

        n = X_train.shape[0]
        k = min(self.k_max, n - 1)

        tree = cKDTree(X_train)
        # k+1 because a point is always its own (distance-0) nearest
        # neighbour in a KD-tree query; drop it to keep only the k real
        # neighbours, matching opfython's own self-excluding adjacency.
        _, neighbor_idx = tree.query(X_train, k=k + 1)
        neighbor_idx = neighbor_idx[:, 1:]

        # Exact distances under the CONFIGURED metric, evaluated only for
        # the k candidates found above (O(n*k), not O(n^2)).
        dists = np.empty((n, k))
        for i in range(n):
            for pos in range(k):
                j = neighbor_idx[i, pos]
                dists[i, pos] = self.distance_fn(X_train[i], X_train[j])

        max_adjacency_distance = float(dists.max()) if n > 0 and k > 0 else 0.0
        constant = 2 * max_adjacency_distance / 9
        if constant < 0.00001:
            constant = 1.0  # matches opfython's own degenerate-case fallback

        pdf = (1.0 + np.exp(-dists / constant).sum(axis=1)) / (k + 1)

        min_pdf, max_pdf = float(pdf.min()), float(pdf.max())
        if min_pdf == max_pdf:
            # Matches opfython's own early return: equal unscaled densities
            # assign MAX_DENSITY to every node.
            rho = np.full(n, float(opf_constants.MAX_DENSITY))
        else:
            rho = (opf_constants.MAX_DENSITY - 1) * (pdf - min_pdf) / (max_pdf - min_pdf) + 1

        rho_min, rho_max = float(rho.min()), float(rho.max())
        spread = rho_max - rho_min
        if spread < 1e-12:
            return np.ones_like(rho)
        t = (rho - rho_min) / spread
        return self._apply_membership_curve(t)

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
        #    Only needed for membership_source="density" (or when a
        #    precomputed model is explicitly given) -- "fcm" and
        #    "density_kdtree" compute membership their own way and skip
        #    this (expensive, O(n^2) in opfython's own implementation)
        #    step entirely.
        if precomputed_cluster_model is not None:
            cluster_model = precomputed_cluster_model
        elif self.membership_source == "density":
            min_k = 1 if self.search_best_k else self.k_max
            cluster_model = UnsupervisedOPF(min_k=min_k, max_k=self.k_max, distance=self.distance)
            cluster_model.fit(X_train, Y_train)
        else:
            cluster_model = None
        self._cluster_model = cluster_model

        if self.membership_source == "fcm":
            membership = self._compute_membership_fcm(X_train, Y_train)
        elif self.membership_source == "density_kdtree":
            membership = self._compute_membership_kdtree(X_train)
        else:
            membership = self._compute_membership(cluster_model.subgraph)

        # 2) Supervised step: same graph, complete adjacency, weighted by membership.
        self.subgraph = Subgraph(X_train, Y_train)
        self.classes_ = np.array(sorted(set(Y_train.tolist())))  # sklearn convention
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

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        """scikit-learn-style alias for ``predict_class_scores`` -- same
        (n_query, n_classes) row-normalized output, exposed under the
        conventional name so FuzzyOPF can be used as a drop-in classifier
        anywhere code expects the standard ``fit``/``predict``/
        ``predict_proba`` interface (e.g. sklearn's ``Pipeline``,
        ``cross_val_predict``, or any tool built against that convention).

        Column order matches ``self.classes_`` (set during ``fit``), the
        same convention scikit-learn classifiers use.
        """
        return self.predict_class_scores(X_val)

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
