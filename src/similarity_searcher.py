import numpy as np
from typing import Optional
from src.single_indexer import SingleIndexer
from src.hybrid_indexer import HybridIndexer

from src.constants import (
    RISK_TYPES,
    DEFAULT_SINGLE_INDEX_TYPE,
    DEFAULT_HYBRID_INDEX_TYPES,
)


class SimilaritySearcher:
    """
    A similarity searcher that uses a single indexer for the full matrix and a hybrid indexer for the feature groups.
    """

    def __init__(
        self,
        full_matrix: np.ndarray,
        group_shapes: dict[str, int],
        group_index_types: Optional[dict[str, str]] = None,
        full_index_type: Optional[str] = None,
        full_index_batch_size: int = 100000,
        hybrid_index_batch_size: int = 100000,
    ):
        """
        Initialize the similarity searcher.

        Parameters
        ----------
        full_matrix : np.ndarray
            The full matrix to index.
        group_shapes : dict[str, int]
            The shapes of the groups.
        group_index_types : Optional[dict[str, str]]
            The types of the groups.
        full_index_type : Optional[str]
            The type of the full index.
        full_index_batch_size : int
            The batch size for the full index.
        hybrid_index_batch_size : int
            The batch size for the hybrid index.
        """
        if group_index_types is None:
            group_index_types = DEFAULT_HYBRID_INDEX_TYPES.copy()
        if full_index_type is None:
            full_index_type = DEFAULT_SINGLE_INDEX_TYPE

        self.single_indexer = SingleIndexer(
            full_matrix.shape[1], index_type=full_index_type
        )
        self.hybrid_indexer = HybridIndexer(group_shapes, group_index_types)
        self.group_shapes = group_shapes
        self.full_matrix = full_matrix
        self.full_index_batch_size = full_index_batch_size
        self.hybrid_index_batch_size = hybrid_index_batch_size
        self._build_all()

    def _build_all(self):
        """
        Build the indexes for the full matrix and for each feature group.
        """
        self.single_indexer.build_index(
            self.full_matrix, batch_size=self.full_index_batch_size
        )
        self.hybrid_indexer.build_indexes(
            self.full_matrix, batch_size=self.hybrid_index_batch_size
        )

    def _resolve_groups_and_weights(
        self, selected_groups: list[str] | None, weights: dict[str, float] | None
    ) -> tuple[list[str], dict[str, float]]:
        """
        Resolve the groups and weights.

        Parameters
        ----------
        selected_groups : list[str]
            The groups to select.
        weights : dict[str, float]
            The weights for the groups.

        Returns
        -------
        selected_groups : list[str]
            The groups to select.
        weights : dict[str, float]
            The weights for the groups.
        """
        all_groups = list(self.group_shapes.keys())
        selected_groups = selected_groups or all_groups

        unknown = set(selected_groups) - set(all_groups + RISK_TYPES)
        if unknown:
            raise ValueError(f"Unknown group(s): {unknown}")

        # Handle risk_* columns by collapsing them to "risk_scores"
        risk_columns = [g for g in selected_groups if g in RISK_TYPES]
        if risk_columns:
            selected_groups = [g for g in selected_groups if g not in RISK_TYPES]
            selected_groups.append("risk_scores")
            if weights:
                max_risk_weight = max(weights.get(r, 1.0) for r in risk_columns)
                weights["risk_scores"] = max_risk_weight

        weights = weights or {}
        final_weights = {g: weights.get(g, 1.0) for g in selected_groups}

        # Drop zero-weighted
        selected_groups = [g for g in selected_groups if final_weights[g] > 0.0]
        if not selected_groups:
            raise ValueError("No selected groups with non-zero weight.")

        self.selected_groups = selected_groups
        self.weights = final_weights

        return selected_groups, final_weights

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 100,
        selected_groups: Optional[list[str]] = None,
        weights: Optional[dict[str, float]] = None,
        hybrid_margin_factor: int = 2,
    ):
        """
        Search for the top_k most similar ECGs to the query vector.

        Parameters
        ----------
        query_vec : np.ndarray
            The query vector.
        top_k : int
            The number of neighbors to retrieve.
        selected_groups : Optional[list[str]]
            The groups to select.
        weights : Optional[dict[str, float]]
            The weights for the groups.
        hybrid_margin_factor : int
            The margin factor for the hybrid index.

        Returns
        -------
        top_indices : np.ndarray
            The indices of the top_k most similar ECGs.
        top_distances : np.ndarray
            The distances of the top_k most similar ECGs.
        """
        all_groups = list(self.group_shapes.keys())

        selected_groups, weights = self._resolve_groups_and_weights(
            selected_groups, weights
        )

        # If all groups are used, prefer single index search
        if set(selected_groups) == set(all_groups):
            return self.single_indexer.search(query_vec, top_k)

        # Otherwise use hybrid index
        return self.hybrid_indexer.search(
            query_vec, top_k, selected_groups, weights, hybrid_margin_factor
        )
