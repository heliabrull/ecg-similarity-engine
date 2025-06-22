import faiss
import numpy as np
import os
import warnings
from tqdm import tqdm

from src.constants import DEFAULT_HYBRID_INDEX_TYPES

# Enable parallelism
faiss.omp_set_num_threads(os.cpu_count())


class HybridIndexer:
    """
    HybridIndexer is a class that builds and searches multiple FAISS indexes for different groups of data.
    It supports both flat (L2) and HNSW indexes.
    """

    def __init__(
        self,
        group_shapes: dict[str, int],
        group_index_types: dict[str, str] | None = None,
    ):
        """
        Initialize the HybridIndexer.

        Parameters
        ----------
        group_shapes: dict[str, int]
            A dictionary mapping group names to their dimensions.
        group_index_type: dict[str, str] | None
            A dictionary mapping group names to their index types.
        """
        self.group_shapes = group_shapes
        self.group_slices = {}
        self.indexes = {}
        self.index_types = group_index_types or DEFAULT_HYBRID_INDEX_TYPES.copy()
        self._init_group_slices()

    def _init_group_slices(self):
        """
        Initialize the group slices for each group.
        """
        start = 0
        for group, width in self.group_shapes.items():
            self.group_slices[group] = (start, start + width)
            start += width

    def build_indexes(
        self,
        full_matrix: np.ndarray,
        hnsw_M: int = 32,
        batch_size: int = 100000,
        hnsw_ef_search: int = 200,
    ):
        """
        Build the indexes for each group.

        Parameters
        ----------
        full_matrix: np.ndarray
            The full matrix to build the indexes from.
        batch_size: int
            The batch size to use for building the indexes.
        hnsw_M: int
            The M parameter for the HNSW index.
        hnsw_ef_search: int
            The efSearch parameter for the HNSW index.
        """
        for group, (start, end) in self.group_slices.items():
            group_data = full_matrix[:, start:end].astype(np.float32)
            dim = end - start
            index_type = self.index_types.get(group, "flat")

            if index_type == "flat":
                group_index = faiss.IndexFlatL2(dim)
            elif index_type == "hnsw":
                group_index = faiss.IndexHNSWFlat(dim, hnsw_M)
                group_index.hnsw.efSearch = hnsw_ef_search
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            for i in tqdm(
                range(0, len(group_data), batch_size),
                desc=f"HybridIndexer: building for group <{group}> with index type <{index_type}>",
            ):
                group_index.add(group_data[i : i + batch_size])

            self.indexes[group] = group_index

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        selected_groups: list[str] | None = None,
        weights: dict[str, float] | None = None,
        margin_factor: int = 2,
    ):
        """
        Search the indexes for the query vector.

        Parameters
        ----------
        query_vec: np.ndarray
            The query vector to search for.
        top_k: int
            The number of neighbors to return.
        selected_groups: list[str] | None
            The groups to search in. If None, all groups are searched.
        weights: dict[str, float] | None
            The weights for each group. If None, all groups are weighted equally.
        margin_factor: int
            The margin factor to use for the search.
        """

        normalized_dists = {}
        candidate_set = set()

        # If there is only one group selected, use that group's index directly
        if len(selected_groups) == 1:
            group = selected_groups[0]
            start, end = self.group_slices[group]
            group_index = self.indexes[group]
            query = query_vec[:, start:end].astype(np.float32)
            D, I = group_index.search(query, top_k)
            valid = I[0] != -1
            D = D[0][valid]
            I = I[0][valid]
            if len(I) < top_k:
                warnings.warn(
                    f"Only {len(I)} valid neighbors found out of requested {top_k}. "
                    "Consider increasing `efSearch` or `M` in HNSW index parameters.",
                    UserWarning,
                )
            return I, D

        # If there are multiple groups, we need to retrieve results from each group
        for group in selected_groups:
            start, end = self.group_slices[group]
            query = query_vec[:, start:end].astype(np.float32)
            group_index = self.indexes[group]
            D, I = group_index.search(query, top_k * margin_factor)

            dists = D[0]
            idxs = I[0]

            # Filter out invalid results (-1) from FAISS
            valid = idxs != -1
            dists = dists[valid]
            idxs = idxs[valid]

            if len(idxs) == 0:
                continue

            all_d = np.array(dists)
            mean = all_d.mean()
            std = all_d.std() + 1e-8

            # Directly store normalized values
            normalized_dists[group] = {
                idx: (d - mean) / std for idx, d in zip(idxs, dists)
            }
            candidate_set.update(idxs)

        # Compute candidate scores with pre-normalized distances
        scores = {}
        for idx in candidate_set:
            score = 0.0
            for group in selected_groups:
                weight = weights[group]
                norm_vals = normalized_dists.get(group, {})
                default_score = max(norm_vals.values()) + 1e-8
                score += weight * norm_vals.get(idx, default_score)
            scores[idx] = score

        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        top_indices = [i for i, _ in sorted_items[:top_k]]
        top_distances = [d for _, d in sorted_items[:top_k]]
        if len(top_indices) < top_k:
            warnings.warn(
                f"Only {len(top_indices)} valid neighbors found out of requested {top_k}. "
                "Consider increasing `efSearch` or `M` in HNSW index parameters.",
                UserWarning,
            )
        return top_indices, top_distances
