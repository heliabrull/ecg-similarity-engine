import faiss
import numpy as np
import warnings
from tqdm import tqdm

from src.constants import DEFAULT_SINGLE_INDEX_TYPE


class SingleIndexer:
    def __init__(
        self,
        dim: int,
        index_type: str = DEFAULT_SINGLE_INDEX_TYPE,
        hnsw_M: int = 32,
        hnsw_ef_search: int = 128,
    ):
        self.dim = dim
        self.index_type = index_type
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(dim)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_M)
            self.index.hnsw.efSearch = hnsw_ef_search
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def build_index(self, full_matrix: np.ndarray, batch_size: int = 100000):
        """
        Build the index.

        Parameters
        ----------
        full_matrix: np.ndarray
            The full matrix to build the index from.
        batch_size: int
            The batch size to use for building the index.
        """
        for i in tqdm(
            range(0, len(full_matrix), batch_size),
            desc=f"SingleIndex: building index with type <{self.index_type}>",
        ):
            self.index.add(full_matrix[i : i + batch_size])

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        D, I = self.index.search(query_vec.astype(np.float32), top_k)
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
