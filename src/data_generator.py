import numpy as np
import pandas as pd

from src.constants import (
    CLUSTERS,
    RISK_TYPES,
    RISK_VARIANCES,
    BEAT_TYPES,
    MAX_HEART_RATE,
    MIN_HEART_RATE,
    HEART_RATE_COL_NAME,
)


class DataGenerator:
    """
    DataGenerator class for generating synthetic ECG data.
    """

    def __init__(
        self,
        num_ecgs: int = 1000,
        embedding_dim: int = 64,
        random_state: int = 42,
    ):
        """
        Initialize the DataGenerator.

        Parameters
        ----------
        num_ecgs : int
            Number of ECGs to generate.
        embedding_dim : int
            Dimension of the embedding space.
        random_state : int
            Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)
        self.num_ecgs = num_ecgs
        self.embedding_dim = embedding_dim
        self.cluster_list = list(CLUSTERS.keys())
        self.risk_types = RISK_TYPES
        self.risk_means_matrix = np.array(
            [
                [CLUSTERS[cluster]["risk_means"][risk] for risk in self.risk_types]
                for cluster in self.cluster_list
            ]
        )
        self.risk_variances = RISK_VARIANCES
        self.ecg_ids = None
        self.generated_clusters = None

    def generate_ids(self) -> None:
        """Generate a list of ECG IDs."""
        self.ecg_ids = [f"ecg_{i}" for i in range(self.num_ecgs)]

    def generate_cluster_assignments(self) -> None:
        """Generate cluster assignments based on the specified distribution."""
        self.generated_clusters = self.rng.choice(
            list(CLUSTERS.keys()),
            size=self.num_ecgs,
            p=[v["distribution"] for v in CLUSTERS.values()],
        )

    @staticmethod
    def _compute_beta_params(
        mu: np.ndarray, var: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Beta distribution parameters a and b for a given mean and variance.

        Parameters
        ----------
        mu : np.ndarray
            Array of means per sample (shape: [n_samples]).
        var : float
            Scalar variance value to use across the array.

        Returns
        -------
        (a, b) : tuple of np.ndarray
            Arrays of Beta distribution parameters a and b (same shape as mu).
        """
        if not np.all((0 < mu) & (mu < 1)):
            raise ValueError("All mean values must be in (0, 1) range.")

        scale = (mu * (1 - mu)) / var - 1
        a = mu * scale
        b = (1 - mu) * scale
        return a, b

    def generate_risk_scores(self) -> np.ndarray:
        """
        Generate risk scores for all ECGs using Beta distributions,
        based on cluster-specific means and risk-specific variances.

        Returns
        -------
        np.ndarray
            Matrix of shape (num_ecgs, num_risk_types) with sampled probabilities.
        """
        if self.generated_clusters is None:
            raise ValueError("Cluster assignments must be generated first.")

        n = self.num_ecgs
        probs = np.zeros((n, len(self.risk_types)))

        for i, risk in enumerate(self.risk_types):
            variance = self.risk_variances[risk]
            mu_array = np.zeros(n)

            for cluster in self.cluster_list:
                mask = self.generated_clusters == cluster
                mu_val = self.risk_means_matrix[self.cluster_list.index(cluster), i]
                mu_array[mask] = mu_val

            a, b = self._compute_beta_params(mu_array, variance)
            probs[:, i] = self.rng.beta(a, b)

        return probs

    def generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for each ECG based on cluster assignments.

        Returns
        -------
        np.ndarray
            Embeddings of shape (num_ecgs, embedding_dim).
        """
        if self.generated_clusters is None:
            raise ValueError("Cluster assignments must be generated first.")

        D = self.embedding_dim
        embeddings = np.zeros((self.num_ecgs, D))

        # Set random means per cluster
        cluster_names = self.cluster_list
        means = {
            cluster: self.rng.normal(0, 1, self.embedding_dim)
            for cluster in self.cluster_list
        }
        scales = {
            cluster: CLUSTERS[cluster]["embedding_scale"]
            for cluster in self.cluster_list
        }

        for cluster in cluster_names:
            mask = self.generated_clusters == cluster
            num = np.sum(mask)
            if num == 0:
                continue
            embeddings[mask] = (
                means[cluster] + self.rng.normal(0, 1, (num, D)) * scales[cluster]
            )

        return embeddings

    def generate_beat_proportions(self) -> np.ndarray:
        """Generate beat proportions for each ECG based on cluster assignments.

        Returns
        -------
        np.ndarray
            Beat proportions of shape (num_ecgs, 5).
        """
        if self.generated_clusters is None:
            raise ValueError("Cluster assignments must be generated first.")

        n = self.num_ecgs

        cluster_beats = {
            cluster: CLUSTERS[cluster]["beat_props"] for cluster in self.cluster_list
        }
        beat_matrix = np.zeros((n, len(BEAT_TYPES)))

        for cluster, target in cluster_beats.items():
            mask = self.generated_clusters == cluster
            num = np.sum(mask)
            if num == 0:
                continue
            alpha = np.array(target) * 100  # Scale for Dirichlet distribution
            beat_matrix[mask] = self.rng.dirichlet(alpha, size=num)

        return beat_matrix

    def generate_heart_rates(self) -> np.ndarray:
        """Generate heart rates for each ECG based on cluster assignments.

        Returns
        -------
        np.ndarray
            Heart rates of shape (num_ecgs,).
        """
        if self.generated_clusters is None:
            raise ValueError("Cluster assignments must be generated first.")

        hr = np.zeros(self.num_ecgs)
        for cluster in self.cluster_list:
            mask = self.generated_clusters == cluster
            mu = CLUSTERS[cluster]["heart_rate_mu"]
            sigma = CLUSTERS[cluster]["heart_rate_sigma"]
            hr[mask] = np.clip(
                self.rng.normal(mu, sigma, size=np.sum(mask)),
                MIN_HEART_RATE,
                MAX_HEART_RATE,
            )

        return hr

    def generate_data(self) -> pd.DataFrame:
        """Generate the complete dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with ECG data.
            Columns:
            - ecg_id: ECG ID
            - cluster: Cluster assignment
            - heart_rate: Heart rate
            - risk_*: Risk score for each risk type
            - embedding_*: Embedding for each dimension
            - beat_proportions: Beat proportions for each beat type
        """
        self.generate_ids()
        self.generate_cluster_assignments()
        risk_scores = self.generate_risk_scores()
        embeddings = self.generate_embeddings()
        beat_proportions = self.generate_beat_proportions()
        heart_rates = self.generate_heart_rates()

        data = pd.DataFrame(
            {
                "ecg_id": self.ecg_ids,
                "cluster": self.generated_clusters,
                HEART_RATE_COL_NAME: heart_rates,
            }
        )

        risk_scores_df = pd.DataFrame(risk_scores, columns=self.risk_types)
        embeddings_df = pd.DataFrame(
            embeddings, columns=[f"embedding_{i}" for i in range(self.embedding_dim)]
        )
        beat_proportions_df = pd.DataFrame(
            beat_proportions, columns=[f"prop_{bt}" for bt in BEAT_TYPES]
        )
        data = pd.concat(
            [data, risk_scores_df, embeddings_df, beat_proportions_df], axis=1
        )
        data.set_index("ecg_id", inplace=True)
        return data
