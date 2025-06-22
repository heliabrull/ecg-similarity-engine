import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.constants import (
    HEART_RATE_COL_NAME,
    RISK_TYPES,
    BEAT_TYPES,
)


class DataPreprocessor:
    """
    Preprocess ECG data for similarity search.
    """

    def __init__(self, pca_components: int = 5):
        """
        Initialize the DataPreprocessor.

        Parameters
        ----------
        n_components: int
            Number of PCA components to use.
        """
        self.pca_components = pca_components
        self.pca = PCA(n_components=self.pca_components)

        # Initialize scalers
        self.heart_scaler = StandardScaler()
        self.embedding_scaler = StandardScaler()
        self.risk_scalers = {risk: StandardScaler() for risk in RISK_TYPES}
        self.group_shapes = {}

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the DataPreprocessor to the data and transform it.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing ECG data.

        Returns
        -------
        np.ndarray
            Transformed ECG data.
        """

        self.embedding_cols = [
            col for col in df.columns if col.startswith("embedding_")
        ]
        self.prop_cols = [f"prop_{bt}" for bt in BEAT_TYPES]

        # 1. Heart rate: standardize
        heart_std = self.heart_scaler.fit_transform(df[[HEART_RATE_COL_NAME]])
        self.group_shapes[HEART_RATE_COL_NAME] = heart_std.shape[1]

        # 2. Risk scores: standardize individually
        risk_scaled = []
        for scaler, col in zip(self.risk_scalers.values(), RISK_TYPES):
            scaled_col = scaler.fit_transform(df[[col]])
            risk_scaled.append(scaled_col)
        risk_std = np.hstack(risk_scaled)
        self.group_shapes["risk_scores"] = risk_std.shape[1]

        # 3. Embeddings: standardize + PCA
        embedding_std = self.embedding_scaler.fit_transform(df[self.embedding_cols])
        embedding_pca = self.pca.fit_transform(embedding_std)
        self.group_shapes["embedding"] = embedding_pca.shape[1]

        # 4. Beat proportions: already [0, 1]
        prop_std = df[self.prop_cols]
        self.group_shapes["beat_props"] = prop_std.shape[1]

        # Combine all: stack horizontally
        return np.hstack([heart_std, risk_std, embedding_pca, prop_std])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using the fitted scalers.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing ECG data.

        Returns
        -------
        np.ndarray
            Transformed ECG data.
        """

        heart_std = self.heart_scaler.transform(df[[HEART_RATE_COL_NAME]])

        risk_scaled = []
        for col in RISK_TYPES:
            scaler = self.risk_scalers[col]
            risk_scaled.append(scaler.transform(df[[col]]))

        risk_std = np.hstack(risk_scaled)

        embedding_std = self.embedding_scaler.transform(df[self.embedding_cols])
        embedding_pca = self.pca.transform(embedding_std)

        prop_std = df[self.prop_cols]

        return np.hstack([heart_std, risk_std, embedding_pca, prop_std])
