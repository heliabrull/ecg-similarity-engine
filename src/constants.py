# Cluster-specific constants
CLUSTERS = {
    "normal": {
        "distribution": 0.4,
        "risk_means": {
            "risk_afib": 0.05,
            "risk_bradycardia": 0.05,
            "risk_tachycardia": 0.05,
            "risk_pvc": 0.05,
            "risk_ischemia": 0.05,
        },
        "beat_props": [0.95, 0.0125, 0.0125, 0.0125, 0.0125],
        "embedding_scale": 0.5,
        "heart_rate_mu": 75,
        "heart_rate_sigma": 5,
    },
    "afib_prone": {
        "distribution": 0.2,
        "risk_means": {
            "risk_afib": 0.85,
            "risk_bradycardia": 0.05,
            "risk_tachycardia": 0.05,
            "risk_pvc": 0.05,
            "risk_ischemia": 0.05,
        },
        "beat_props": [0.65, 0.15, 0.025, 0.025, 0.15],
        "embedding_scale": 0.8,
        "heart_rate_mu": 110,
        "heart_rate_sigma": 15,
    },
    "bradycardia": {
        "distribution": 0.1,
        "risk_means": {
            "risk_afib": 0.05,
            "risk_bradycardia": 0.85,
            "risk_tachycardia": 0.05,
            "risk_pvc": 0.05,
            "risk_ischemia": 0.05,
        },
        "beat_props": [0.95, 0.0125, 0.0125, 0.0125, 0.0125],
        "embedding_scale": 0.8,
        "heart_rate_mu": 55,
        "heart_rate_sigma": 5,
    },
    "tachycardia": {
        "distribution": 0.1,
        "risk_means": {
            "risk_afib": 0.05,
            "risk_bradycardia": 0.05,
            "risk_tachycardia": 0.85,
            "risk_pvc": 0.05,
            "risk_ischemia": 0.05,
        },
        "beat_props": [0.95, 0.0125, 0.0125, 0.0125, 0.0125],
        "embedding_scale": 0.8,
        "heart_rate_mu": 110,
        "heart_rate_sigma": 10,
    },
    "ischemia": {
        "distribution": 0.1,
        "risk_means": {
            "risk_afib": 0.05,
            "risk_bradycardia": 0.05,
            "risk_tachycardia": 0.05,
            "risk_pvc": 0.30,
            "risk_ischemia": 0.85,
        },
        "beat_props": [0.86, 0.025, 0.065, 0.025, 0.025],
        "embedding_scale": 0.8,
        "heart_rate_mu": 80,
        "heart_rate_sigma": 10,
    },
    "pvc_heavy": {
        "distribution": 0.1,
        "risk_means": {
            "risk_afib": 0.05,
            "risk_bradycardia": 0.05,
            "risk_tachycardia": 0.05,
            "risk_pvc": 0.85,
            "risk_ischemia": 0.30,
        },
        "beat_props": [0.5875, 0.025, 0.35, 0.025, 0.0125],
        "embedding_scale": 0.8,
        "heart_rate_mu": 80,
        "heart_rate_sigma": 10,
    },
}

# List of risk types
RISK_TYPES = [
    "risk_afib",
    "risk_bradycardia",
    "risk_tachycardia",
    "risk_pvc",
    "risk_ischemia",
]

# Variance of the model output for each risk type
RISK_VARIANCES = {
    "risk_afib": 0.003,
    "risk_bradycardia": 0.005,
    "risk_tachycardia": 0.007,
    "risk_pvc": 0.0085,
    "risk_ischemia": 0.010,
}

# List of beat types
BEAT_TYPES = ["N", "S", "V", "F", "Q"]

# Heart rate constants
HEART_RATE_COL_NAME = "heart_rate"
MAX_HEART_RATE = 200
MIN_HEART_RATE = 30

# Default index types
DEFAULT_HYBRID_INDEX_TYPES = {
    "heart_rate": "flat",
    "risk_scores": "hnsw",
    "embedding": "hnsw",
    "beat_props": "hnsw",
}

DEFAULT_SINGLE_INDEX_TYPE = "hnsw"
