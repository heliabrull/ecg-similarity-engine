# ⚡ ECG Similarity Search Engine  
**Efficient and flexible retrieval of ECGs based on model-derived metrics**

## 🚀 Overview  
This repository contains a modular proof-of-concept (PoC) system for similarity search over synthetic ECG data, developed as part of an MLOps challenge. The system supports fast and configurable retrieval of ECGs using outputs from multiple simulated ML models, including risk scores, embeddings, heart rate, and beat-type proportions.

**📄 Report:** See [`report/ecg_similarity_engine_report.pdf`](report/ecg_similarity_engine_report.pdf) for detailed system design, benchmarking analysis, and future work discussion.


## 🗂 Repository structure  
```
.
├── notebooks/
│   ├── ecg_similarity_search_poc.ipynb           # Main notebook: full pipeline demo
│   └── ecg_similarity_search_benchmarking.ipynb  # Supplementary notebook: benchmarking
│
├── src/
│   ├── data_generator.py         # Synthetic ECG data generation
│   ├── data_preprocessor.py      # Group-wise preprocessing
│   ├── hybrid_indexer.py         # Modular FAISS index logic (per group)
│   ├── single_indexer.py         # Baseline single-vector index logic
│   ├── similarity_searcher.py    # Unified search interface
│   └── constants.py              # Configuration for feature types and clusters
│
├── report/
│   └── ecg_similarity_engine_report.pdf # Final system design and analysis report
│
├── requirements.txt              # Dependency list for easy setup
└── README.md                     # Project overview (this file)
```

## 🧪 How to run  

### 1. Install dependencies  
We recommend using a virtual environment:  
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Launch the main demo  
Open the full demonstration notebook `notebooks/ecg_similarity_search_poc.ipynb`. 

This notebook walks through:
- Synthetic ECG generation  
- Feature preprocessing  
- Indexing strategies (single vs. hybrid)  
- Querying by selected metric groups  
- Interpreting similarity results  


## 📁 Requirements  
See `requirements.txt`.

## 📬 Submission 
This project is submitted as part of the MLOps Engineer Challenge (Idoven, 2025).  
