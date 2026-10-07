# Phishing & Malicious URL Detection

Ensemble of fine-tuned DistilBERT + XGBoost for phishing email detection.
F1: 0.94 | AUC-ROC: 0.97 | p99 latency: <50ms

## Results

| Model            | F1   | AUC-ROC |
|------------------|------|---------|
| DistilBERT       | 0.92 | 0.96    |
| XGBoost          | 0.88 | 0.93    |
| Ensemble (final) | 0.94 | 0.97    |

## Quickstart

pip install -r requirements.txt
python data/prepare_data.py
python -m src.train_distilbert
python -m src.train_xgboost
python -m src.ensemble
uvicorn api.main:app --host 0.0.0.0 --port 8000

## Tech Stack
Python, PyTorch, HuggingFace Transformers, XGBoost, FastAPI, Polars, Docker
