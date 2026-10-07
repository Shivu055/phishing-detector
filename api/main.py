# makes api a package
import json, time
from contextlib import asynccontextmanager
from typing import List
import joblib, numpy as np, torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

DISTILBERT_DIR = "models/distilbert"
XGBOOST_DIR    = "models/xgboost"
ENSEMBLE_DIR   = "models/ensemble"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
_models: dict  = {}

def _extract_features(text):
    t, n = text.lower(), max(len(text), 1)
    return np.array([[
        len(text), text.count("http://"), text.count("https://"),
        text.count("@"), text.count("!"), text.count("$"), text.count("?"),
        int("click here" in t), int("verify your account" in t),
        int("urgent" in t), int("password" in t),
        int("bank" in t or "account" in t),
        int("dear customer" in t or "dear user" in t),
        int("won" in t or "winner" in t or "prize" in t),
        int("free" in t), text.count(" ") / n,
        sum(c.isupper() for c in text) / n, text.count("\n"),
    ]], dtype=np.float32)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading DistilBERT...")
    _models["tokenizer"] = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
    _models["bert"] = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_DIR).to(DEVICE)
    _models["bert"].eval()
    print("Loading XGBoost...")
    _models["xgb"] = joblib.load(f"{XGBOOST_DIR}/model.joblib")
    with open(f"{ENSEMBLE_DIR}/config.json") as f:
        _models["cfg"] = json.load(f)
    print("All models ready.")
    yield
    _models.clear()

app = FastAPI(title="Phishing Detector", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      example="Congratulations! You won a prize. Click here.")

class PredictResponse(BaseModel):
    label: str
    is_phishing: bool
    phishing_prob: float
    bert_prob: float
    xgb_prob: float
    latency_ms: float
    model_version: str = "1.0.0"

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., max_length=32)

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    total_latency_ms: float

def _predict_one(text):
    t0  = time.perf_counter()
    enc = _models["tokenizer"](text, truncation=True, padding="max_length",
                                max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = _models["bert"](**enc).logits
    bert_prob = float(torch.softmax(logits, dim=1)[0, 1].cpu())
    xgb_prob  = float(_models["xgb"].predict_proba(_extract_features(text))[0, 1])
    cfg = _models["cfg"]
    ens = cfg["w_bert"] * bert_prob + cfg["w_xgb"] * xgb_prob
    return {
        "label":         "phishing" if ens >= cfg["threshold"] else "benign",
        "is_phishing":   bool(ens >= cfg["threshold"]),
        "phishing_prob": round(ens, 4),
        "bert_prob":     round(bert_prob, 4),
        "xgb_prob":      round(xgb_prob, 4),
        "latency_ms":    round((time.perf_counter() - t0) * 1000, 2),
    }

@app.get("/")
def root(): return {"status": "ok", "version": "1.0.0"}

@app.get("/health")
def health(): return {"status": "healthy", "models_loaded": list(_models.keys())}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try: return PredictResponse(**_predict_one(req.text))
    except Exception as e: raise HTTPException(500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    try:
        results = [PredictResponse(**_predict_one(t)) for t in req.texts]
        return BatchResponse(results=results,
                             total_latency_ms=round((time.perf_counter()-t0)*1000, 2))
    except Exception as e: raise HTTPException(500, detail=str(e))
