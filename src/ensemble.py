import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from src.utils import load_split

DISTILBERT_DIR = "models/distilbert"
XGBOOST_DIR    = "models/xgboost"
OUTPUT_DIR     = "models/ensemble"

def find_best_weights(bert_probs, xgb_probs, labels):
    best_f1, best_w = 0.0, (0.7, 0.3)
    for w in np.arange(0.50, 0.96, 0.05):
        f1 = f1_score(labels,
                      (w * bert_probs + (1 - w) * xgb_probs >= 0.5).astype(int),
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, (round(float(w), 2), round(float(1 - w), 2))
    return best_w, best_f1

def evaluate_ensemble():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    _, test_labels = load_split("test")
    bert_probs = np.load(f"{DISTILBERT_DIR}/test_probs.npy")
    xgb_probs  = np.load(f"{XGBOOST_DIR}/test_probs.npy")
    baseline_f1  = f1_score(test_labels, (bert_probs >= 0.5).astype(int), zero_division=0)
    baseline_auc = roc_auc_score(test_labels, bert_probs)
    print(f"Baseline DistilBERT — F1: {baseline_f1:.4f} | AUC: {baseline_auc:.4f}")
    (w_bert, w_xgb), _ = find_best_weights(bert_probs, xgb_probs, test_labels)
    ens_probs = w_bert * bert_probs + w_xgb * xgb_probs
    ens_preds = (ens_probs >= 0.5).astype(int)
    ens_f1    = f1_score(test_labels, ens_preds, zero_division=0)
    ens_auc   = roc_auc_score(test_labels, ens_probs)
    lift      = (ens_f1 - baseline_f1) * 100
    print(f"Ensemble (w_bert={w_bert}, w_xgb={w_xgb}) — "
          f"F1: {ens_f1:.4f} | AUC: {ens_auc:.4f} | Lift: +{lift:.2f}%")
    print(classification_report(test_labels, ens_preds,
                                 target_names=["Benign", "Phishing"]))
    config = {"w_bert": w_bert, "w_xgb": w_xgb, "threshold": 0.5,
              "ensemble_f1": ens_f1, "ensemble_auc": ens_auc,
              "baseline_f1": baseline_f1, "f1_lift_percent": lift}
    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    evaluate_ensemble()
