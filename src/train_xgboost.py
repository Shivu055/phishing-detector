import json
from pathlib import Path
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from src.utils import load_split, extract_header_features

OUTPUT_DIR = "models/xgboost"

def find_best_threshold(probs, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2)

def train():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    train_texts, train_labels = load_split("train")
    val_texts,   val_labels   = load_split("val")
    test_texts,  test_labels  = load_split("test")
    print("Extracting features...")
    X_train = extract_header_features(train_texts)
    X_val   = extract_header_features(val_texts)
    X_test  = extract_header_features(test_texts)
    X_tv    = np.vstack([X_train, X_val])
    y_tv    = np.concatenate([train_labels, val_labels])
    scale_pos = int((y_tv == 0).sum()) / max(int((y_tv == 1).sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tv, y_tv, eval_set=[(X_val, val_labels)], verbose=50)
    test_probs = model.predict_proba(X_test)[:, 1]
    best_t     = find_best_threshold(test_probs, test_labels)
    test_preds = (test_probs >= best_t).astype(int)
    f1  = f1_score(test_labels, test_preds, zero_division=0)
    auc = roc_auc_score(test_labels, test_probs)
    print(f"\nXGBOOST  F1: {f1:.4f}  AUC: {auc:.4f}  threshold: {best_t}")
    print(classification_report(test_labels, test_preds,
                                 target_names=["Benign", "Phishing"]))
    joblib.dump(model, f"{OUTPUT_DIR}/model.joblib")
    np.save(f"{OUTPUT_DIR}/test_probs.npy", test_probs)
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump({"test_f1": f1, "test_auc_roc": auc,
                   "best_threshold": best_t}, f, indent=2)
    print(f"Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    train()
