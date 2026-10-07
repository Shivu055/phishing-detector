import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from src.utils import load_split

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256
BATCH_SIZE = 32
LR         = 2e-5
MAX_EPOCHS = 10
PATIENCE   = 3
OUTPUT_DIR = "models/distilbert"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

def compute_class_weights(labels):
    counts  = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_probs, all_labels, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            total_loss += criterion(out.logits, lbls).item()
            probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            preds = out.logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), f1, auc, np.array(all_probs)

def train():
    print(f"Device: {DEVICE}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    train_texts, train_labels = load_split("train")
    val_texts,   val_labels   = load_split("val")
    test_texts,  test_labels  = load_split("test")
    tokenizer    = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(EmailDataset(train_texts, train_labels, tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(EmailDataset(val_texts,   val_labels,   tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(EmailDataset(test_texts,  test_labels,  tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model     = DistilBertForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler    = GradScaler(device=DEVICE)
    best_val_f1, patience_left, history = 0.0, PATIENCE, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE):
                out  = model(input_ids=ids, attention_mask=mask)
                loss = criterion(out.logits, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss {loss.item():.4f}")
        avg = total_loss / len(train_loader)
        vl, vf, va, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | Train: {avg:.4f} | Val F1: {vf:.4f} | Val AUC: {va:.4f}")
        history.append(dict(epoch=epoch, train_loss=avg, val_loss=vl, val_f1=vf, val_auc=va))
        if vf > best_val_f1:
            best_val_f1, patience_left = vf, PATIENCE
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ Best model saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break
    model = DistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)
    _, tf, ta, tp = evaluate(model, test_loader, criterion)
    print(f"\nTEST  F1: {tf:.4f}  AUC: {ta:.4f}")
    print(classification_report(test_labels, (tp >= 0.5).astype(int),
                                 target_names=["Benign", "Phishing"]))
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump({"test_f1": tf, "test_auc_roc": ta, "history": history}, f, indent=2)
    np.save(f"{OUTPUT_DIR}/test_probs.npy", tp)
    print(f"Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    train()
