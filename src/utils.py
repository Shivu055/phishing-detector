import numpy as np
import polars as pl

def load_split(split, data_dir="data/processed"):
    df = pl.read_parquet(f"{data_dir}/{split}.parquet")
    return df["text"].to_list(), df["label"].to_numpy().astype(int)

def extract_header_features(texts):
    features = []
    for text in texts:
        t = text.lower()
        n = max(len(text), 1)
        features.append([
            len(text), text.count("http://"), text.count("https://"),
            text.count("@"), text.count("!"), text.count("$"), text.count("?"),
            int("click here" in t), int("verify your account" in t),
            int("urgent" in t), int("password" in t),
            int("bank" in t or "account" in t),
            int("dear customer" in t or "dear user" in t),
            int("won" in t or "winner" in t or "prize" in t),
            int("free" in t),
            text.count(" ") / n,
            sum(c.isupper() for c in text) / n,
            text.count("\n"),
        ])
    return np.array(features, dtype=np.float32)
