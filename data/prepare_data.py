import os
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import polars as pl

RAW_DATA_PATH = "data/raw/emails.csv"
DB_PATH       = "data/pipeline.db"

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY, run_at TEXT,
            input_hash TEXT, n_rows INTEGER,
            n_phishing INTEGER, n_benign INTEGER, output_path TEXT
        )
    """)
    conn.commit()

def hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_and_clean(path):
    df = pl.read_csv(path)
    if "Email Text" in df.columns and "Email Type" in df.columns:
        df = df.rename({"Email Text": "text", "Email Type": "label_str"})
        df = df.with_columns(
            pl.when(pl.col("label_str").str.to_lowercase().str.contains("phishing"))
            .then(pl.lit(1)).otherwise(pl.lit(0)).alias("label")
        ).drop("label_str")
    elif "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' & 'label' OR 'Email Text' & 'Email Type'")
    df = (
        df.drop_nulls(subset=["text", "label"])
        .with_columns([
            pl.col("text").str.strip_chars().alias("text"),
            pl.col("label").cast(pl.Int8),
        ])
        .filter(pl.col("text").str.len_chars() > 10)
        .unique(subset=["text"])
    )
    return df

def split_and_save(df, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    phishing = df.filter(pl.col("label") == 1).sample(fraction=1.0, shuffle=True, seed=42)
    benign   = df.filter(pl.col("label") == 0).sample(fraction=1.0, shuffle=True, seed=42)
    def sp(frame):
        n = len(frame)
        return frame[:int(n*0.8)], frame[int(n*0.8):int(n*0.9)], frame[int(n*0.9):]
    p_tr, p_v, p_te = sp(phishing)
    b_tr, b_v, b_te = sp(benign)
    train = pl.concat([p_tr, b_tr]).sample(fraction=1.0, shuffle=True, seed=42)
    val   = pl.concat([p_v,  b_v ]).sample(fraction=1.0, shuffle=True, seed=42)
    test  = pl.concat([p_te, b_te]).sample(fraction=1.0, shuffle=True, seed=42)
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame.write_parquet(f"{output_dir}/{name}.parquet")
        print(f"Saved {name}: {len(frame):,} rows")

def run_pipeline():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Put your dataset at: {RAW_DATA_PATH}\n"
            "Download: https://www.kaggle.com/datasets/subhajournal/phishingemails"
        )
    print("Loading data...")
    df = load_and_clean(RAW_DATA_PATH)
    n_p = df.filter(pl.col("label") == 1).height
    n_b = df.filter(pl.col("label") == 0).height
    print(f"Total: {df.height:,} | Phishing: {n_p:,} | Benign: {n_b:,}")
    split_and_save(df, "data/processed")
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    conn.execute("INSERT INTO pipeline_runs VALUES (?,?,?,?,?,?,?)",
        (run_id, datetime.now().isoformat(), hash_file(RAW_DATA_PATH),
         df.height, n_p, n_b, "data/processed"))
    conn.commit()
    conn.close()
    print(f"Done. Run logged: {run_id}")

if __name__ == "__main__":
    run_pipeline()
