import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from config import NUM_CLIENTS, DATASET_PATH
OUTPUT_DIR = "shards"

def load_and_prepare_data(path):
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in all_files], ignore_index=True)

    df.drop(columns=["Flow ID", "Source IP", "Destination IP", "Timestamp"], errors="ignore", inplace=True)
    label_col = df.columns[-1]
    df[label_col] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y

def split_and_save(X, y, num_clients, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = list(zip(X, y))
    np.random.shuffle(data)
    shard_size = len(data) // num_clients

    for i in range(num_clients):
        start = i * shard_size
        end = None if i == num_clients - 1 else (i + 1) * shard_size
        shard = data[start:end]
        X_shard, y_shard = zip(*shard)
        np.savez_compressed(os.path.join(output_dir, f"client_{i}.npz"), X=np.array(X_shard), y=np.array(y_shard))

if __name__ == "__main__":
    X, y = load_and_prepare_data(DATASET_PATH)
    split_and_save(X, y, NUM_CLIENTS, OUTPUT_DIR)
    print(f"✅ Saved {NUM_CLIENTS} client shards in '{OUTPUT_DIR}'")
