import flwr as fl
import numpy as np
import os
import keras
import csv
from config import USE_QUANTIZATION
import time
# Simple CNN model for classification
def build_model(input_shape, num_classes=2):
    model = keras.models.Sequential([
        keras.layers.Conv1D(32, 3, activation="relu", input_shape=input_shape),
        keras.layers.MaxPooling1D(),
        keras.layers.Conv1D(64, 3, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Load client-specific data shard
def load_data(client_id):
    shard_file = os.path.join("shards", f"client_{client_id}.npz")
    if not os.path.exists(shard_file):
        raise FileNotFoundError(f"Shard not found: {shard_file}")
    with np.load(shard_file) as data:
        return data["X"], data["y"]

#Quantization for communication efficiency
def stochastic_quantize(weights, num_levels=1024):
    quantized = []
    for w in weights:
        if w is None:
            quantized.append(None)
            continue

        w = w.astype(np.float32)
        max_val = np.max(np.abs(w))
        if max_val == 0:
            quantized.append(np.zeros_like(w))
            continue

        scale = max_val / (num_levels // 2 - 1)
        normalized = w / scale

        # Apply stochastic rounding
        lower = np.floor(normalized)
        prob = normalized - lower
        rnd = np.random.rand(*normalized.shape)
        rounded = np.where(rnd < prob, lower + 1, lower)

        q = np.clip(rounded, -num_levels // 2, num_levels // 2 - 1).astype(np.int8)
        quantized.append(q)
    return quantized

class Client(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        self.X_train, self.y_train = load_data(self.cid)
        self.model = build_model(input_shape=self.X_train.shape[1:])
        print(f"[Client {self.cid}] Initialized with data shape: {self.X_train.shape}")

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):

        self.model.set_weights(parameters)

        # Start timing
        start = time.time()

        # Train model
        history = self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)

        # Stop timing
        duration = time.time() - start

        # Get training metrics
        train_loss = history.history["loss"][0]
        train_accuracy = history.history["accuracy"][0]

        # Original weights
        original_weights = self.model.get_weights()
        original_size = sum(w.nbytes for w in original_weights if w is not None) / 1e6

        # Quantize
        if USE_QUANTIZATION:
            # Apply quantization
            quantized_weights = stochastic_quantize(original_weights)
            quantized_size = sum(w.size for w in quantized_weights if w is not None) / 1e6  # MB
            compression_ratio = quantized_size / original_size if original_size != 0 else 0
        else:
            # No quantization
            quantized_weights = original_weights
            quantized_size = original_size
            compression_ratio = 1.0

        # Round tracking
        round_num = config.get("round")

        # Save to CSV
        csv_path = "all_client_metrics.csv"
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "client_id", "round", "train_time_sec", "original_size_MB",
                    "quantized_size_MB", "train_accuracy", "train_loss", "compression_ratio"
                ])
            writer.writerow([
                self.cid, round_num, duration, original_size,
                quantized_size, train_accuracy, train_loss, compression_ratio
            ])

        self.model.save(f"client_{self.cid}_model.h5")

        return quantized_weights, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        print(f"[Client {self.cid}] Evaluation | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

        # Save evaluation results to CSV
        with open(f"client_{self.cid}_training_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["eval", self.cid, loss, accuracy, ""])

        return loss, len(self.X_train), {"accuracy": accuracy}

# Run client
if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client(client_id))
