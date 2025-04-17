import os
import time
import numpy as np
from keras.models import load_model
from config import NUM_CLIENTS
# Setup
model_dir = "15_30_with_models"  #
shard_dir = "shards"
num_clients = NUM_CLIENTS

# Output file
output_path = "testing_times.csv"
with open(output_path, "w") as f:
    f.write("client_id,loss,accuracy,eval_time_sec\n")

# Loop through clients
for cid in range(num_clients):
    model_path = os.path.join(model_dir, f"client_{cid}_model.h5")
    data_path = os.path.join(shard_dir, f"client_{cid}.npz")

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print(f"Skipping client {cid}")
        continue

    model = load_model(model_path)
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    # Measure evaluation time
    start = time.time()
    loss, accuracy = model.evaluate(X, y, verbose=0)
    eval_time = time.time() - start

    # Log result
    with open(output_path, "a") as f:
        f.write(f"{cid},{loss:.4f},{accuracy:.4f},{eval_time:.4f}\n")

    print(f"Client {cid} | Accuracy: {accuracy:.4f} | Eval Time: {eval_time:.2f}s")
