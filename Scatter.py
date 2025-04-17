import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Config
client_id = 0
non_q_model_dir = "15_30_non_models"
q_model_dir = "15_30_with_models"
shard_dir = "shards"

#Load Data
shard_path = os.path.join(shard_dir, f"client_{client_id}.npz")
data = np.load(shard_path)
X = data["X"]
y_true = data["y"]

#Load Models
non_q_model = load_model(os.path.join(non_q_model_dir, f"client_{client_id}_model.h5"))
q_model = load_model(os.path.join(q_model_dir, f"client_{client_id}_model.h5"))

#Get Predictions
y_prob_non_q = non_q_model.predict(X)[:, 1]
y_prob_q = q_model.predict(X)[:, 1]

#Plot setup
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot non-quantized
axes[0].scatter(
    x=np.arange(len(y_prob_non_q)),
    y=y_prob_non_q,
    c=y_true,
    cmap='bwr',
    alpha=0.7,
    edgecolor='none'
)
axes[0].set_title("Without Gradient Quantization")
axes[0].set_xlabel("Data Point Index")
axes[0].set_ylabel("Class Probability")

# Plot quantized
axes[1].scatter(
    x=np.arange(len(y_prob_q)),
    y=y_prob_q,
    c=y_true,
    cmap='bwr',
    alpha=0.7,
    edgecolor='none'
)
axes[1].set_title("With Gradient Quantization")
axes[1].set_xlabel("Data Point Index")

plt.tight_layout()
plt.savefig("scatter_prediction_comparison.png", dpi=300)
plt.show()
