import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay
)
from keras.models import load_model
from config import NUM_CLIENTS

# Paths to your saved models and data
model_dir = "15_30_non_models"
shard_dir = "shards"
num_clients = NUM_CLIENTS  # update if needed

# Storage for all predictions and labels
all_y_true = []
all_y_pred = []
all_y_prob = []

for cid in range(num_clients):
    model_path = os.path.join(model_dir, f"client_{cid}_model.h5")
    shard_path = os.path.join(shard_dir, f"client_{cid}.npz")

    if not os.path.exists(model_path) or not os.path.exists(shard_path):
        print(f"Skipping client {cid} (missing model or data)")
        continue

    print(f"Evaluating client {cid}...")

    model = load_model(model_path)
    data = np.load(shard_path)
    X = data["X"]
    y_true = data["y"]

    y_probs = model.predict(X)
    y_pred = np.argmax(y_probs, axis=1)
    y_prob_class1 = y_probs[:, 1]

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)
    all_y_prob.append(y_prob_class1)

# Combine all predictions across clients
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
y_prob_all = np.concatenate(all_y_prob)

# Classification metrics
report = classification_report(y_true_all, y_pred_all, output_dict=True)
precision = report['1']['precision']
recall = report['1']['recall']
f1 = report['1']['f1-score']
auc = roc_auc_score(y_true_all, y_prob_all)
tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

# Print results
print("\nCombined Metrics Across All Clients:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"FPR:       {fpr:.4f}")
print(f"FNR:       {fnr:.4f}")

# ROC Curve
fpr_curve, tpr_curve, _ = roc_curve(y_true_all, y_prob_all)
plt.figure()
plt.plot(fpr_curve, tpr_curve, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curve (All Clients)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malicious"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Combined Confusion Matrix (All Clients)")
plt.tight_layout()
plt.show()

# Class probability per data point
plt.figure()
plt.plot(y_prob_all, marker='o', linestyle='-', label="Class 1 Probability")
plt.xlabel("Data Point Index")
plt.ylabel("Predicted Probability")
plt.title("Class 1 Probability (All Clients)")
plt.grid(True)
plt.tight_layout()
plt.show()
