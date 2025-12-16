from __future__ import annotations

import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def find_best_threshold(y_true, y_proba, grid=None):
    """
    Pick a probability threshold that maximizes F1 on validation data.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return best_t, best_f1


def evaluate_predictions(y_true, y_pred, model_name: str, out_dir: str = "results/metrics"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    with open(out_path / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{model_name}: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}")
    return metrics


def plot_confusion(y_true, y_pred, model_name: str, out_dir: str = "results/figures"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path / f"confusion_{model_name}.png", dpi=150)
    plt.close()
