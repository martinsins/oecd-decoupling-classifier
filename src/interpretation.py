from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_tree_importance(
    tree_model,
    feature_names: Sequence[str],
    out_dir: str = "results/figures",
    filename: str = "tree_importances.png",
) -> Path:
    """
    Save a simple bar plot of DecisionTreeClassifier.feature_importances_.

    Parameters
    ----------
    tree_model : fitted sklearn.tree.DecisionTreeClassifier
    feature_names : list of feature names in the same order as training columns
    out_dir : output directory
    filename : output image name

    Returns
    -------
    Path to the saved figure.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    importances = np.asarray(tree_model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1]

    labels = [feature_names[i] for i in order]
    values = importances[order]

    plt.figure(figsize=(6, 3))
    plt.bar(labels, values)
    plt.ylabel("Importance")
    plt.title("Decision tree feature importance")
    plt.tight_layout()
    fig_path = out_path / filename
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return fig_path
