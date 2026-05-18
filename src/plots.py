"""Plotting helpers — thin wrappers around matplotlib so the notebooks read clean."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def missingness_heatmap(df: pd.DataFrame, top_n: int = 100, ax=None):
    """Heatmap of the `top_n` columns with the most NaNs. Reveals row-wise patterns."""
    miss = df.isna().mean().sort_values(ascending=False).head(top_n).index
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(df[miss].isna().T.values, aspect="auto", cmap="binary", interpolation="nearest")
    ax.set_title(f"Missing values — top {top_n} sparsest sensors")
    ax.set_xlabel("Wafer index (row)")
    ax.set_ylabel("Sensor")
    return ax


def pr_curves(curves: Iterable[tuple[str, np.ndarray, np.ndarray]], ax=None):
    """`curves` is an iterable of (label, y_true, y_proba)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for label, y_true, y_proba in curves:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(recall, precision, label=label, linewidth=1.6)
    base = float(np.mean([np.mean(yt) for _, yt, _ in curves]))
    ax.axhline(base, color="grey", linestyle="--", linewidth=1, label=f"baseline = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return ax


def roc_curves(curves: Iterable[tuple[str, np.ndarray, np.ndarray]], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for label, y_true, y_proba in curves:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, label=label, linewidth=1.6)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def confusion_plot(cm: np.ndarray, class_names=("pass", "fail"), ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax
