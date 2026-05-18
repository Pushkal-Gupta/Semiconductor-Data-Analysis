"""Evaluation helpers shared by the SECOM modeling notebook.

The point here is to keep the notebook focused on *modeling decisions* rather
than reimplementing the same metric block three times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    name: str
    pr_auc: float
    roc_auc: float
    f1_default: float
    f1_tuned: float
    recall_tuned: float
    threshold: float
    confusion: np.ndarray

    def as_row(self) -> Dict[str, float | str]:
        return {
            "model": self.name,
            "PR-AUC": round(self.pr_auc, 4),
            "ROC-AUC": round(self.roc_auc, 4),
            "F1 @0.5": round(self.f1_default, 4),
            "F1 (tuned)": round(self.f1_tuned, 4),
            "Recall (tuned)": round(self.recall_tuned, 4),
            "threshold": round(self.threshold, 4),
        }


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = 0.6) -> float:
    """Pick the highest-precision threshold whose recall is still >= min_recall.

    Motivation: in a fab, recall (catching defects) matters more than precision
    up to a point — but we still want to maximize precision at that recall
    floor so the engineering team isn't drowning in false alarms.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one extra point at threshold=infinity
    precision = precision[:-1]
    recall = recall[:-1]
    eligible = recall >= min_recall
    if not eligible.any():
        # No threshold meets the recall floor — fall back to the highest-recall one
        return float(thresholds[np.argmax(recall)])
    best_idx = np.argmax(np.where(eligible, precision, -np.inf))
    return float(thresholds[best_idx])


def evaluate(name: str, y_true: np.ndarray, y_proba: np.ndarray,
             min_recall: float = 0.6) -> EvalResult:
    """Compute every metric we report in the modeling notebook."""
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default, zero_division=0)

    threshold = tune_threshold(y_true, y_proba, min_recall=min_recall)
    y_pred_tuned = (y_proba >= threshold).astype(int)
    f1_tuned = f1_score(y_true, y_pred_tuned, zero_division=0)
    recall_tuned = recall_score(y_true, y_pred_tuned, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_tuned)

    return EvalResult(
        name=name,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        f1_default=f1_default,
        f1_tuned=f1_tuned,
        recall_tuned=recall_tuned,
        threshold=threshold,
        confusion=cm,
    )
