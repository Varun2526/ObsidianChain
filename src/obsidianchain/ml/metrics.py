"""Phase 6 metrics. SPEC 6.2.

Accuracy is never reported: at 5.4% prevalence it is not informative.
PR-AUC is always accompanied by its no-skill baseline, which is the positive
prevalence of whatever set it was computed on - quoting a PR-AUC without it
lets a prevalence shift read as a performance change, which is exactly the
confound SPEC 6.5 P3 exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)

K_VALUES = (10, 50, 100)


def normalised_average_precision(y_true, scores) -> tuple[float, float]:
    """``(nAP, prevalence)``. SPEC 6.5 P3.

    ``nAP = (AP - prevalence) / (1 - prevalence)`` puts no-skill at 0 and
    perfect at 1 in EVERY period, which raw AP does not: AP's floor is the
    prevalence itself, so a prevalence drop lowers AP with no change in the
    model. Lift (``AP / prevalence``) fixes the floor but not the ceiling,
    which is ``1 / prevalence`` and differs 4.4x between the two test
    sub-periods - so lift was rejected in favour of this.
    """
    y_true = np.asarray(y_true)
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    if not np.isfinite(prevalence) or prevalence in (0.0, 1.0):
        return float("nan"), prevalence
    ap = float(average_precision_score(y_true, scores))
    return (ap - prevalence) / (1.0 - prevalence), prevalence


def precision_at_k(y_true, scores, k: int) -> float:
    """Precision among the k highest-scored items."""
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y_true[top].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    """Share of ALL positives captured in the k highest-scored items.

    Reported beside precision@k at cluster level so an aggregation cannot
    look good merely by being conservative (SPEC 6.2).
    """
    y_true = np.asarray(y_true)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    k = min(k, len(y_true))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y_true[top].sum() / positives)


@dataclass
class Report:
    """One evaluation of one score vector against one label vector."""

    label: str
    n: int
    positives: int
    prevalence: float
    pr_auc: float
    pr_auc_baseline: float
    normalised_ap: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    brier: float
    logloss: float
    precision_at: dict = field(default_factory=dict)
    recall_at: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {
            "label": self.label, "n": self.n, "positives": self.positives,
            "prevalence": self.prevalence, "pr_auc": self.pr_auc,
            "pr_auc_baseline": self.pr_auc_baseline,
            "normalised_ap": self.normalised_ap, "roc_auc": self.roc_auc,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "brier": self.brier, "logloss": self.logloss,
        }
        out.update({f"precision_at_{k}": v for k, v in self.precision_at.items()})
        out.update({f"recall_at_{k}": v for k, v in self.recall_at.items()})
        return out


def evaluate(label: str, y_true, scores, threshold: float = 0.5,
             with_recall_at_k: bool = False) -> Report:
    """Full metric set for one score vector.

    ``threshold`` only affects the point-estimate precision/recall/F1. The
    ranking metrics and the calibration metrics are threshold-free.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    predicted = (scores >= threshold).astype(int)
    n_ap, _ = normalised_average_precision(y_true, scores)

    both_classes = 0 < y_true.sum() < len(y_true)
    return Report(
        label=label, n=int(len(y_true)), positives=int(y_true.sum()),
        prevalence=prevalence,
        pr_auc=float(average_precision_score(y_true, scores)) if both_classes else float("nan"),
        pr_auc_baseline=prevalence,
        normalised_ap=n_ap,
        roc_auc=float(roc_auc_score(y_true, scores)) if both_classes else float("nan"),
        precision=float(precision_score(y_true, predicted, zero_division=0)),
        recall=float(recall_score(y_true, predicted, zero_division=0)),
        f1=float(f1_score(y_true, predicted, zero_division=0)),
        brier=float(brier_score_loss(y_true, scores)),
        logloss=float(log_loss(y_true, np.clip(scores, 1e-15, 1 - 1e-15), labels=[0, 1])),
        precision_at={k: precision_at_k(y_true, scores, k) for k in K_VALUES},
        recall_at=(
            {k: recall_at_k(y_true, scores, k) for k in K_VALUES}
            if with_recall_at_k else {}
        ),
    )


def calibration_curve(y_true, scores, bins: int = 10) -> pd.DataFrame:
    """Observed frequency against mean predicted probability, per bin."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        rows.append({
            "bin": b, "lower": edges[b], "upper": edges[b + 1],
            "n": int(mask.sum()),
            "mean_predicted": float(scores[mask].mean()),
            "observed_frequency": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)
