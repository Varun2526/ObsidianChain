"""The Phase 6 experiment: M0-M3 ablation, cluster aggregation, P1-P4.

SPEC 6.4 and 6.5. Every threshold, every label definition and every
prediction was fixed in the specification before this module existed, so
nothing here chooses anything on the strength of a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from obsidianchain.features import dataset
from obsidianchain.ml import metrics, model

#: SPEC 6.4. Cumulative, in order.
STAGES = (
    ("M0", ("M0",)),
    ("M0+M1", ("M0", "M1")),
    ("M0+M1+M2", ("M0", "M1", "M2")),
    ("M0+M1+M2+M3", ("M0", "M1", "M2", "M3")),
)

#: SPEC 2. Four candidates, evaluated on the same test set.
AGGREGATIONS = ("AGG-MAX", "AGG-TOPK", "AGG-QUANT", "AGG-WMEAN")
TOPK = 3
QUANTILE = 0.90

#: SPEC 2.1. Primary derived cluster label, plus the registered sensitivity.
CLUSTER_LABEL_PRIMARY = 0.10
CLUSTER_LABEL_SENSITIVITY = "at_least_one"

#: SPEC 6.5 P3. The test split is reported whole and also cut at t43.
SHUTDOWN_T = 43


@dataclass
class StageResult:
    name: str
    features: list[str]
    test: metrics.Report
    validation: metrics.Report
    bands: list


def run_ablation(frame: pd.DataFrame) -> list[StageResult]:
    """Fit and evaluate each cumulative stage on an identical split."""
    train = frame[frame["split"] == dataset.SPLIT_TRAIN]
    validation = frame[frame["split"] == dataset.SPLIT_VALIDATION]
    test = frame[frame["split"] == dataset.SPLIT_TEST]

    results = []
    for name, groups in STAGES:
        features = dataset.feature_columns(groups)
        trained = model.fit(train, validation, features)
        results.append(StageResult(
            name=name, features=features,
            test=metrics.evaluate(name, test["y"], trained.scores(test)),
            validation=metrics.evaluate(
                f"{name}/validation", validation["y"], trained.scores(validation)
            ),
            bands=trained.bands,
        ))
    return results


def fit_full(frame: pd.DataFrame) -> tuple[model.TrainedModel, pd.DataFrame]:
    """The full-feature model plus the scored test split."""
    train = frame[frame["split"] == dataset.SPLIT_TRAIN]
    validation = frame[frame["split"] == dataset.SPLIT_VALIDATION]
    test = frame[frame["split"] == dataset.SPLIT_TEST].copy()
    features = dataset.feature_columns(("M0", "M1", "M2", "M3"))
    trained = model.fit(train, validation, features)
    test["risk"] = trained.scores(test)
    test["severity"] = trained.severity(test["risk"])
    return trained, test


# ---- cluster aggregation ------------------------------------------------


def aggregate_clusters(scored_test: pd.DataFrame, clusters: pd.DataFrame
                       ) -> pd.DataFrame:
    """Cluster-level scores and derived labels, TEST members only.

    Restricting to test members keeps the cluster evaluation inside the split
    that the model never saw. A cluster scored partly from train members
    would be scored partly from addresses the model was fitted on, and its
    label would then describe a different population from its score.
    """
    joined = scored_test.merge(clusters, on="address", how="inner")
    grouped = joined.groupby("cluster_id")

    out = pd.DataFrame(index=grouped.size().index)
    out["n_members"] = grouped.size()
    out["illicit"] = grouped["y"].sum()
    out["illicit_share"] = grouped["y"].mean()

    out["AGG-MAX"] = grouped["risk"].max()
    out["AGG-TOPK"] = grouped["risk"].apply(
        lambda s: s.nlargest(min(TOPK, len(s))).mean()
    )
    out["AGG-QUANT"] = grouped["risk"].quantile(QUANTILE)
    weights = joined["n_txs_asof_t"].fillna(0.0) + 1.0
    joined = joined.assign(_w=weights, _wr=joined["risk"] * weights)
    wsum = joined.groupby("cluster_id")["_w"].sum()
    out["AGG-WMEAN"] = joined.groupby("cluster_id")["_wr"].sum() / wsum

    out["label_primary"] = (out["illicit_share"] >= CLUSTER_LABEL_PRIMARY).astype(int)
    out["label_sensitivity"] = (out["illicit"] >= 1).astype(int)
    return out.reset_index()


def evaluate_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    """Precision@K and Recall@K per aggregation, under both labels."""
    rows = []
    for label_name, column in (
        ("primary_share>=0.10", "label_primary"),
        ("sensitivity_>=1_illicit", "label_sensitivity"),
    ):
        for agg in AGGREGATIONS:
            report = metrics.evaluate(
                f"{agg}/{label_name}", clusters[column], clusters[agg],
                with_recall_at_k=True,
            )
            row = report.as_dict()
            row["aggregation"] = agg
            row["cluster_label"] = label_name
            rows.append(row)
    return pd.DataFrame(rows)


# ---- the pre-registered predictions -------------------------------------


def evaluate_predictions(stages: list[StageResult], clusters: pd.DataFrame,
                         scored_test: pd.DataFrame) -> pd.DataFrame:
    """P1-P4, each reported as held or refuted against its frozen threshold."""
    by_name = {s.name: s for s in stages}
    rows = []

    delta3 = by_name["M0+M1+M2+M3"].test.pr_auc - by_name["M0+M1+M2"].test.pr_auc
    rows.append({
        "prediction": "P1",
        "statement": "delta(M3-M2) PR-AUC within +/-0.005 of zero",
        "observed": delta3,
        "threshold": 0.005,
        "held": bool(abs(delta3) <= 0.005),
    })

    primary = clusters
    p50 = {}
    for agg in ("AGG-MAX", "AGG-TOPK"):
        p50[agg] = metrics.precision_at_k(
            primary["label_primary"], primary[agg], 50
        )
    rows.append({
        "prediction": "P2",
        "statement": "AGG-MAX ranks below AGG-TOPK on cluster Precision@50",
        "observed": p50["AGG-MAX"] - p50["AGG-TOPK"],
        "threshold": 0.0,
        "held": bool(p50["AGG-MAX"] < p50["AGG-TOPK"]),
    })

    early = scored_test[scored_test["first_t"] < SHUTDOWN_T]
    late = scored_test[scored_test["first_t"] >= SHUTDOWN_T]
    nap_early, prev_early = metrics.normalised_average_precision(
        early["y"], early["risk"]
    )
    nap_late, prev_late = metrics.normalised_average_precision(
        late["y"], late["risk"]
    )
    drop = nap_early - nap_late
    rows.append({
        "prediction": "P3",
        "statement": "nAP(t42) - nAP(t43-t49) >= 0.10",
        "observed": drop,
        "threshold": 0.10,
        "held": bool(np.isfinite(drop) and drop >= 0.10),
        "detail": (
            f"nAP early {nap_early:.4f} (prevalence {prev_early:.4f}, "
            f"n={len(early):,}); nAP late {nap_late:.4f} "
            f"(prevalence {prev_late:.4f}, n={len(late):,})"
        ),
    })

    d1 = by_name["M0+M1"].test.pr_auc - by_name["M0"].test.pr_auc
    d2 = by_name["M0+M1+M2"].test.pr_auc - by_name["M0+M1"].test.pr_auc
    rows.append({
        "prediction": "P4",
        "statement": "delta(M1-M0) > delta(M2-M1) on PR-AUC",
        "observed": d1 - d2,
        "threshold": 0.0,
        "held": bool(d1 > d2),
    })
    return pd.DataFrame(rows)
