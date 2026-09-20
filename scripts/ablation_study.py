"""Run ablation study across PS feature subsets on Train/Validation splits."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from obsidianchain.pipeline.features_ps import (
    GROUP_A_TRANSACTION,
    GROUP_B_ADDRESS_HISTORY,
    GROUP_C_GRAPH,
    GROUP_D_PATTERNS,
)

DATASET_DIR = Path("data/models/ps_native/datasets")
OUT_FILE = Path("data/models/ps_native/v1/ablation.json")

train_df = pd.read_parquet(DATASET_DIR / "train.parquet")
val_df = pd.read_parquet(DATASET_DIR / "validation.parquet")

y_train = train_df["y"].values
y_val = val_df["y"].values

feature_subsets = {
    "A_behaviour_only": GROUP_A_TRANSACTION + GROUP_B_ADDRESS_HISTORY,
    "B_behaviour_plus_graph": GROUP_A_TRANSACTION + GROUP_B_ADDRESS_HISTORY + GROUP_C_GRAPH,
    "C_behaviour_graph_patterns": GROUP_A_TRANSACTION + GROUP_B_ADDRESS_HISTORY + GROUP_C_GRAPH + GROUP_D_PATTERNS,
}

results = {}

for name, cols in feature_subsets.items():
    X_train = train_df[cols].fillna(0.0).values
    X_val = val_df[cols].fillna(0.0).values

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_probs = rf.predict_proba(X_val)[:, 1]

    pr_auc = float(average_precision_score(y_val, val_probs))
    roc_auc = float(roc_auc_score(y_val, val_probs))
    brier = float(brier_score_loss(y_val, val_probs))

    top100_idx = np.argsort(val_probs)[::-1][:100]
    p_at_100 = float(np.mean(y_val[top100_idx]))

    results[name] = {
        "features_count": len(cols),
        "features": cols,
        "val_pr_auc": pr_auc,
        "val_roc_auc": roc_auc,
        "val_brier": brier,
        "precision_at_100": p_at_100,
    }
    print(f"[{name}] (n={len(cols)}) -> PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}, P@100: {p_at_100:.4f}")

# Group D note for network features
results["D_with_network"] = {
    "features_count": len(GROUP_A_TRANSACTION + GROUP_B_ADDRESS_HISTORY + GROUP_C_GRAPH + GROUP_D_PATTERNS) + 4,
    "note": "Network telemetry features (Group E) are unobserved in the Elliptic++ research base dataset and are natively represented as NaN. Network ablation cannot be fairly evaluated on this split.",
    "status": "UNAVAILABLE_ON_BASE_SPLIT"
}

OUT_FILE.write_text(json.dumps(results, indent=2))
print("Ablation results saved to", OUT_FILE)
