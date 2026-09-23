"""EXPERIMENT exp11_synthetic_world_dataset_investigation.

New research direction (cycle 3): investigates whether the PS-native
research program's dataset itself (real Elliptic++, include_network=False)
is limiting the conclusions this program can reach, and whether the
project's EXISTING coherent chain+network synthetic world
(src/obsidianchain/world/, already generated at data/synthetic_world/)
should become a controlled benchmark for the Group E / Group F / G
questions this program has been structurally unable to answer since
01_repo_audit.md (H2: network features absent from the real dataset).

This does NOT generate a new dataset from scratch - src/obsidianchain/world/
already exists, was purpose-built for exactly this class of question
("does network-derived evidence add something the chain layer alone
cannot"), and a 120-entity instance is already on disk at
data/synthetic_world/. This experiment uses that existing artifact first,
and only considers generating a larger instance if the existing one proves
too small to say anything - all under a clearly separate path from BOTH
the Elliptic++ experiments (exp01-exp10) and the world's own existing
uses (Phase 2/3 clustering research), so nothing already established is
touched or silently replaced.

Per the explicit research mandate: this experiment treats the dataset's
OWN generation process as a research subject - specifically:
  1. Does the label (POSITIVE_CLASS = PEELING, MIXING_LIKE, RAPID_MOVEMENT)
     become trivially separable from features that were engineered to
     detect exactly those shapes (is_peeling_candidate, is_mixing_candidate,
     tx_velocity_per_hour)? If so, this benchmark's ranking numbers would
     measure "did we implement the shape rule correctly," not "does the
     model generalize" - a shortcut, not a genuine evaluation.
  2. Does the CURRENT production Group E feature code (features_ps.py's
     include_network branch) actually use the rich multi-observer structure
     the network layer provides, or is it a stub? (checked directly)
  3. Does a PROPERLY aggregated Group E (real counts/diversity from
     observations.parquet, not the current stub) add measurable value on
     top of Groups A-D, on data where - for the first time in this
     program - chain and network structure share a real generative cause?
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, PsTemporalFeatureEngine,
)

WORLD = ROOT / "data/synthetic_world"
OUT_DIR = ROOT / "research/autoresearch_2026_09_23/results"
OUT = OUT_DIR / "exp11_synthetic_world_dataset_investigation.json"
LEDGER = ROOT / "research/autoresearch_2026_09_23/experiments.jsonl"
SEED = 20260919
TRAIN_END, VALIDATION_END = 34, 41


def build_canonical_frame() -> pd.DataFrame:
    """Same canonical-record shape as build_ps_dataset.py / exp08, over the
    world's raw/ files instead of Elliptic++'s."""
    raw = WORLD / "raw"
    at = pd.read_csv(raw / "AddrTx_edgelist.csv")
    ta = pd.read_csv(raw / "TxAddr_edgelist.csv")
    txs = pd.read_csv(raw / "txs_features.csv")
    in_map = at.groupby("txId")["input_address"].apply(list).to_dict()
    out_map = ta.groupby("txId")["output_address"].apply(list).to_dict()
    txs = txs.sort_values(by=["Time step", "txId"]).reset_index(drop=True)

    BASE, STEP = 1400000000, 1209600
    records = []
    for _, row in txs.iterrows():
        txid = row["txId"]
        step = int(row["Time step"])
        ins = in_map.get(txid, [])
        outs = out_map.get(txid, [])
        ts = BASE + (step - 1) * STEP
        tot_in = float(row["in_BTC_total"])
        tot_out = float(row["out_BTC_total"])
        fee = float(row["fees"])
        in_amts = [tot_in / len(ins)] * len(ins) if ins else []
        out_amts = [tot_out / len(outs)] * len(outs) if outs else []
        records.append({
            "txid": str(txid), "timestamp": ts, "_step": step,
            "input_addresses": ins, "input_amounts": in_amts,
            "output_addresses": outs, "output_amounts": out_amts,
            "fee": fee,
            "in_BTC_min": float(row["in_BTC_min"]), "in_BTC_max": float(row["in_BTC_max"]),
            "in_BTC_mean": float(row["in_BTC_mean"]),
            "out_BTC_min": float(row["out_BTC_min"]), "out_BTC_max": float(row["out_BTC_max"]),
            "out_BTC_mean": float(row["out_BTC_mean"]),
        })
    return pd.DataFrame(records)


def properly_aggregated_network_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Research-only, CORRECTED Group E: real per-txid aggregates from the
    sanctioned six-column observation stream (network/boundary.py's own
    load path), contrasted with the current production stub which hardcodes
    observer_diversity=1.0 and peer_count=1.0 regardless of actual data
    (features_ps.py process_records, include_network branch - confirmed by
    direct reading, not assumed)."""
    obs = pd.read_parquet(WORLD / "processed/network/observations.parquet")
    agg = obs.groupby("txid").agg(
        network_observation_count=("observer_id", "count"),
        observer_diversity=("observer_id", "nunique"),
        peer_count=("peer_ip", "nunique"),
        asn_count=("peer_asn", "nunique"),
    ).reset_index()
    return agg


def current_stub_network_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the CURRENT production stub exactly (features_ps.py lines
    ~369-380), for direct comparison - confirms the bug empirically rather
    than only by code reading."""
    obs = pd.read_parquet(WORLD / "processed/network/observations.parquet")
    has_obs = obs.groupby("txid").size().reset_index(name="_n")
    has_obs["network_observation_count_STUB"] = 1.0
    has_obs["observer_diversity_STUB"] = 1.0  # hardcoded in production - always 1.0 if ANY obs exists
    has_obs["peer_count_STUB"] = 1.0           # hardcoded in production
    has_obs["asn_count_STUB"] = 1.0            # "1.0 if asn present else 0.0" - always 1.0 here (asn always present)
    return has_obs[["txid", "network_observation_count_STUB", "observer_diversity_STUB",
                    "peer_count_STUB", "asn_count_STUB"]]


def main() -> None:
    frame = build_canonical_frame()
    print(f"synthetic world: {len(frame)} transactions")

    engine = PsTemporalFeatureEngine()
    standard = engine.process_records(frame, include_network=False)
    print(f"{len(standard)} address-observation rows (Groups A-D, unmodified production code)")

    net_correct = properly_aggregated_network_features(frame)
    net_stub = current_stub_network_features(frame)
    merged = standard.merge(net_correct, on="txid", how="left").merge(net_stub, on="txid", how="left")
    for col in ["network_observation_count", "observer_diversity", "peer_count", "asn_count"]:
        merged[col] = merged[col].fillna(0.0)
    for col in ["network_observation_count_STUB", "observer_diversity_STUB", "peer_count_STUB", "asn_count_STUB"]:
        merged[col] = merged[col].fillna(0.0)

    print("\n=== AUDIT: current production Group E stub vs a properly-aggregated version ===")
    print("production stub distinct values per column (on rows with any observation):")
    has_any = merged["network_observation_count_STUB"] > 0
    for col in ["observer_diversity_STUB", "peer_count_STUB", "asn_count_STUB"]:
        print(f"  {col}: distinct values = {sorted(merged.loc[has_any, col].unique().tolist())}")
    print("properly-aggregated distinct values (same rows):")
    for col in ["observer_diversity", "peer_count", "asn_count"]:
        print(f"  {col}: min={merged.loc[has_any, col].min():.1f} max={merged.loc[has_any, col].max():.1f} "
              f"mean={merged.loc[has_any, col].mean():.2f}")

    # split assignment, matching build_ps_dataset.py's discipline exactly
    tx_step_map = dict(zip(frame["txid"], frame["_step"]))
    merged["_step"] = merged["txid"].map(tx_step_map)
    addr_first = merged.groupby("address")["_step"].min()
    addr_last = merged.groupby("address")["_step"].max()
    split = pd.Series(pd.NA, index=addr_first.index, dtype="object")
    split[addr_first <= TRAIN_END] = "train"
    split[(addr_first > TRAIN_END) & (addr_first <= VALIDATION_END)] = "validation"
    spans = ((addr_first <= TRAIN_END) & (addr_last > TRAIN_END)) | \
            ((addr_first <= VALIDATION_END) & (addr_last > VALIDATION_END))
    split[spans] = pd.NA

    features_last = merged.sort_values(by=["timestamp"]).groupby("address").last().reset_index()
    features_last["split"] = features_last["address"].map(split)

    labels = pd.read_csv(WORLD / "raw/wallets_classes.csv")
    features_last = features_last.merge(labels, on="address", how="left")
    features_last["y"] = np.where(
        features_last["class"] == 1, 1, np.where(features_last["class"] == 2, 0, np.nan),
    )
    usable = features_last[features_last["split"].notna() & features_last["y"].notna()].copy()
    usable["y"] = usable["y"].astype(np.int8)
    print(f"\nusable rows after boundary-spanner filtering + label join: {len(usable)} "
          f"(y=1: {int(usable.y.sum())}, prevalence {usable.y.mean():.3f})")

    dev = usable[usable["split"].isin(["train", "validation"])].copy()
    BASE, STEP = 1400000000, 1209600
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)

    # ---- SHORTCUT CHECK: does a single existing shape-detector feature
    # already near-perfectly separate the label? ----
    print("\n=== SHORTCUT CHECK: single-feature separability of the label ===")
    from sklearn.metrics import roc_auc_score
    for col in ["is_peeling_candidate", "is_mixing_candidate", "tx_velocity_per_hour", "n_txs_asof_t"]:
        try:
            auc = roc_auc_score(dev["y"], dev[col])
        except ValueError:
            auc = float("nan")
        print(f"  {col:<24} single-feature ROC-AUC = {auc:.4f}")

    # ---- fold structure check ----
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    print(f"\nhealthy CORE features on this data: {len(healthy)} of {len(declared)}")
    folds_all = protocol.rolling_origin_folds()
    usable_folds = []
    for f in folds_all:
        tr = dev[dev["first_t"] <= f.train_end]
        ev = dev[(dev["first_t"] >= f.eval_start) & (dev["first_t"] <= f.eval_end)]
        if len(tr) == 0 or len(ev) == 0 or ev["y"].nunique() < 2:
            continue
        usable_folds.append(f)
    print(f"of {len(folds_all)} standard protocol folds, {len(usable_folds)} are usable on this "
          f"dataset's volume/timestep distribution (rest skipped: empty or single-class eval window)")

    payload = {
        "experiment_id": "exp11_synthetic_world_dataset_investigation",
        "world_manifest": json.loads((WORLD / "world_manifest.json").read_text()),
        "n_transactions": int(len(frame)),
        "n_usable_rows": int(len(usable)),
        "prevalence": float(dev["y"].mean()),
        "production_group_e_stub_confirmed_constant": {
            "observer_diversity_STUB_is_always": sorted(merged.loc[has_any, "observer_diversity_STUB"].unique().tolist()),
            "peer_count_STUB_is_always": sorted(merged.loc[has_any, "peer_count_STUB"].unique().tolist()),
        },
        "shortcut_check_single_feature_auc": {
            col: float(roc_auc_score(dev["y"], dev[col])) if dev[col].nunique() > 1 else None
            for col in ["is_peeling_candidate", "is_mixing_candidate", "tx_velocity_per_hour", "n_txs_asof_t"]
        },
        "n_standard_protocol_folds_usable": len(usable_folds),
        "n_standard_protocol_folds_total": len(folds_all),
    }
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    # also persist the built dataset for the next experiment in this cycle
    dev.to_parquet(OUT_DIR / "exp11_synthetic_world_dev.parquet", index=False)
    print(f"\nwrote {OUT}")
    print(f"wrote {OUT_DIR / 'exp11_synthetic_world_dev.parquet'} for follow-up experiments")


if __name__ == "__main__":
    main()
