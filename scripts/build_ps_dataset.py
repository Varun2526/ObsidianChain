"""Build the PS-Native Supervised ML Training, Validation, and Test Datasets.

Transforms local Elliptic++ raw graph and wallet labels into canonical
PS-compatible address-as-of-t feature matrices:
    train.parquet       (timesteps 1-34)
    validation.parquet  (timesteps 35-41)
    test.parquet        (timesteps 42-49)
    manifest.json

PROVENANCE:
    Research-derived canonical training representation constructed from:
    data/raw/AddrTx_edgelist.csv
    data/raw/TxAddr_edgelist.csv
    data/raw/txs_features.csv
    data/raw/wallets_classes.csv

    Timestamp conversion uses deterministic surrogate:
    t = 1400000000 + (Time_step - 1) * 1209600
    (labeled explicitly as ELLIPTIC_TIMESTEP_SURROGATE).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    PsTemporalFeatureEngine,
)

BASE_TIMESTAMP = 1400000000
TIMESTEP_SECONDS = 1209600  # 14 days in seconds

TRAIN_END = 34
VALIDATION_END = 41
TEST_END = 49


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_ps_dataset(
    data_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    t_start = time.time()
    root = Path(data_root) if data_root is not None else Path(os.environ.get("OBSIDIANCHAIN_DATA", "data"))
    out = Path(output_dir) if output_dir is not None else root / "models" / "ps_native" / "datasets"
    out.mkdir(parents=True, exist_ok=True)

    raw_dir = root / "raw"
    addr_tx_path = raw_dir / "AddrTx_edgelist.csv"
    tx_addr_path = raw_dir / "TxAddr_edgelist.csv"
    tx_feat_path = raw_dir / "txs_features.csv"
    wallets_classes_path = raw_dir / "wallets_classes.csv"

    for p in [addr_tx_path, tx_addr_path, tx_feat_path, wallets_classes_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required raw dataset not found: {p}")

    print("Hashing source raw datasets...")
    source_hashes = {
        "AddrTx_edgelist.csv": _compute_sha256(addr_tx_path),
        "TxAddr_edgelist.csv": _compute_sha256(tx_addr_path),
        "txs_features.csv": _compute_sha256(tx_feat_path),
        "wallets_classes.csv": _compute_sha256(wallets_classes_path),
    }

    print("Loading raw files...")
    at = pd.read_csv(addr_tx_path)
    ta = pd.read_csv(tx_addr_path)
    txs = pd.read_csv(
        tx_feat_path,
        usecols=["txId", "Time step", "fees", "in_BTC_total", "out_BTC_total", "total_BTC"],
    )
    labels = pd.read_csv(wallets_classes_path)

    print("Grouping input/output addresses per txId...")
    in_map = at.groupby("txId")["input_address"].apply(list).to_dict()
    out_map = ta.groupby("txId")["output_address"].apply(list).to_dict()

    print("Sorting transactions chronologically by (Time step, txId)...")
    txs = txs.sort_values(by=["Time step", "txId"], ascending=True).reset_index(drop=True)

    # Build canonical records
    records: list[dict] = []
    print("Converting to canonical PS stream...")
    for _, row in txs.iterrows():
        txid = row["txId"]
        step = int(row["Time step"])
        ins = in_map.get(txid, [])
        outs = out_map.get(txid, [])
        ts = BASE_TIMESTAMP + (step - 1) * TIMESTEP_SECONDS

        tot_in = float(row["in_BTC_total"]) if pd.notna(row["in_BTC_total"]) else float(row["total_BTC"])
        tot_out = float(row["out_BTC_total"]) if pd.notna(row["out_BTC_total"]) else float(row["total_BTC"])
        fee = float(row["fees"]) if pd.notna(row["fees"]) else 0.0

        in_amts = [tot_in / len(ins)] * len(ins) if ins else []
        out_amts = [tot_out / len(outs)] * len(outs) if outs else []

        records.append({
            "txid": str(txid),
            "timestamp": ts,
            "_step": step,
            "input_addresses": ins,
            "input_amounts": in_amts,
            "output_addresses": outs,
            "output_amounts": out_amts,
            "fee": fee,
            "script_type": "p2pkh",
        })

    frame = pd.DataFrame(records)
    print(f"Executing PsTemporalFeatureEngine on {len(frame)} canonical records...")
    engine = PsTemporalFeatureEngine()
    features = engine.process_records(frame, include_network=False)
    print(f"Extracted {len(features)} address observation rows in {time.time()-t_start:.2f}s.")

    # Attach timestep from frame
    tx_step_map = dict(zip(frame["txid"], frame["_step"]))
    features["_step"] = features["txid"].map(tx_step_map)

    # Compute first and last active timestep per address
    addr_first_step = features.groupby("address")["_step"].min()
    addr_last_step = features.groupby("address")["_step"].max()

    # Assign split and drop boundary-spanning addresses
    print("Assigning split boundaries and filtering spanners...")
    split_series = pd.Series(pd.NA, index=addr_first_step.index, dtype="object")
    split_series[addr_first_step <= TRAIN_END] = "train"
    split_series[(addr_first_step > TRAIN_END) & (addr_first_step <= VALIDATION_END)] = "validation"
    split_series[addr_first_step > VALIDATION_END] = "test"

    # Spanners across boundaries get NA
    spans_train_val = (addr_first_step <= TRAIN_END) & (addr_last_step > TRAIN_END)
    spans_val_test = (addr_first_step <= VALIDATION_END) & (addr_last_step > VALIDATION_END)
    split_series[spans_train_val | spans_val_test] = pd.NA

    # Keep only the last observation per address (observation point as-of last_t)
    features_last = features.sort_values(by=["timestamp"], ascending=True).groupby("address").last().reset_index()
    features_last["split"] = features_last["address"].map(split_series)

    # Merge labels (joined last, label-blind)
    print("Joining wallet ground-truth labels...")
    features_last = features_last.merge(labels, on="address", how="left")

    # Target y: Class 1 = 1, Class 2 = 0, Class 3 = NaN (excluded)
    features_last["y"] = np.where(
        features_last["class"] == 1, 1,
        np.where(features_last["class"] == 2, 0, np.nan)
    )

    # Filter to usable rows: non-spanning and explicitly labeled
    usable = features_last[features_last["split"].notna() & features_last["y"].notna()].copy()
    usable["y"] = usable["y"].astype(np.int8)

    save_cols = ["address", "txid", "timestamp", "y"] + CORE_PS_FEATURE_COLUMNS

    splits_data = {}
    row_counts = {}
    pos_counts = {}
    neg_counts = {}
    artifact_hashes = {}

    for s_name in ["train", "validation", "test"]:
        s_df = usable[usable["split"] == s_name][save_cols].reset_index(drop=True)
        s_path = out / f"{s_name}.parquet"
        s_df.to_parquet(s_path, index=False)
        splits_data[s_name] = s_df
        row_counts[s_name] = int(len(s_df))
        pos_counts[s_name] = int((s_df["y"] == 1).sum())
        neg_counts[s_name] = int((s_df["y"] == 0).sum())
        artifact_hashes[f"{s_name}.parquet"] = _compute_sha256(s_path)
        print(f"Saved {s_name}.parquet: {len(s_df)} rows ({pos_counts[s_name]} pos, {neg_counts[s_name]} neg).")

    manifest = {
        "schema": "obsidianchain.ps_dataset_manifest/1",
        "provenance_type": "RESEARCH_DERIVED_CANONICAL_TRAINING_REPRESENTATION",
        "generator": "scripts/build_ps_dataset.py",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_dataset_hashes": source_hashes,
        "feature_schema_version": "ps_native_features/1",
        "feature_list": CORE_PS_FEATURE_COLUMNS,
        "label_schema": {
            "0": "licit (class 2)",
            "1": "illicit (class 1)",
            "excluded": "unknown (class 3)",
        },
        "timestamp_source": "ELLIPTIC_TIMESTEP_SURROGATE",
        "timestamp_conversion_rule": f"t = {BASE_TIMESTAMP} + (step - 1) * {TIMESTEP_SECONDS}",
        "split_boundaries": {
            "train": f"step 1-{TRAIN_END}",
            "validation": f"step {TRAIN_END+1}-{VALIDATION_END}",
            "test": f"step {VALIDATION_END+1}-{TEST_END}",
        },
        "row_counts": row_counts,
        "positive_counts": pos_counts,
        "negative_counts": neg_counts,
        "artifact_hashes": artifact_hashes,
        "total_usable_rows": sum(row_counts.values()),
        "duration_seconds": round(time.time() - t_start, 2),
    }

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path} in {time.time()-t_start:.2f}s total.")
    return manifest


if __name__ == "__main__":
    build_ps_dataset()
