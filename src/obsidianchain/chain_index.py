"""Persist the observed transaction graph the investigation API needs.

Why this exists
---------------
Money-flow tracing needs the address-transaction graph: which addresses
spent into a transaction and which it paid. The alert artifacts only carry
the slice of it that touches an alert, so tracing stopped at an alert's
edge. The read-only API may not read ``raw/`` or recompute anything (its
contract is "a response is a file that was already written"), so the graph
is written here, once, by an offline command:

    make run ARGS="build-chain-index"

What is written (``<data_root>/processed/``)
--------------------------------------------
``chain_edges.parquet``        address, txid, role (input / output), timestep
                               one row per address-transaction incidence
``chain_transactions.parquet`` txid, timestep, fee_btc, in_btc, out_btc,
                               n_inputs, n_outputs
``watchlist_seeds.parquet``    address, source, label: OFAC SDN Bitcoin
                               addresses and analyst watchlists, the same
                               offline sources the pipeline seeds propagation
                               with (io/watchlist.py). External attributions,
                               labelled as such.

What is deliberately NOT written
--------------------------------
Nothing from ``wallets_classes.csv`` or ``txs_classes.csv``. Those are the
Elliptic++ illicit / licit labels: the models' training and evaluation
ground truth. Serving them in an investigation view would present a hidden
label as an observed fact. Only observed chain structure and per-transaction
value summaries are persisted.

Per-address amounts are not written either: the source has per-transaction
totals only (see ``research/reproduction/build_ps_dataset.py``), and an even
split would be a fabricated number.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from obsidianchain import provenance as prov

CHAIN_EDGES = "chain_edges.parquet"
CHAIN_TRANSACTIONS = "chain_transactions.parquet"
WATCHLIST_SEEDS = "watchlist_seeds.parquet"
INDEX_VERSION = "1.0.0"
ARTIFACT_SCHEMA = "obsidianchain.chain_index/1"

_TX_COLUMNS = ["txId", "Time step", "fees", "in_BTC_total", "out_BTC_total"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_tables(addr_tx: pd.DataFrame, tx_addr: pd.DataFrame, txs: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure transform, so tests can build from hand-made frames."""
    step = txs.set_index("txId")["Time step"]
    ins = pd.DataFrame({"address": addr_tx["input_address"].astype(str),
                        "txid": addr_tx["txId"].astype("int64"), "role": "input"})
    outs = pd.DataFrame({"address": tx_addr["output_address"].astype(str),
                         "txid": tx_addr["txId"].astype("int64"), "role": "output"})
    edges = pd.concat([ins, outs], ignore_index=True).drop_duplicates()
    edges["timestep"] = edges["txid"].map(step).astype("Int16")
    edges = edges.sort_values(["address", "timestep", "txid", "role"], kind="stable").reset_index(drop=True)

    counts_in = ins.groupby("txid").size()
    counts_out = outs.groupby("txid").size()
    tx = pd.DataFrame({
        "txid": txs["txId"].astype("int64"),
        "timestep": txs["Time step"].astype("int16"),
        "fee_btc": pd.to_numeric(txs["fees"], errors="coerce"),
        "in_btc": pd.to_numeric(txs["in_BTC_total"], errors="coerce"),
        "out_btc": pd.to_numeric(txs["out_BTC_total"], errors="coerce"),
    })
    tx["n_inputs"] = tx["txid"].map(counts_in).fillna(0).astype("int32")
    tx["n_outputs"] = tx["txid"].map(counts_out).fillna(0).astype("int32")
    tx = tx.sort_values("txid").reset_index(drop=True)
    return edges, tx


def build(data_root, processed_root=None) -> dict:
    data_root = Path(data_root)
    raw = data_root / "raw"
    processed_root = Path(processed_root) if processed_root is not None else data_root / "processed"
    files = {"AddrTx_edgelist.csv": raw / "AddrTx_edgelist.csv",
             "TxAddr_edgelist.csv": raw / "TxAddr_edgelist.csv",
             "txs_features.csv": raw / "txs_features.csv"}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; the chain index is built from Elliptic++ raw files")
    edges, tx = build_tables(pd.read_csv(files["AddrTx_edgelist.csv"]),
                             pd.read_csv(files["TxAddr_edgelist.csv"]),
                             pd.read_csv(files["txs_features.csv"], usecols=_TX_COLUMNS))
    inputs = {f"{name}_sha256": _sha256(path) for name, path in files.items()}
    ofac = raw / "ofac_sdn" / "sdn_xml.zip"
    inputs["ofac_sdn_sha256"] = _sha256(ofac) if ofac.is_file() else "absent"
    inputs["chain_index_version"] = INDEX_VERSION
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    record = prov.Provenance(
        provenance_type=prov.ProvenanceType.PRODUCTION,
        dataset_id="elliptic++",
        synthetic_network=None,
        generator_version=INDEX_VERSION,
        inputs=inputs,
        artifact={"artifact_schema": ARTIFACT_SCHEMA,
                  "contains_labels": False,
                  "per_address_amounts": "not available in the source; not written"},
        run_fingerprint=fingerprint,
        notes=("Observed chain structure and per-transaction value totals only.",
               "No wallet or transaction class labels are included."),
    )
    from obsidianchain.io import watchlist
    seeds = watchlist.load_default_seeds(raw / "ofac_sdn" / "sdn_xml.zip", data_root / "watchlists")
    seeds_frame = pd.DataFrame([{"address": x.address, "source": x.source, "label": x.label} for x in seeds],
                               columns=["address", "source", "label"])
    out = {}
    for name, frame in ((CHAIN_EDGES, edges), (CHAIN_TRANSACTIONS, tx), (WATCHLIST_SEEDS, seeds_frame)):
        path = prov.write_frame(frame, processed_root / name, record)
        out[name] = {"path": str(path), "rows": len(frame)}
    out["run_fingerprint"] = fingerprint
    out["addresses"] = int(edges["address"].nunique())
    return out
