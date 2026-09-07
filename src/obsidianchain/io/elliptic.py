"""Elliptic++ loaders and the address string -> dense integer mapping.

Co-spend evidence comes from ``AddrTx_edgelist.csv`` and nothing else.

    input_address, txId

Every address that is an *input* to the same transaction is co-spending with
every other input of that transaction: building the transaction required the
private key for each one, so a single entity controls all of them. That is the
common-input-ownership heuristic, and connected components of the resulting
graph are candidate entities.

``AddrAddr_edgelist.csv`` is deliberately never read by this module. Its
columns are ``input_address, output_address`` - "A paid B" - which is money
flow, not shared ownership. Union-find over it would merge every payer with
every payee into one meaningless component. It is listed below only so the
name is documented as excluded.

Memory: all work after the initial read is done on int32 arrays. A 34-char
address costs ~83 bytes as a Python string against 4 as an int32 code, and
that cost would otherwise be paid again per edge endpoint and per hash key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---- dataset files ----------------------------------------------------
# Verified against the real download; do not re-derive.
ADDR_TX = "AddrTx_edgelist.csv"          # 477,117 rows: input_address, txId
TX_ADDR = "TxAddr_edgelist.csv"          # 837,124 rows: txId, output_address
WALLETS_CLASSES = "wallets_classes.csv"  # 822,942 rows: address, class

#: Money-flow interaction graph. NEVER a clustering input. See module docstring.
ADDR_ADDR_DO_NOT_CLUSTER = "AddrAddr_edgelist.csv"

#: The one file co-spend may be derived from.
COSPEND_SOURCE = ADDR_TX

DEFAULT_DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))

# int32 caps the address universe at 2.1e9 ids; Elliptic++ has ~1.3e6.
CodeDType = np.int32


@dataclass
class CoSpendGraph:
    """Co-spend edges over a dense integer address space.

    ``edges`` holds star edges, not all-pairs: a transaction with k inputs
    contributes k-1 edges (the first input against each of the others), which
    yields identical connected components to the k(k-1)/2 all-pairs form for a
    fraction of the work.
    """

    n_addresses: int
    """Size of the address universe; also the number of union-find nodes."""

    edges: np.ndarray
    """(m, 2) int32 array of co-spend edges."""

    edge_tx_ids: np.ndarray
    """Raw txId behind each row of ``edges``, for replaying in time order."""

    universe_codes: np.ndarray
    """Address code for each row of wallets_classes.csv, in file order.

    This is what lets a per-address column from that file (the class label,
    say) be reindexed into code space without re-hashing the strings.
    """

    n_transactions: int
    """Distinct transactions contributing at least one input edge."""

    n_input_rows: int
    """Rows read from AddrTx_edgelist.csv, before de-duplication."""

    n_input_pairs: int
    """Distinct (address, txId) input pairs after de-duplication."""

    n_universe_addresses: int
    """Distinct addresses in wallets_classes.csv."""

    n_input_only_addresses: int
    """Addresses seen in AddrTx but absent from wallets_classes.csv."""

    addresses: np.ndarray | None = None
    """Address labels indexed by code, or None when not retained."""

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def address_for(self, code: int) -> str:
        """Map an integer code back to its address string."""
        if self.addresses is None:
            raise ValueError(
                "address labels were not retained; "
                "call load_cospend_graph(keep_labels=True)"
            )
        return str(self.addresses[code])


def find_dataset_file(name: str, data_root: Path | None = None) -> Path:
    """Locate a dataset CSV under ``<data_root>/raw``, searching recursively.

    Extracting the Elliptic++ download into a subdirectory is fine.
    """
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    raw_dir = root / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Copy the Elliptic++ CSVs into it; "
            f"see 'make verify'."
        )
    direct = raw_dir / name
    if direct.is_file():
        return direct
    matches = sorted(p for p in raw_dir.rglob(name) if p.is_file())
    if not matches:
        raise FileNotFoundError(
            f"{name} not found under {raw_dir}. Run 'make verify' to see what "
            f"is present and where to get the dataset."
        )
    return matches[0]


def load_address_universe(data_root: Path | None = None) -> pd.Series:
    """Read the address column of wallets_classes.csv.

    This defines the denominator for coverage: addresses that only ever appear
    on the output side still count as addresses, they simply cannot be
    co-spend clustered.
    """
    path = find_dataset_file(WALLETS_CLASSES, data_root)
    frame = _read_csv_checked(path, ["address"])
    return frame["address"]


def load_input_edges(data_root: Path | None = None) -> pd.DataFrame:
    """Read AddrTx_edgelist.csv - the input-side address/transaction edges."""
    path = find_dataset_file(ADDR_TX, data_root)
    return _read_csv_checked(path, ["input_address", "txId"])


def _read_csv_checked(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read exactly ``columns``, failing clearly if the schema is unexpected."""
    try:
        return pd.read_csv(path, usecols=columns)
    except ValueError as exc:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        raise ValueError(
            f"{path.name} does not have the expected columns.\n"
            f"  expected: {columns}\n"
            f"  found:    {header}\n"
            f"Original error: {exc}"
        ) from exc


def star_edges(
    addr_codes: np.ndarray, tx_codes: np.ndarray
) -> tuple[np.ndarray, int]:
    """Turn (address, transaction) input pairs into star co-spend edges.

    Groups by transaction and links the first input of each transaction to
    every other input of that transaction. A transaction with 500 inputs
    yields 499 edges rather than the 124,750 an all-pairs expansion would
    produce; the connected components are identical either way.

    Transactions with a single input contribute no edges.

    Returns ``(edges, n_transactions)`` where ``edges`` is (m, 2) int32.
    """
    edges, n_transactions, _ = _star_edges_impl(addr_codes, tx_codes)
    return edges, n_transactions


def _star_edges_impl(
    addr_codes: np.ndarray, tx_codes: np.ndarray
) -> tuple[np.ndarray, int, np.ndarray]:
    """As :func:`star_edges`, plus the transaction code behind each edge.

    The extra array is what lets a caller replay the edge list in
    transaction-time order, which the cumulative timestep analysis needs.
    """
    if addr_codes.shape != tx_codes.shape:
        raise ValueError("addr_codes and tx_codes must be the same length")

    dtype = addr_codes.dtype
    if addr_codes.size == 0:
        return np.empty((0, 2), dtype=dtype), 0, np.empty(0, dtype=tx_codes.dtype)

    # Stable sort groups equal transaction ids into contiguous runs while
    # keeping input order within each transaction, so the star centre is
    # deterministic.
    order = np.argsort(tx_codes, kind="stable")
    addrs = addr_codes[order]
    txs = tx_codes[order]

    starts_mask = np.empty(txs.size, dtype=bool)
    starts_mask[0] = True
    np.not_equal(txs[1:], txs[:-1], out=starts_mask[1:])

    start_positions = np.flatnonzero(starts_mask)
    n_transactions = int(start_positions.size)

    # Broadcast each group's first address across the whole group.
    group_index = np.cumsum(starts_mask) - 1
    centres = addrs[start_positions][group_index]

    # Every row except each group's own first row becomes one edge.
    is_centre = np.zeros(txs.size, dtype=bool)
    is_centre[start_positions] = True
    followers = ~is_centre

    edges = np.empty((int(followers.sum()), 2), dtype=dtype)
    edges[:, 0] = centres[followers]
    edges[:, 1] = addrs[followers]
    return edges, n_transactions, txs[followers]


def load_cospend_graph(
    data_root: Path | None = None, keep_labels: bool = False
) -> CoSpendGraph:
    """Load Elliptic++ and derive the co-spend edge list.

    Steps: read the address universe and the input edges, map every address
    string to a dense int32 code with :func:`pandas.factorize`, drop duplicate
    (address, transaction) pairs, then build star edges per transaction.

    ``keep_labels=False`` (the default) discards the ~1.3M address strings
    once codes exist, which is most of the peak memory. Pass True only when
    you need to map a code back to an address.
    """
    universe = load_address_universe(data_root)
    input_frame = load_input_edges(data_root)

    n_universe_rows = len(universe)
    n_universe_addresses = int(universe.nunique())

    # One factorize over universe + input addresses guarantees a single code
    # space, and picks up any input address missing from wallets_classes.csv.
    combined = pd.concat(
        [universe, input_frame["input_address"]], ignore_index=True
    )
    codes, uniques = pd.factorize(combined)
    n_addresses = int(len(uniques))

    if n_addresses > np.iinfo(CodeDType).max:
        raise ValueError(
            f"{n_addresses} addresses exceeds the {CodeDType.__name__} code space"
        )

    addr_codes = codes[n_universe_rows:].astype(CodeDType, copy=False)
    universe_codes = codes[:n_universe_rows].astype(CodeDType, copy=True)
    labels = np.asarray(uniques) if keep_labels else None
    del combined, universe, codes, uniques

    tx_codes_raw, tx_uniques = pd.factorize(input_frame["txId"])
    tx_codes = tx_codes_raw.astype(CodeDType, copy=False)
    tx_ids = np.asarray(tx_uniques)
    n_input_rows = int(len(input_frame))
    del input_frame, tx_codes_raw, tx_uniques

    # An address listed twice as an input to the same transaction is one
    # co-spend fact, not two. Union-find is idempotent so this only saves
    # work, but it keeps the reported edge count honest.
    pairs = pd.DataFrame({"addr": addr_codes, "tx": tx_codes})
    pairs = pairs.drop_duplicates()
    addr_codes = pairs["addr"].to_numpy(dtype=CodeDType, copy=False)
    tx_codes = pairs["tx"].to_numpy(dtype=CodeDType, copy=False)
    n_input_pairs = int(len(pairs))
    del pairs

    edges, n_transactions, edge_tx_codes = _star_edges_impl(addr_codes, tx_codes)
    edge_tx_ids = tx_ids[edge_tx_codes]

    return CoSpendGraph(
        n_addresses=n_addresses,
        edges=edges,
        edge_tx_ids=edge_tx_ids,
        universe_codes=universe_codes,
        n_transactions=n_transactions,
        n_input_rows=n_input_rows,
        n_input_pairs=n_input_pairs,
        n_universe_addresses=n_universe_addresses,
        n_input_only_addresses=n_addresses - n_universe_addresses,
        addresses=labels,
    )
