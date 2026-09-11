"""The address-transaction incidence table: the substrate everything uses.

Why this exists
---------------
SPEC 0.3 established that ``wallets_features.csv`` is a whole-life profile
replicated across timesteps - 52 of its 55 columns already know the address's
future - so address features must be recomputed from the transaction layer.
This module builds the one table that recomputation needs, and every feature
group in Phase 6 reads it rather than the raw files.

``wallets_features.csv`` is never opened by this package. The three columns
SPEC 0.3 found safe are all first-appearance block numbers, and first
appearance is derivable here as ``min(Time step)`` without touching a file
whose other 52 columns leak. Not reading it at all is a stronger guarantee
than reading three columns carefully.

Shape
-----
One row per (address, transaction, role). ``role`` is ``in`` when the address
is an input (it spent) and ``out`` when it is an output (it received). An
address that both funds and receives from the same transaction appears twice,
which is correct: those are two different events.

Addresses are factorised to ``int32`` codes on load. 822,942 Base58 strings
cost about 60 bytes each held as objects; the codes cost four.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Transaction columns Phase 6 may read. The 93 Local_feature_* and 72
#: Aggregate_feature_* columns are deliberately absent: the aggregates are
#: excluded on unresolved provenance (SPEC 0.7, 3.3) and the locals are
#: anonymised and therefore unexplainable (SPEC 3.3).
TX_COLUMNS = [
    "txId", "Time step", "total_BTC", "fees", "size",
    "num_input_addresses", "num_output_addresses",
    "in_BTC_total", "in_BTC_max", "in_BTC_mean",
    "out_BTC_total", "out_BTC_max", "out_BTC_mean",
]

ROLE_IN = 0
ROLE_OUT = 1


@dataclass(frozen=True)
class Incidence:
    """Address-transaction incidence plus the transaction table it indexes."""

    frame: pd.DataFrame
    """Columns: ``code`` (int32), ``txId``, ``role``, ``Time step``."""

    transactions: pd.DataFrame
    """One row per transaction, indexed by ``txId``, holding TX_COLUMNS."""

    addresses: np.ndarray
    """``addresses[code]`` is the Base58 string. Factorisation order."""

    @property
    def n_addresses(self) -> int:
        return int(len(self.addresses))

    def code_of(self) -> pd.Series:
        """address string -> code, for joining a label table on."""
        return pd.Series(
            np.arange(len(self.addresses), dtype=np.int32),
            index=self.addresses,
        )

    def last_timestep(self) -> pd.Series:
        """The last timestep each address was active in.

        This is the observation point ``t`` for that address (SPEC 4.4): once
        boundary-spanning addresses are dropped, every transaction an address
        touches lies inside one split, so aggregating to its last timestep
        cannot reach across a split boundary.
        """
        return self.frame.groupby("code")["Time step"].max()

    def first_timestep(self) -> pd.Series:
        """The first timestep each address was active in. Assigns the split."""
        return self.frame.groupby("code")["Time step"].min()


def load_incidence(data_root: Path) -> Incidence:
    """Build the incidence table from AddrTx, TxAddr and txs_features.

    Reads no label file and no ``wallets_features.csv``.
    """
    raw = Path(data_root) / "raw"
    tx = pd.read_csv(raw / "txs_features.csv", usecols=TX_COLUMNS)
    tx["Time step"] = tx["Time step"].astype(np.int16)

    spends = pd.read_csv(raw / "AddrTx_edgelist.csv")
    spends = spends.rename(columns={"input_address": "address"})
    spends["role"] = np.int8(ROLE_IN)
    receives = pd.read_csv(raw / "TxAddr_edgelist.csv")
    receives = receives.rename(columns={"output_address": "address"})
    receives["role"] = np.int8(ROLE_OUT)

    both = pd.concat(
        [spends[["address", "txId", "role"]], receives[["address", "txId", "role"]]],
        ignore_index=True,
    )
    codes, addresses = pd.factorize(both["address"], sort=True)
    both["code"] = codes.astype(np.int32)
    both = both.drop(columns=["address"])

    timestep = tx.set_index("txId")["Time step"]
    both["Time step"] = both["txId"].map(timestep).astype(np.int16)
    # A transaction referenced by an edgelist but absent from txs_features has
    # no timestep, so it cannot be placed in time and cannot be used under an
    # as-of-t rule. Dropped rather than imputed.
    both = both.dropna(subset=["Time step"]).reset_index(drop=True)

    return Incidence(
        frame=both, transactions=tx.set_index("txId"),
        addresses=np.asarray(addresses, dtype=object),
    )
