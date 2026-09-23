"""Capture contract (``obsidianchain.capture_contract/1``).

Applied to the frame ``io.ingest`` returns (canonical columns, one row per
network observation). Ingest already rejects rows without a txid and nulls
non-numeric numbers with a warning. This contract checks what ingest does
not, and QUARANTINES the affected transaction instead of repairing it:

Per transaction (all rows of a txid):
  CONFLICTING_CHAIN_FACTS  rows of one txid disagree on inputs, outputs,
                           amounts or fee. The engine would silently keep
                           the first row; which one is "right" is unknowable.
  AMOUNT_LIST_MISMATCH     amounts given but not one per address
  NEGATIVE_OR_NONFINITE_AMOUNT
  NEGATIVE_FEE
  FEE_EXCEEDS_INPUTS       fee > total input value (beyond rounding)
  ADDRESS_INVALID          empty or over-long address string
  TXID_INVALID             over-long or containing whitespace

Per row (the observation is dropped, the transaction kept):
  TIMESTAMP_OUT_OF_RANGE   before the genesis block or more than a day ahead
  PORT_OUT_OF_RANGE        not in 0..65535
  IP_INVALID               src_ip / dst_ip not a valid IPv4/IPv6 address
  ASN_OUT_OF_RANGE         negative or above 2**32 - 1

If more than ``MAX_QUARANTINE_SHARE`` of transactions are quarantined the
capture is refused as a whole: at that point the file is not what it claims
to be, and analysing the remainder would misrepresent it.
"""

from __future__ import annotations

import ipaddress
import math
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

CAPTURE_CONTRACT_VERSION = "obsidianchain.capture_contract/1"

#: 2009-01-03, the genesis block.
MIN_TIMESTAMP = 1230768000
#: Clock skew tolerated into the future.
FUTURE_TOLERANCE_SECONDS = 86400
MAX_ID_LENGTH = 128
#: Fee may exceed inputs by this much before it counts as a violation.
FEE_TOLERANCE = 1e-8
MAX_QUARANTINE_SHARE = 0.5


@dataclass
class CaptureContractReport:
    version: str = CAPTURE_CONTRACT_VERSION
    transactions: int = 0
    rows: int = 0
    quarantined: dict[str, list[str]] = field(default_factory=dict)
    """txid -> violation kinds."""
    rows_dropped: dict[str, int] = field(default_factory=dict)
    """violation kind -> rows dropped."""
    refused: bool = False

    @property
    def ok(self) -> bool:
        return not self.refused

    def as_dict(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for vs in self.quarantined.values():
            for v in vs:
                kinds[v] = kinds.get(v, 0) + 1
        return {
            "version": self.version, "ok": self.ok, "refused": self.refused,
            "transactions": self.transactions, "rows": self.rows,
            "quarantined_transactions": len(self.quarantined),
            "quarantine_kinds": kinds,
            "quarantined_txids_sample": sorted(self.quarantined)[:20],
            "rows_dropped": self.rows_dropped,
            "max_quarantine_share": MAX_QUARANTINE_SHARE,
        }


def _as_list(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else []


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _chain_key(row) -> tuple:
    return (
        tuple(str(a) for a in _as_list(row.get("input_addresses"))),
        tuple(str(a) for a in _as_list(row.get("output_addresses"))),
        tuple(str(a) for a in _as_list(row.get("input_amounts"))),
        tuple(str(a) for a in _as_list(row.get("output_amounts"))),
        str(row.get("fee")),
    )


def _is_chain_row(row) -> bool:
    return bool(_as_list(row.get("input_addresses")) or _as_list(row.get("output_addresses")))


def _tx_violations(row) -> list[str]:
    out: list[str] = []
    txid = str(row.get("txid") or "")
    if len(txid) > MAX_ID_LENGTH or any(c.isspace() for c in txid):
        out.append("TXID_INVALID")
    total_in = 0.0
    for addr_col, amt_col in (("input_addresses", "input_amounts"), ("output_addresses", "output_amounts")):
        addrs, amts = _as_list(row.get(addr_col)), _as_list(row.get(amt_col))
        if any((not str(a).strip()) or len(str(a)) > MAX_ID_LENGTH for a in addrs):
            out.append("ADDRESS_INVALID")
        if amts and len(amts) != len(addrs):
            out.append("AMOUNT_LIST_MISMATCH")
        for a in amts:
            f = _num(a)
            if f is None or not math.isfinite(f) or f < 0:
                out.append("NEGATIVE_OR_NONFINITE_AMOUNT")
                break
            if addr_col == "input_addresses":
                total_in += f
    fee = _num(row.get("fee"))
    amounts_valid = "NEGATIVE_OR_NONFINITE_AMOUNT" not in out
    if fee is not None and math.isfinite(fee):
        if fee < 0:
            out.append("NEGATIVE_FEE")
        # Only comparable when every input amount was valid.
        elif amounts_valid and _as_list(row.get("input_amounts")) and fee > total_in + FEE_TOLERANCE:
            out.append("FEE_EXCEEDS_INPUTS")
    return sorted(set(out))


def _row_violation(row, now: float) -> str | None:
    ts = _num(row.get("timestamp"))
    if ts is not None and math.isfinite(ts) and not (MIN_TIMESTAMP <= ts <= now + FUTURE_TOLERANCE_SECONDS):
        return "TIMESTAMP_OUT_OF_RANGE"
    for col in ("src_port", "dst_port"):
        p = _num(row.get(col))
        if p is not None and math.isfinite(p) and not (0 <= p <= 65535):
            return "PORT_OUT_OF_RANGE"
    for col in ("src_ip", "dst_ip"):
        v = row.get(col)
        if v is not None and not pd.isna(v) and str(v).strip():
            try:
                ipaddress.ip_address(str(v).strip())
            except ValueError:
                return "IP_INVALID"
    asn = _num(row.get("asn"))
    if asn is not None and math.isfinite(asn) and not (0 <= asn <= 2**32 - 1):
        return "ASN_OUT_OF_RANGE"
    return None


def enforce_capture_contract(frame: pd.DataFrame, now: float | None = None
                             ) -> tuple[pd.DataFrame, CaptureContractReport]:
    """Return the frame without quarantined transactions or bad rows, and a report."""
    now = time.time() if now is None else now
    report = CaptureContractReport(rows=int(len(frame)))
    if frame.empty:
        return frame, report
    report.transactions = int(frame["txid"].nunique())

    keep_rows = []
    for idx, row in frame.iterrows():
        kind = _row_violation(row, now)
        if kind is None:
            keep_rows.append(idx)
        else:
            report.rows_dropped[kind] = report.rows_dropped.get(kind, 0) + 1
    frame = frame.loc[keep_rows]

    for txid, group in frame.groupby("txid", sort=False):
        chain_rows = [r for _, r in group.iterrows() if _is_chain_row(r)]
        kinds: set[str] = set()
        if len({_chain_key(r) for r in chain_rows}) > 1:
            kinds.add("CONFLICTING_CHAIN_FACTS")
        if chain_rows:
            kinds.update(_tx_violations(chain_rows[0]))
        if kinds:
            report.quarantined[str(txid)] = sorted(kinds)

    if report.transactions and len(report.quarantined) / report.transactions > MAX_QUARANTINE_SHARE:
        report.refused = True
        return frame.iloc[0:0], report
    clean = frame[~frame["txid"].astype(str).isin(report.quarantined)].reset_index(drop=True)
    return clean, report
