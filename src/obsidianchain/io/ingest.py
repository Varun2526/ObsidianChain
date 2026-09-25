"""Bulk metadata ingestion: CSV, JSON and XML.

Why this module exists
----------------------
The problem statement asks for a system that "ingests bulk Bitcoin
transaction/network metadata (in CSV/JSON/XML)". Until now the only loader in
this project was :mod:`obsidianchain.io.elliptic`, which reads one fixed
Elliptic++ schema from four known filenames. That is a dataset reader, not an
ingestion path: it cannot accept a file it has not seen before, cannot report
what was wrong with one, and has no JSON or XML support at all.

What this does and does not do
------------------------------
It parses, normalises and VALIDATES. It does not score, cluster or model.
Feeding a new capture through the analytical pipeline is an offline pipeline
run (minutes of work over a whole dataset), not something a parser can
pretend to do - so this module reports exactly what it received and what it
could correlate, and stops there.

Honesty about the field list
----------------------------
The PS names a minimum field set. Two of those fields, ``dst_ip`` and
``dst_port``, describe the RECEIVING side of an announcement. This project's
observation model is passive: an observer sees a peer announce a transaction.
The announcing peer is ``src``; the observer is the destination and is
identified by ``observer_id`` rather than an address. So a record may carry
``dst_ip``, and it is preserved when present, but its absence is reported as
absent rather than filled in.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: The PS minimum field set, plus the two optional fields it also names.
#: Order is the contract: a normalised frame always has these columns in this
#: order, whatever the input format was.
CANONICAL_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
    "input_addresses", "output_addresses", "input_amounts", "output_amounts",
    "fee", "script_type", "geo_country", "asn",
]

#: Optional fields beyond the PS schema that the network layer needs. The PS
#: names network fields per observation but no observer identity; a capture
#: collected from several vantage points needs one to tell "two peers
#: announced it to one sensor" from "one peer announced it to two sensors".
#: Kept after the canonical columns, so the canonical order is unchanged.
#: Accepted aliases map to the one name used downstream.
EXTENSION_COLUMNS = ["observer_id"]
EXTENSION_ALIASES = {"observer": "observer_id", "sensor_id": "observer_id"}
OUTPUT_COLUMNS = CANONICAL_COLUMNS + EXTENSION_COLUMNS

#: Without these a record cannot be correlated at all: there is nothing to
#: join to the blockchain layer, or nothing to place in time.
REQUIRED_COLUMNS = ["txid"]

#: Fields the PS names that this project's observation model does not supply.
#: Recorded so a validation report can say "absent by design" rather than
#: leaving a reader to wonder whether parsing dropped them.
LIST_COLUMNS = [
    "input_addresses", "output_addresses", "input_amounts", "output_amounts",
]

NUMERIC_COLUMNS = ["src_port", "dst_port", "fee", "asn"]

#: Separators accepted inside a CSV cell holding a list. Semicolon first
#: because a comma cannot appear unquoted in a CSV field.
LIST_SEPARATORS = (";", "|", " ")

SUPPORTED_SUFFIXES = {".csv": "csv", ".json": "json", ".xml": "xml"}


class IngestError(ValueError):
    """The input could not be parsed at all."""


@dataclass
class ValidationReport:
    """What arrived, what was usable, and what was wrong with the rest."""

    source_format: str
    rows_read: int = 0
    rows_valid: int = 0
    columns_present: list[str] = field(default_factory=list)
    columns_missing: list[str] = field(default_factory=list)
    required_missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exact_duplicates_rejected: int = 0
    network_observations_preserved: int = 0

    @property
    def ok(self) -> bool:
        return not self.required_missing and self.rows_valid > 0

    def as_dict(self) -> dict:
        return {
            "source_format": self.source_format,
            "rows_read": self.rows_read,
            "rows_valid": self.rows_valid,
            "rows_rejected": self.rows_read - self.rows_valid,
            "columns_present": self.columns_present,
            "columns_missing": self.columns_missing,
            "required_missing": self.required_missing,
            "errors": self.errors[:50],
            "warnings": self.warnings[:50],
            "ok": self.ok,
            "exact_duplicates_rejected": self.exact_duplicates_rejected,
            "network_observations_preserved": self.network_observations_preserved,
        }


def detect_format(path, declared: str | None = None) -> str:
    """Format from an explicit declaration, else the suffix.

    Sniffing the content would be guessing; a caller that knows the format
    should say so, and a suffix is the next most reliable signal.
    """
    if declared:
        declared = declared.lower().lstrip(".")
        if declared not in {"csv", "json", "xml"}:
            raise IngestError(
                f"{declared!r} is not a supported format; expected csv, json or xml"
            )
        return declared
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(
            f"cannot determine the format of {Path(path).name!r}. Supported "
            f"formats are CSV, JSON and XML; pass the format explicitly if "
            f"the file has no recognised suffix."
        )
    return SUPPORTED_SUFFIXES[suffix]


def _split_list(value) -> list:
    """A CSV cell into a list, without splitting a single bare value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass  # a bracketed string that is not JSON is treated as text
    for separator in LIST_SEPARATORS:
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _read_csv(path) -> tuple[pd.DataFrame, str]:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as exc:  # pandas raises many types for a bad CSV
        raise IngestError(f"could not parse {Path(path).name} as CSV: {exc}") from exc
    return frame, "csv"


def _read_json(path) -> tuple[pd.DataFrame, str]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise IngestError(
            f"could not parse {Path(path).name} as JSON: {exc}"
        ) from exc
    if isinstance(payload, dict):
        # A wrapper object is common; take the first list-valued key rather
        # than demanding one particular envelope shape.
        for key in ("records", "data", "transactions", "observations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise IngestError(
            f"{Path(path).name} holds a JSON {type(payload).__name__}; a list "
            f"of records, or an object wrapping one, is expected"
        )
    return pd.DataFrame(payload), "json"


def _read_xml(path) -> tuple[pd.DataFrame, str]:
    """Parse a record-per-element XML document.

    Repeated child elements become a list, so ``<input_addresses><a>x</a>
    <a>y</a></input_addresses>`` and two sibling ``<input_address>`` elements
    both arrive as a two-element list. Attributes are read as fields too,
    because both spellings occur in real exports.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise IngestError(
            f"could not parse {Path(path).name} as XML: {exc}"
        ) from exc
    root = tree.getroot()
    records = []
    for element in list(root):
        record: dict = dict(element.attrib)
        for child in element:
            grandchildren = list(child)
            if grandchildren:
                value: object = [
                    (g.text or "").strip() for g in grandchildren
                ]
            else:
                value = (child.text or "").strip()
            if child.tag in record:
                existing = record[child.tag]
                if not isinstance(existing, list):
                    existing = [existing]
                existing.append(value)
                record[child.tag] = existing
            else:
                record[child.tag] = value
        records.append(record)
    if not records:
        raise IngestError(
            f"{Path(path).name} contains no record elements under "
            f"<{root.tag}>"
        )
    return pd.DataFrame(records), "xml"


_READERS = {"csv": _read_csv, "json": _read_json, "xml": _read_xml}


def ingest(path, declared_format: str | None = None
           ) -> tuple[pd.DataFrame, ValidationReport]:
    """Parse and normalise one bulk metadata file.

    Returns the normalised frame (canonical columns, in canonical order) and
    a report describing everything that was missing, coerced or rejected.
    The frame contains only rows that validated; the report says how many
    did not and why, rather than silently shrinking.
    """
    source_format = detect_format(path, declared_format)
    raw, source_format = _READERS[source_format](path)

    report = ValidationReport(source_format=source_format, rows_read=int(len(raw)))
    raw.columns = [str(c).strip() for c in raw.columns]
    for alias, name in EXTENSION_ALIASES.items():
        if alias in raw.columns and name not in raw.columns:
            raw = raw.rename(columns={alias: name})
    present = [c for c in CANONICAL_COLUMNS if c in raw.columns]
    report.columns_present = present
    report.columns_missing = [c for c in CANONICAL_COLUMNS if c not in raw.columns]
    report.required_missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]

    if report.required_missing:
        report.errors.append(
            f"required column(s) {report.required_missing} are absent; without "
            f"them a record cannot be joined to the blockchain layer"
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS), report

    for name in LIST_COLUMNS:
        if name in report.columns_missing:
            report.warnings.append(
                f"{name} is absent; wallet-level correlation is unavailable "
                f"for this file"
            )
    for name in ("src_ip", "asn", "timestamp"):
        if name in report.columns_missing:
            report.warnings.append(
                f"{name} is absent; network-layer correlation is degraded"
            )

    frame = pd.DataFrame(index=raw.index)
    for column in OUTPUT_COLUMNS:
        if column not in raw.columns:
            frame[column] = None
            continue
        values = raw[column]
        if column in LIST_COLUMNS:
            frame[column] = values.map(_split_list)
        elif column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(values, errors="coerce")
            bad = int(frame[column].isna().sum() - values.isna().sum())
            if bad > 0:
                report.warnings.append(
                    f"{column}: {bad} value(s) were not numeric and became null"
                )
        else:
            frame[column] = values.astype("string")

    valid = frame["txid"].notna() & (frame["txid"].astype("string").str.len() > 0)
    rejected = int((~valid).sum())
    if rejected:
        report.errors.append(f"{rejected} row(s) have no txid and were rejected")
    frame = frame[valid].reset_index(drop=True)

    # ---- duplicate handling ------------------------------------------
    # Blockchain identity and network observation identity are DIFFERENT.
    # An exact duplicate blockchain record is rejected (same txid, same
    # inputs, outputs, amounts, fee, script_type).  But multiple network
    # observations of the same TXID from different sources (different
    # src_ip, src_port, or timestamp) are preserved — they represent
    # distinct P2P relay vantage points and are analytically meaningful.
    if len(frame) > 1:
        frame, report = _deduplicate(frame, report)

    report.rows_valid = int(len(frame))
    return frame[OUTPUT_COLUMNS], report


def _canonical_tuples(addresses, amounts) -> list[tuple[str, str]]:
    """Pair addresses with corresponding amounts before sorting.

    Prevents destroying address <-> amount correspondence by never sorting
    addresses and amounts independently.
    """
    if addresses is None or (isinstance(addresses, float) and pd.isna(addresses)):
        addrs = []
    elif isinstance(addresses, (list, tuple)):
        addrs = [str(a) if a is not None and not (isinstance(a, float) and pd.isna(a)) else "" for a in addresses]
    else:
        addrs = [str(addresses)]

    if amounts is None or (isinstance(amounts, float) and pd.isna(amounts)):
        amts = []
    elif isinstance(amounts, (list, tuple)):
        amts = [str(a) if a is not None and not (isinstance(a, float) and pd.isna(a)) else "" for a in amounts]
    else:
        amts = [str(amounts)]

    n = max(len(addrs), len(amts))
    pairs: list[tuple[str, str]] = []
    for i in range(n):
        addr = addrs[i] if i < len(addrs) else ""
        amt = amts[i] if i < len(amts) else ""
        pairs.append((addr, amt))
    return sorted(pairs)


def blockchain_record_key(record: dict | pd.Series) -> str:
    """Canonical identity key for a blockchain transaction record.

    Uses canonical input/output tuples sorted as (address, amount) pairs
    to preserve the address <-> amount relationship.
    Includes fields that define a blockchain record under the canonical schema:
    txid, canonical input tuples, canonical output tuples, fee, and script_type.
    """
    txid = str(record.get("txid") or "").strip()
    inputs = _canonical_tuples(record.get("input_addresses"), record.get("input_amounts"))
    outputs = _canonical_tuples(record.get("output_addresses"), record.get("output_amounts"))
    fee = str(record.get("fee") if record.get("fee") is not None and not pd.isna(record.get("fee")) else "").strip()
    script_type = str(record.get("script_type") if record.get("script_type") is not None and not pd.isna(record.get("script_type")) else "").strip()

    payload = json.dumps({
        "txid": txid,
        "inputs": inputs,
        "outputs": outputs,
        "fee": fee,
        "script_type": script_type,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def observation_key(record: dict | pd.Series) -> str:
    """Canonical identity key for a network observation.

    Key composition:
        hash(txid + observer_id + src_ip + src_port + timestamp)

    FORENSIC SCHEMA LIMITATION:
    The canonical Problem Statement (PS) schema specifies network fields:
    (timestamp, src_ip, dst_ip, src_port, dst_port, geo_country, asn)
    and does NOT include an observer_id column. When observer_id is not
    present in the input record, observer_id defaults to empty string in
    the observation key. If an observer_id (or observer) is provided in
    extended capture data, it is incorporated into the key to distinguish
    distinct observations from different vantage points.
    """
    txid = str(record.get("txid") or "").strip()
    observer_val = record.get("observer_id") if "observer_id" in record else record.get("observer")
    observer_id = str(observer_val or "").strip() if observer_val is not None and not pd.isna(observer_val) else ""
    src_ip = str(record.get("src_ip") if record.get("src_ip") is not None and not pd.isna(record.get("src_ip")) else "").strip()
    src_port = str(record.get("src_port") if record.get("src_port") is not None and not pd.isna(record.get("src_port")) else "").strip()
    timestamp = str(record.get("timestamp") if record.get("timestamp") is not None and not pd.isna(record.get("timestamp")) else "").strip()

    payload = json.dumps({
        "txid": txid,
        "observer_id": observer_id,
        "src_ip": src_ip,
        "src_port": src_port,
        "timestamp": timestamp,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deduplicate(frame: pd.DataFrame, report: ValidationReport) -> tuple[pd.DataFrame, ValidationReport]:
    """Deduplicate records distinguishing blockchain identity from network identity.

    - Exact duplicate blockchain records with identical (or empty) network observations are rejected.
    - Multiple network observations of the same TXID from distinct vantage points (different
      observer_id, src_ip, src_port, or timestamp) are preserved.
    """
    keep_indices: list[int] = []
    seen_row_signatures: set[str] = set()
    seen_blockchain_only: set[str] = set()
    seen_tx_network_keys: dict[str, set[str]] = {}

    for idx, row in frame.iterrows():
        b_key = blockchain_record_key(row)
        o_key = observation_key(row)
        txid = str(row.get("txid") or "").strip()

        has_net = bool(
            pd.notna(row.get("src_ip"))
            and str(row.get("src_ip")).strip() not in ("", "<NA>", "nan", "None")
        )

        row_sig = f"{b_key}:{o_key}"

        if row_sig in seen_row_signatures:
            report.exact_duplicates_rejected += 1
            continue

        if not has_net and b_key in seen_blockchain_only:
            report.exact_duplicates_rejected += 1
            continue

        if has_net:
            if txid in seen_tx_network_keys and o_key not in seen_tx_network_keys[txid]:
                report.network_observations_preserved += 1
            seen_tx_network_keys.setdefault(txid, set()).add(o_key)

        seen_row_signatures.add(row_sig)
        if not has_net:
            seen_blockchain_only.add(b_key)
        keep_indices.append(idx)

    if report.exact_duplicates_rejected > 0:
        report.warnings.append(
            f"{report.exact_duplicates_rejected} exact duplicate record(s) were rejected"
        )

    deduped_frame = frame.loc[keep_indices].reset_index(drop=True)
    return deduped_frame, report


def correlation_summary(frame: pd.DataFrame) -> dict:
    """What the ingested file can actually correlate.

    This is the PS's "correlate network-layer observations with
    blockchain-layer data" question answered for one file, and answered
    honestly: a count of records that carry BOTH a network identifier and a
    wallet identifier, not a claim that the correlation is meaningful.
    """
    if frame.empty:
        return {
            "records": 0, "transactions": 0, "addresses": 0,
            "source_ips": 0, "asns": 0, "correlatable_records": 0,
        }

    def _flatten(column: str) -> set:
        if column not in frame.columns:
            return set()
        out: set = set()
        for cell in frame[column]:
            if isinstance(cell, list):
                out.update(str(v) for v in cell if str(v))
        return out

    addresses = _flatten("input_addresses") | _flatten("output_addresses")
    has_network = frame["src_ip"].notna() if "src_ip" in frame else pd.Series(False, index=frame.index)
    has_wallet = frame["input_addresses"].map(bool) | frame["output_addresses"].map(bool)
    return {
        "records": int(len(frame)),
        "transactions": int(frame["txid"].nunique()),
        "addresses": len(addresses),
        "source_ips": int(frame["src_ip"].nunique()) if "src_ip" in frame else 0,
        "asns": int(frame["asn"].nunique()) if "asn" in frame else 0,
        "correlatable_records": int((has_network & has_wallet).sum()),
        "time_span": _time_span(frame),
    }


def _time_span(frame: pd.DataFrame) -> dict | None:
    if "timestamp" not in frame.columns or frame["timestamp"].isna().all():
        return None
    values = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
    if values.empty:
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dropna()
        if parsed.empty:
            return None
        return {"first": str(parsed.min()), "last": str(parsed.max()), "unit": "parsed datetime"}
    return {"first": float(values.min()), "last": float(values.max()),
            "unit": "numeric as supplied"}
