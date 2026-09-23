# Data contract

Two versioned contracts, both enforced in code; this page describes them.

## Capture contract — `obsidianchain.capture_contract/1` (`contracts/capture.py`)

Input: the canonical capture (CSV / JSON / XML; `io/ingest.py`), one row per
network observation of a transaction.

| Field | Type | Null means | Rule |
|---|---|---|---|
| `txid` | string, 1-128 chars, no whitespace | row rejected (ingest) | required |
| `timestamp` | Unix seconds | unknown time | 2009-01-03 to now + 1 day, else the ROW is dropped |
| `input_addresses`, `output_addresses` | list of strings, each 1-128 chars | no chain data (network-only row) | invalid -> transaction quarantined |
| `input_amounts`, `output_amounts` | list of BTC amounts, one per address | amounts not supplied | negative / non-finite / length mismatch -> quarantined |
| `fee` | BTC | fee not supplied (NaN, never 0) | negative, or above total inputs -> quarantined |
| `src_ip`, `dst_ip` | IPv4 / IPv6 | not observed | invalid -> row dropped |
| `src_port`, `dst_port` | 0-65535 | not observed | out of range -> row dropped |
| `asn` | 0 .. 2^32-1 | not observed | out of range -> row dropped |
| `geo_country`, `script_type` | string | not supplied | free text |

Per transaction, all chain rows of one `txid` must agree on inputs, outputs,
amounts and fee. If they don't: `CONFLICTING_CHAIN_FACTS`, and the transaction
is quarantined. Multiple network observations of one txid are expected and
are aggregated, never replayed. More than 50% of transactions quarantined
refuses the capture (`CaptureContractError`). **Nothing is coerced into a
plausible value.**

Ordering: events are processed by `(timestamp, event_order)`. `event_order`
is optional; the Elliptic++ builder sets it to the spend-DAG level. Events at
the same point are simultaneous and never see each other.

## Feature contract — `obsidianchain.feature_contract/1` (`contracts/features.py`)

Every model feature has a catalog entry (`docs/feature_catalog.md`) giving
its source, information set, same-step behaviour, NaN meaning, range and
integer-ness. Before scoring, the frame is checked:
- missing column, non-numeric, infinite value, unexpected NaN, out of range,
  or non-integer count: status `FEATURE_CONTRACT_VIOLATION` and no score;
- a feature without a catalog entry cannot be served.

## Compatibility

- Feature schema `ps_native_features/N` changes whenever any value can
  change. A model is served only on its own schema; there is no silent
  cross-schema load.
- Capture contract changes bump the contract version; the version is
  written into every run's stage-2 summary.
