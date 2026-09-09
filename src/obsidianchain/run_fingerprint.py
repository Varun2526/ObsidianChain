"""One fingerprint over every input that determines an evidence row.

Why this exists
---------------
An evidence row in ``evidence_funnel.parquet`` is addressed by ``edge_index``.
That index is **not** determined by the network dataset, which is what the
sidecar's ``dataset_sha256`` records. It is determined by two raw chain files:

* ``AddrTx_edgelist.csv`` - ``factorize(txId)`` runs in first-appearance file
  order and ``_star_edges_impl`` stable-sorts by the resulting codes, so the
  file's row order fixes the edge order;
* ``wallets_classes.csv`` - ``load_cospend_graph`` performs one ``factorize``
  over ``concat([universe, input_addresses])`` with the universe **first**, so
  this file's length and content shift every address code, and therefore every
  ``node_a``/``node_b``/``cluster_id`` value.

So an ID of the form ``network_sha:edge_index`` can silently re-point: edit a
chain file, leave the network dataset alone, and the same ID resolves to a
different row with no error anywhere. Two further vectors are invisible to any
file hash at all - switching the clustering heuristic, and changing the
hardcoded ``PROBE_CONFIG`` under which the row statistics were computed.

This module folds all five into one digest, so a stale ID is a detectable
mismatch rather than a wrong answer.

The property it guarantees
--------------------------
::

    SAME IMMUTABLE INPUTS + SAME EDGE ID  ->  SAME EVIDENCE ID
    DIFFERENT INPUTS                      ->  cannot resolve to the old row

Deployment constraint
---------------------
:func:`sha256_file` runs on the **artifact-generation** side only. An
API-only deployment has no ``data/raw/``, so the API reads the persisted
hashes out of the sidecar and calls
:func:`build_evidence_run_fingerprint` with them. It never hashes a file.
``tests/test_api_evidence.py`` asserts the API source never calls
:func:`sha256_file`.

Import discipline
-----------------
Standard library only. Deliberately imports nothing from
``obsidianchain.io.elliptic``, ``obsidianchain.cluster``,
``obsidianchain.network.separation`` or ``obsidianchain.eval``, so that the
API may import it without breaching the no-recomputation boundary declared in
:mod:`obsidianchain.api.boundary`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Domain separator. Keeps this digest from ever being confused with the
#: dataset hashes, which are plain SHA-256 of a file's bytes.
DOMAIN = "obsidianchain.evidence_run/1"

#: The clustering the production funnel is built on. ``multi-input+change``
#: yields a different clustering (562,894 clusters against 569,513) and so a
#: different edge set; naming it forecloses conflating the two.
HEURISTICS_MULTI_INPUT = "multi-input"

#: Label for the configuration the funnel's row statistics were computed
#: under. This is NOT the production rule - see
#: :data:`obsidianchain.eval.evidence_funnel.PROBE_CONFIG`. It is a literal
#: rather than a read value because the config lives in code, and a change to
#: it would alter every statistic while leaving all three file hashes
#: identical. ``tests/test_run_fingerprint.py`` asserts the literal still
#: matches PROBE_CONFIG.
PROBE_ROW_CONFIG = "probe:min_pooled=1,min_observer=2"

#: Hex characters of the digest used in a public ID. 64 bits: collision-free
#: across the handful of dataset generations this project will see, and short
#: enough that an ID stays readable in a URL and a bug report.
FINGERPRINT_LENGTH = 16

#: Order is part of the contract. A dict would serialise by insertion order
#: and a future refactor could reorder it silently, changing every ID.
_FIELD_ORDER = (
    "chain.addr_tx",
    "chain.universe",
    "network",
    "heuristics",
    "row_config",
)

_CHUNK = 1 << 20


class FingerprintInputError(ValueError):
    """Raised when a fingerprint input is missing or not a hex digest."""


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes, streamed.

    Streamed because the chain inputs are 21 MB and 29 MB and there is no
    reason to hold either in memory. Measured cost: about 0.07 s each.

    ARTIFACT-GENERATION SIDE ONLY. The API must not call this: raw data is
    not present on an API-only deployment, and hashing an input per request
    would make the response depend on a file the endpoint does not own.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def row_config_label(min_pooled: int, min_observer: int) -> str:
    """Canonical label for a row-statistics configuration.

    Takes plain integers rather than a ``SeparationConfig`` so this module
    stays free of any import the API is forbidden.
    """
    return f"probe:min_pooled={int(min_pooled)},min_observer={int(min_observer)}"


def _require_digest(name: str, value) -> str:
    """Reject anything that is not a lowercase hex digest.

    A blank or None hash would otherwise hash to a perfectly stable
    fingerprint that silently means "one of the inputs was unknown", which is
    the failure this module exists to prevent.
    """
    if not isinstance(value, str) or not value:
        raise FingerprintInputError(
            f"{name} is required and must be a hex digest; got {value!r}"
        )
    lowered = value.lower()
    if len(lowered) != 64 or any(c not in "0123456789abcdef" for c in lowered):
        raise FingerprintInputError(
            f"{name} must be a 64-character SHA-256 hex digest; got "
            f"{value!r} ({len(value)} chars)"
        )
    return lowered


def _require_label(name: str, value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FingerprintInputError(f"{name} is required; got {value!r}")
    return value.strip()


def canonical_payload(
    *,
    chain_addr_tx_sha256: str,
    chain_universe_sha256: str,
    network_dataset_sha256: str,
    heuristics: str = HEURISTICS_MULTI_INPUT,
    row_statistics_config: str = PROBE_ROW_CONFIG,
) -> str:
    """The exact bytes that get hashed, as text.

    Exposed so a test - and a human debugging an ID mismatch - can see
    precisely what went in, rather than inferring it from a digest.
    """
    values = {
        "chain.addr_tx": _require_digest("chain_addr_tx_sha256", chain_addr_tx_sha256),
        "chain.universe": _require_digest(
            "chain_universe_sha256", chain_universe_sha256
        ),
        "network": _require_digest(
            "network_dataset_sha256", network_dataset_sha256
        ),
        "heuristics": _require_label("heuristics", heuristics),
        "row_config": _require_label(
            "row_statistics_config", row_statistics_config
        ),
    }
    lines = [DOMAIN]
    lines.extend(f"{key}={values[key]}" for key in _FIELD_ORDER)
    return "\n".join(lines)


def build_evidence_run_fingerprint(
    *,
    chain_addr_tx_sha256: str,
    chain_universe_sha256: str,
    network_dataset_sha256: str,
    heuristics: str = HEURISTICS_MULTI_INPUT,
    row_statistics_config: str = PROBE_ROW_CONFIG,
) -> str:
    """Full 64-character fingerprint over every determining input.

    Keyword-only on purpose: five hex-looking strings in a row is exactly the
    signature where a positional swap produces a valid-looking wrong answer.
    """
    payload = canonical_payload(
        chain_addr_tx_sha256=chain_addr_tx_sha256,
        chain_universe_sha256=chain_universe_sha256,
        network_dataset_sha256=network_dataset_sha256,
        heuristics=heuristics,
        row_statistics_config=row_statistics_config,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_fingerprint(full: str) -> str:
    """The truncated form used in a public evidence ID."""
    if not isinstance(full, str) or len(full) < FINGERPRINT_LENGTH:
        raise FingerprintInputError(
            f"fingerprint must be at least {FINGERPRINT_LENGTH} characters; "
            f"got {full!r}"
        )
    return full[:FINGERPRINT_LENGTH].lower()


def fingerprint_from_inputs(**kwargs) -> str:
    """Convenience: build and truncate in one call."""
    return public_fingerprint(build_evidence_run_fingerprint(**kwargs))
