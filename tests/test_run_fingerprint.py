"""The fingerprint over every input that determines an evidence row.

The property under test is the one the design demands:

    SAME IMMUTABLE INPUTS + SAME EDGE ID  ->  SAME EVIDENCE ID
    DIFFERENT INPUTS                      ->  cannot resolve to the old row

Each of the five determining inputs gets its own mutation test. Three of them
are file hashes; two are literals that no file hash would ever catch, which is
precisely why they are in the digest.

Temporary fixture inputs throughout. The real frozen files are never touched.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from obsidianchain import run_fingerprint as rf

A = "a" * 64
B = "b" * 64
C = "c" * 64


def inputs(**overrides) -> dict:
    base = dict(
        chain_addr_tx_sha256=A,
        chain_universe_sha256=B,
        network_dataset_sha256=C,
        heuristics=rf.HEURISTICS_MULTI_INPUT,
        row_statistics_config=rf.PROBE_ROW_CONFIG,
    )
    base.update(overrides)
    return base


# ---- 1. deterministic ---------------------------------------------------


def test_same_inputs_give_the_same_fingerprint() -> None:
    assert rf.build_evidence_run_fingerprint(
        **inputs()
    ) == rf.build_evidence_run_fingerprint(**inputs())


def test_the_fingerprint_is_a_pure_function_of_its_arguments() -> None:
    """No clock, no environment, no filesystem in the digest."""
    first = rf.build_evidence_run_fingerprint(**inputs())
    import os

    os.environ["OBSIDIANCHAIN_PROBE"] = "changed"
    try:
        assert rf.build_evidence_run_fingerprint(**inputs()) == first
    finally:
        del os.environ["OBSIDIANCHAIN_PROBE"]


def test_the_public_form_is_sixteen_hex_characters() -> None:
    public = rf.fingerprint_from_inputs(**inputs())
    assert len(public) == rf.FINGERPRINT_LENGTH == 16
    assert all(c in "0123456789abcdef" for c in public)
    assert rf.build_evidence_run_fingerprint(**inputs()).startswith(public)


def test_case_in_a_digest_does_not_change_the_fingerprint() -> None:
    """Hex is normalised, so an uppercase digest is not a different run."""
    assert rf.build_evidence_run_fingerprint(
        **inputs(chain_addr_tx_sha256=A.upper())
    ) == rf.build_evidence_run_fingerprint(**inputs())


# ---- 2-6. every determining input changes the fingerprint ---------------


@pytest.mark.parametrize("field,value", [
    ("chain_addr_tx_sha256", "d" * 64),
    ("chain_universe_sha256", "d" * 64),
    ("network_dataset_sha256", "d" * 64),
    ("heuristics", "multi-input+change"),
    ("row_statistics_config", "probe:min_pooled=2,min_observer=2"),
])
def test_changing_any_determining_input_changes_the_fingerprint(
    field, value
) -> None:
    """Five inputs, five mutations, five distinct fingerprints.

    The two literals matter as much as the hashes: switching the clustering
    heuristic or the row-statistics config alters every row while leaving all
    three file hashes byte-identical.
    """
    assert rf.build_evidence_run_fingerprint(
        **inputs(**{field: value})
    ) != rf.build_evidence_run_fingerprint(**inputs())


def test_all_five_mutations_are_mutually_distinct() -> None:
    """No two different inputs may collide onto one fingerprint."""
    variants = [
        inputs(),
        inputs(chain_addr_tx_sha256="d" * 64),
        inputs(chain_universe_sha256="d" * 64),
        inputs(network_dataset_sha256="d" * 64),
        inputs(heuristics="multi-input+change"),
        inputs(row_statistics_config="probe:min_pooled=2,min_observer=2"),
    ]
    digests = [rf.build_evidence_run_fingerprint(**v) for v in variants]
    assert len(set(digests)) == len(variants)
    assert len({d[: rf.FINGERPRINT_LENGTH] for d in digests}) == len(variants)


def test_swapping_two_chain_hashes_changes_the_fingerprint() -> None:
    """Field order is part of the contract, not an implementation detail.

    If the payload were an unordered set of values, swapping which file each
    hash came from would collide - and the two chain files determine
    different things.
    """
    assert rf.build_evidence_run_fingerprint(
        **inputs(chain_addr_tx_sha256=B, chain_universe_sha256=A)
    ) != rf.build_evidence_run_fingerprint(**inputs())


# ---- the canonical payload ----------------------------------------------


def test_the_canonical_payload_is_inspectable_and_ordered() -> None:
    """A human debugging an ID mismatch must be able to see what went in."""
    payload = rf.canonical_payload(**inputs())
    lines = payload.split("\n")
    assert lines[0] == rf.DOMAIN
    assert lines[1:] == [
        f"chain.addr_tx={A}",
        f"chain.universe={B}",
        f"network={C}",
        f"heuristics={rf.HEURISTICS_MULTI_INPUT}",
        f"row_config={rf.PROBE_ROW_CONFIG}",
    ]


def test_the_digest_is_sha256_of_the_canonical_payload() -> None:
    """No hidden salt: the digest is exactly what the payload says it is."""
    payload = rf.canonical_payload(**inputs())
    assert rf.build_evidence_run_fingerprint(**inputs()) == hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def test_the_domain_separator_is_present() -> None:
    """Keeps this digest from being confused with a plain file hash."""
    assert rf.DOMAIN == "obsidianchain.evidence_run/1"
    assert rf.build_evidence_run_fingerprint(**inputs()) != hashlib.sha256(
        "\n".join(rf.canonical_payload(**inputs()).split("\n")[1:]).encode()
    ).hexdigest()


# ---- missing or malformed inputs are refused ----------------------------


@pytest.mark.parametrize("field", [
    "chain_addr_tx_sha256", "chain_universe_sha256", "network_dataset_sha256",
])
@pytest.mark.parametrize("bad", [None, "", "abc", "z" * 64, A[:63]])
def test_a_missing_or_malformed_hash_is_refused(field, bad) -> None:
    """A blank hash would otherwise produce a perfectly stable fingerprint
    that silently means "one input was unknown" - the exact failure this
    module exists to prevent."""
    with pytest.raises(rf.FingerprintInputError):
        rf.build_evidence_run_fingerprint(**inputs(**{field: bad}))


@pytest.mark.parametrize("field", ["heuristics", "row_statistics_config"])
@pytest.mark.parametrize("bad", [None, "", "   "])
def test_a_missing_label_is_refused(field, bad) -> None:
    with pytest.raises(rf.FingerprintInputError):
        rf.build_evidence_run_fingerprint(**inputs(**{field: bad}))


def test_the_public_form_refuses_a_short_digest() -> None:
    with pytest.raises(rf.FingerprintInputError):
        rf.public_fingerprint("abc")


def test_the_arguments_are_keyword_only() -> None:
    """Five hex-looking strings in a row is where a positional swap hides."""
    with pytest.raises(TypeError):
        rf.build_evidence_run_fingerprint(A, B, C)


# ---- sha256_file ---------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    payload = b"input_address,txId\naddr0,1\naddr1,1\n"
    path.write_bytes(payload)
    assert rf.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_is_streamed_and_handles_a_large_file(tmp_path: Path) -> None:
    """Larger than the 1 MiB read block, so the chunk loop is exercised."""
    path = tmp_path / "big.csv"
    payload = b"x" * (3 * (1 << 20) + 17)
    path.write_bytes(payload)
    assert rf.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_a_one_byte_change_changes_the_file_hash(tmp_path: Path) -> None:
    """The whole scheme rests on this. Fixture files, never the real ones."""
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    first.write_bytes(b"input_address,txId\naddr0,1\n")
    second.write_bytes(b"input_address,txId\naddr0,2\n")
    assert rf.sha256_file(first) != rf.sha256_file(second)


def test_row_order_alone_changes_the_file_hash(tmp_path: Path) -> None:
    """AddrTx row ORDER fixes edge_index, so order must move the hash."""
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    first.write_bytes(b"input_address,txId\naddr0,1\naddr1,2\n")
    second.write_bytes(b"input_address,txId\naddr1,2\naddr0,1\n")
    assert rf.sha256_file(first) != rf.sha256_file(second)


# ---- the row-config literal must track the real PROBE_CONFIG ------------


def test_the_row_config_literal_matches_the_actual_probe_config() -> None:
    """The one term in the digest that could silently drift.

    PROBE_CONFIG lives in code. If somebody edits it, every persisted
    statistic changes while all three file hashes stay identical, so the
    literal is the only thing standing between that and a silent re-point.
    This test is what keeps the literal honest.
    """
    from obsidianchain.eval.evidence_funnel import PROBE_CONFIG

    assert rf.PROBE_ROW_CONFIG == rf.row_config_label(
        PROBE_CONFIG.min_pooled_observations,
        PROBE_CONFIG.min_observer_observations,
    )
    assert PROBE_CONFIG.min_pooled_observations == 1
    assert PROBE_CONFIG.min_observer_observations == 2


def test_the_heuristics_literal_matches_the_cluster_index() -> None:
    """Both artifacts must agree on which clustering they describe."""
    from obsidianchain.cluster import index

    assert rf.HEURISTICS_MULTI_INPUT == index.HEURISTICS


def test_row_config_label_is_pure() -> None:
    assert rf.row_config_label(1, 2) == "probe:min_pooled=1,min_observer=2"
    assert rf.row_config_label(25, 5) == "probe:min_pooled=25,min_observer=5"


# ---- import discipline ---------------------------------------------------


def test_the_module_imports_nothing_the_api_is_forbidden() -> None:
    """The API must be able to import this without breaching its boundary."""
    import ast

    from obsidianchain.api import boundary

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "obsidianchain" / "run_fingerprint.py"
    ).read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    offenders = sorted(
        n for n in names
        for f in boundary.FORBIDDEN_RECOMPUTATION
        if n == f or n.startswith(f + ".")
    )
    assert not offenders, f"run_fingerprint imports {offenders}"
    assert names <= {"__future__", "hashlib", "pathlib"}, (
        f"unexpected imports: {sorted(names)}"
    )
