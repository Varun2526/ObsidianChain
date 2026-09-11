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


# ---- Phase 5.3 section 6. the production half of the row config --------


def live_row_config() -> str:
    """The label the artifact writer actually persists, built from the two
    live configuration objects rather than retyped here."""
    from obsidianchain import evidence_contract as contract
    from obsidianchain.eval.evidence_funnel import PROBE_CONFIG
    from obsidianchain.network.separation import SeparationConfig

    production = SeparationConfig()  # production defaults, untouched
    return contract.combined_row_config_label(
        probe_min_pooled=PROBE_CONFIG.min_pooled_observations,
        probe_min_observer=PROBE_CONFIG.min_observer_observations,
        production_min_pooled=production.min_pooled_observations,
        production_min_observer=production.min_observer_observations,
        production_alpha=production.alpha,
        production_min_effect=production.min_effect,
    )


def test_the_combined_label_matches_both_live_configurations() -> None:
    """Schema /2 rows are determined by two configurations, so the digest
    term has to name both. Checked against the live objects: if either is
    edited, every persisted number changes while all three file hashes stay
    identical, and this is the only thing standing between that and a silent
    re-point."""
    from obsidianchain.eval.evidence_funnel import PROBE_CONFIG
    from obsidianchain.network.separation import SeparationConfig

    production = SeparationConfig()
    label = live_row_config()
    assert label.startswith(rf.PROBE_ROW_CONFIG + "|")
    assert PROBE_CONFIG.min_pooled_observations == 1
    assert PROBE_CONFIG.min_observer_observations == 2
    assert production.min_pooled_observations == 25
    assert production.min_observer_observations == 5
    assert production.alpha == 1e-4
    assert production.min_effect == 0.05
    assert label == (
        "probe:min_pooled=1,min_observer=2"
        "|production:min_pooled=25,min_observer=5,"
        "alpha=1e-04,min_effect=0.05"
    )


def persisted_funnel_sidecar() -> dict:
    """The evidence funnel's sidecar, or a FAILURE - never a skip.

    These two tests are the only ones that check what the last regeneration
    actually wrote to disk, as opposed to what the code would produce if
    asked. Their own docstrings say so. Letting them skip on an absent
    artifact means the suite reports green on exactly the machine where the
    persisted fingerprint has never been checked at all, and an id minted
    from a stale label is indistinguishable from a correct one.

    So an absent artifact fails here. The suite's other artifact-dependent
    tests skip by design, because they measure frozen numbers that a machine
    without the dataset genuinely cannot measure; these two measure an
    IDENTITY, and an unverified identity is worse than a missing number.
    """
    import json
    import os

    from obsidianchain import provenance as prov

    root = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))
    sidecar = root / "processed" / ("evidence_funnel.parquet" + prov.META_SUFFIX)
    assert sidecar.is_file(), (
        f"{sidecar} is absent, so the persisted run fingerprint has not been "
        f"verified against the code that mints evidence ids. This test must "
        f"not skip: generate the artifact with "
        f"'make run ARGS=\"evidence-funnel\"'."
    )
    return json.loads(sidecar.read_text(encoding="utf-8"))


def test_the_persisted_row_config_is_the_combined_label() -> None:
    """The sidecar on disk must carry both halves, not just the probe half.

    This is the one that would actually catch a regression: the label could
    be correct in code and still not be what the last regeneration wrote.
    """
    meta = persisted_funnel_sidecar()
    assert meta["inputs"]["row_statistics_config"] == live_row_config()


def test_the_persisted_fingerprint_is_the_one_the_inputs_imply() -> None:
    """Recomputed from the sidecar's own inputs block and compared.

    A persisted digest that did not follow from the persisted inputs would
    make every evidence id unfalsifiable.
    """
    meta = persisted_funnel_sidecar()
    recomputed = rf.build_evidence_run_fingerprint(**meta["inputs"])
    assert recomputed == meta["run_fingerprint"]


def test_the_persisted_sidecar_actually_carries_a_fingerprint() -> None:
    """Guards the identity gate's own escape hatch.

    ``provenance_gate.require_artifact_identity`` returns early when a
    sidecar declares no fingerprint, because several artifacts in this
    project legitimately have none. That early return must never apply to
    THIS artifact: a funnel regenerated without a fingerprint would pass the
    torn-publish check by being unidentifiable rather than by being intact.
    """
    meta = persisted_funnel_sidecar()
    fingerprint = meta.get("run_fingerprint")
    assert isinstance(fingerprint, str) and len(fingerprint) == 64, (
        f"the evidence funnel sidecar declares run_fingerprint="
        f"{fingerprint!r}; without one the identity gate cannot detect a "
        f"torn publish and silently permits the artifact"
    )


PRODUCTION_MUTATIONS = {
    "min_pooled": dict(production_min_pooled=26),
    "min_observer": dict(production_min_observer=6),
    "alpha": dict(production_alpha=1e-3),
    "min_effect": dict(production_min_effect=0.06),
}


def fingerprint_for(**overrides) -> str:
    from obsidianchain import evidence_contract as contract

    base = dict(
        probe_min_pooled=1, probe_min_observer=2,
        production_min_pooled=25, production_min_observer=5,
        production_alpha=1e-4, production_min_effect=0.05,
    )
    return rf.build_evidence_run_fingerprint(
        chain_addr_tx_sha256=A,
        chain_universe_sha256=B,
        network_dataset_sha256=C,
        heuristics=rf.HEURISTICS_MULTI_INPUT,
        row_statistics_config=contract.combined_row_config_label(
            **{**base, **overrides}
        ),
    )


@pytest.mark.parametrize("name", sorted(PRODUCTION_MUTATIONS))
def test_changing_a_production_parameter_changes_the_fingerprint(name) -> None:
    """Each of the four is a determining input for the seven /2 columns."""
    assert fingerprint_for() != fingerprint_for(**PRODUCTION_MUTATIONS[name])


def test_the_four_production_mutations_are_mutually_distinct() -> None:
    """Not merely different from the baseline - different from each other.

    Four fingerprints that all differed from the base but collided with one
    another would still let one production change be mistaken for another.
    """
    digests = {
        name: fingerprint_for(**mutation)
        for name, mutation in PRODUCTION_MUTATIONS.items()
    }
    assert len(set(digests.values())) == 4, digests


def test_the_canonical_payload_carries_all_four_production_parameters() -> None:
    """Section 6: inspectable, so the digest can be audited by eye."""
    from obsidianchain import evidence_contract as contract

    payload = rf.canonical_payload(
        chain_addr_tx_sha256=A,
        chain_universe_sha256=B,
        network_dataset_sha256=C,
        heuristics=rf.HEURISTICS_MULTI_INPUT,
        row_statistics_config=contract.combined_row_config_label(
            probe_min_pooled=1, probe_min_observer=2,
            production_min_pooled=25, production_min_observer=5,
            production_alpha=1e-4, production_min_effect=0.05,
        ),
    )
    for term in ("min_pooled=25", "min_observer=5",
                 "alpha=1e-04", "min_effect=0.05"):
        assert term in payload, term
    # and the probe half is still there beside it
    assert "probe:min_pooled=1,min_observer=2" in payload


def test_a_probe_only_fingerprint_differs_from_the_combined_one() -> None:
    """The Phase 5.2 digest and the Phase 5.3 digest must not collide.

    They cannot describe the same artifact: one covers thirteen columns, the
    other twenty. If they agreed, a 5.2 id would resolve against a /2
    artifact and silently pick up a verdict it was never minted for.
    """
    probe_only = rf.build_evidence_run_fingerprint(
        chain_addr_tx_sha256=A,
        chain_universe_sha256=B,
        network_dataset_sha256=C,
        heuristics=rf.HEURISTICS_MULTI_INPUT,
        row_statistics_config=rf.PROBE_ROW_CONFIG,
    )
    assert probe_only != fingerprint_for()


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
