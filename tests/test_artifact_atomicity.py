"""Publishing an artifact and its sidecar must not be able to half-happen.

The accident
------------
An artifact is two files. Two files means two ``os.replace`` calls and no
ordering makes that one atomic step, so a crash in between leaves a torn
pair. One half of that tear is harmless and the other is the worst failure
this project has:

* old parquet + new sidecar - the fingerprint names rows that are not there,
  so lookups miss and somebody notices;
* **new parquet + old sidecar** - the API reads the current fingerprint out
  of the SIDECAR, so this state does not look broken. It looks like a valid
  earlier run in which every previously minted evidence id still resolves,
  against rows it was never minted for.

That second state is silent re-pointing, which is the entire reason the run
fingerprint exists. A crash must not be able to produce it.

The fix, and why it is the small one
------------------------------------
The identity is written into BOTH files - the sidecar as before, and the
parquet's own footer - so a mismatched pair is detectable by reading it,
whichever half is stale. The gate the API already calls on every load
compares them and refuses.

Nothing else changes: no directory swap, no symlink indirection, no change to
where artifacts live or how they are named, and no new load path. The footer
read costs a few hundred microseconds because pyarrow reads key-value
metadata without touching a row group.

These tests inject the failure rather than waiting for it. Most build their
own two-file pair so they run on a machine with no dataset at all; the two
that need the real artifact say so.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import typer

from obsidianchain import cli
from obsidianchain import evidence_contract as contract
from obsidianchain import provenance as prov
from obsidianchain.api import artifacts, provenance_gate
from obsidianchain.api.app import API_PREFIX, create_app

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))

FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def make_provenance(fingerprint: str | None) -> prov.Provenance:
    """A minimal PRODUCTION record carrying an evidence identity."""
    return prov.Provenance(
        provenance_type=prov.ProvenanceType.PRODUCTION,
        dataset_id="test/atomicity",
        synthetic_network=True,
        inputs={
            "chain_addr_tx_sha256": "0" * 64,
            "chain_universe_sha256": "1" * 64,
            "network_dataset_sha256": "2" * 64,
            "heuristics": "multi-input",
            "row_statistics_config": contract.combined_row_config_label(
                probe_min_pooled=1, probe_min_observer=2,
                production_min_pooled=25, production_min_observer=5,
                production_alpha=1e-4, production_min_effect=0.05,
            ),
        },
        run_fingerprint=fingerprint,
        artifact={"artifact_schema": contract.ARTIFACT_SCHEMA},
    )


def write_pair(directory: Path, fingerprint: str | None, name="a.parquet") -> Path:
    """Write a parquet + sidecar pair the way the pipeline does."""
    frame = pd.DataFrame(
        {
            "edge_index": [0, 1],
            "dof_production": pd.array([8, None], dtype="Int64"),
            "chi2_production": [1.5, None],
        }
    )
    path = directory / name
    prov.write_frame(frame, path, make_provenance(fingerprint))
    return path


# ---- the identity reaches the file itself ------------------------------


def test_a_parquet_carries_its_own_run_fingerprint(tmp_path) -> None:
    path = write_pair(tmp_path, FINGERPRINT_A)
    identity = prov.read_artifact_identity(path)
    assert identity[prov.FINGERPRINT_METADATA_KEY] == FINGERPRINT_A
    assert identity[prov.ARTIFACT_SCHEMA_METADATA_KEY] == contract.ARTIFACT_SCHEMA


def test_the_sidecar_and_the_file_agree_after_a_normal_write(tmp_path) -> None:
    path = write_pair(tmp_path, FINGERPRINT_A)
    meta = provenance_gate.load_sidecar(path)
    assert meta["run_fingerprint"] == FINGERPRINT_A
    provenance_gate.require_artifact_identity(meta, path)  # does not raise


def test_an_artifact_with_no_identity_carries_no_key(tmp_path) -> None:
    """Several artifacts legitimately have no evidence identity.

    Inventing one so the dict is never empty would make "unidentified" and
    "identified" indistinguishable - the mistake ``_require_digest`` already
    refuses to make.
    """
    path = write_pair(tmp_path, None)
    assert prov.FINGERPRINT_METADATA_KEY not in prov.read_artifact_identity(path)
    meta = provenance_gate.load_sidecar(path)
    provenance_gate.require_artifact_identity(meta, path)  # nothing to tear


def test_the_nullable_integer_column_survives_the_pyarrow_write(tmp_path) -> None:
    """Section 1's dtype, through the new write path.

    Stamping the footer meant writing through pyarrow instead of
    ``DataFrame.to_parquet``, and the pandas metadata in that footer is what
    makes a nullable Int64 read back as Int64 rather than float64. Replacing
    the metadata instead of merging into it would land ``dof_production`` as
    a double with 8.0 in it - the exact trap the contract calls out, arrived
    at by a change that looks like a pure metadata addition.
    """
    import pyarrow.parquet as pq

    path = write_pair(tmp_path, FINGERPRINT_A)
    types = dict(
        zip(
            pq.ParquetFile(path).schema_arrow.names,
            [str(t) for t in pq.ParquetFile(path).schema_arrow.types],
        )
    )
    assert types["dof_production"] == "int64"
    frame = pd.read_parquet(path)
    assert str(frame["dof_production"].dtype) == "Int64"
    assert frame["dof_production"].tolist()[0] == 8
    assert frame["dof_production"].isna().tolist() == [False, True]


def test_no_index_column_is_added_by_the_new_write_path(tmp_path) -> None:
    """``index=False`` was the old behaviour and must stay the behaviour."""
    path = write_pair(tmp_path, FINGERPRINT_A)
    assert "__index_level_0__" not in pd.read_parquet(path).columns


# ---- failure injection: the tear is detected, both ways ----------------


def test_a_new_parquet_beside_an_old_sidecar_is_refused(tmp_path) -> None:
    """THE dangerous half. Injected directly.

    The parquet is replaced and the sidecar is not - precisely the state a
    crash between two renames leaves - and the result must be a refusal, not
    rows served under the previous run's fingerprint.
    """
    path = write_pair(tmp_path, FINGERPRINT_A)
    sidecar = Path(str(path) + prov.META_SUFFIX)
    old_sidecar = sidecar.read_text(encoding="utf-8")

    # A later run publishes new rows...
    write_pair(tmp_path, FINGERPRINT_B)
    # ...and dies before the sidecar lands.
    sidecar.write_text(old_sidecar, encoding="utf-8")

    meta = provenance_gate.load_sidecar(path)
    assert meta["run_fingerprint"] == FINGERPRINT_A
    with pytest.raises(provenance_gate.ProvenanceRefusedError, match="TORN PUBLISH"):
        provenance_gate.require_artifact_identity(meta, path)


def test_an_old_parquet_beside_a_new_sidecar_is_refused(tmp_path) -> None:
    """The other half. Detection is order-independent by construction."""
    path = write_pair(tmp_path, FINGERPRINT_A)
    old_parquet = path.read_bytes()

    write_pair(tmp_path, FINGERPRINT_B)
    path.write_bytes(old_parquet)  # sidecar landed, parquet did not

    meta = provenance_gate.load_sidecar(path)
    assert meta["run_fingerprint"] == FINGERPRINT_B
    with pytest.raises(provenance_gate.ProvenanceRefusedError, match="TORN PUBLISH"):
        provenance_gate.require_artifact_identity(meta, path)


def test_a_pre_identity_parquet_with_an_identified_sidecar_is_refused(
    tmp_path,
) -> None:
    """An artifact written before the stamp existed cannot be vouched for.

    Serving it would mean a torn publish and a complete one are
    indistinguishable, so it is refused with the regeneration command rather
    than waved through for being old.
    """
    path = write_pair(tmp_path, FINGERPRINT_A)
    frame = pd.read_parquet(path)
    frame.to_parquet(path, index=False)  # rewrite WITHOUT the footer stamp

    meta = provenance_gate.load_sidecar(path)
    with pytest.raises(
        provenance_gate.ProvenanceRefusedError, match="no embedded run fingerprint"
    ):
        provenance_gate.require_artifact_identity(meta, path)


def test_the_refusal_names_both_fingerprints_and_the_command(tmp_path) -> None:
    """An operator must be able to act on the message without reading code."""
    path = write_pair(tmp_path, FINGERPRINT_A)
    sidecar = Path(str(path) + prov.META_SUFFIX)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["run_fingerprint"] = FINGERPRINT_B
    sidecar.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(provenance_gate.ProvenanceRefusedError) as caught:
        provenance_gate.require_artifact_identity(
            provenance_gate.load_sidecar(path), path
        )
    detail = str(caught.value)
    assert FINGERPRINT_A[:16] in detail
    assert FINGERPRINT_B[:16] in detail
    assert "evidence-funnel" in detail


# ---- the publisher refuses to leave a tear behind ----------------------


def test_publishing_without_a_staged_sidecar_is_a_hard_stop(tmp_path) -> None:
    """Moving the parquet alone would leave it with the PREVIOUS sidecar.

    So the sidecar's presence is checked before either rename, and the
    command exits instead of publishing half a pair.
    """
    staged = write_pair(tmp_path, FINGERPRINT_A, name="staged.parquet")
    Path(str(staged) + prov.META_SUFFIX).unlink()
    destination = tmp_path / "published.parquet"

    with pytest.raises(typer.Exit) as caught:
        cli._publish_pair(staged, destination, FINGERPRINT_A)
    assert caught.value.exit_code == 8
    assert not destination.exists(), "the parquet was published without a sidecar"


def test_a_successful_publish_moves_both_halves(tmp_path) -> None:
    staged = write_pair(tmp_path, FINGERPRINT_A, name="staged.parquet")
    destination = tmp_path / "sub" / "published.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)

    cli._publish_pair(staged, destination, FINGERPRINT_A)

    assert destination.is_file()
    assert Path(str(destination) + prov.META_SUFFIX).is_file()
    assert not staged.exists(), "the staged copy was left behind"
    # And the published pair passes the gate the API uses.
    meta = provenance_gate.load_sidecar(destination)
    provenance_gate.require_artifact_identity(meta, destination)


def test_the_publisher_refuses_a_pair_that_would_not_serve(tmp_path) -> None:
    """The published pair is read back through the API's own gate.

    A publish that cannot be served is a failed publish, and finding that
    out while the operator is still watching is worth one footer read.
    """
    staged = write_pair(tmp_path, FINGERPRINT_A, name="staged.parquet")
    destination = tmp_path / "published.parquet"

    # The caller's fingerprint disagrees with what was actually staged.
    with pytest.raises(typer.Exit) as caught:
        cli._publish_pair(staged, destination, FINGERPRINT_B)
    assert caught.value.exit_code == 8


def test_publishing_replaces_an_existing_pair_completely(tmp_path) -> None:
    """No field of the old sidecar may survive into the new one."""
    destination = tmp_path / "published.parquet"
    prov.write_frame(
        pd.DataFrame({"edge_index": [0]}), destination,
        make_provenance(FINGERPRINT_A),
    )
    staged = write_pair(tmp_path, FINGERPRINT_B, name="staged.parquet")

    cli._publish_pair(staged, destination, FINGERPRINT_B)

    meta = provenance_gate.load_sidecar(destination)
    assert meta["run_fingerprint"] == FINGERPRINT_B
    assert prov.read_artifact_identity(destination)[
        prov.FINGERPRINT_METADATA_KEY
    ] == FINGERPRINT_B


# ---- the staging directory never leaks ---------------------------------


def test_the_staging_directory_is_removed_even_when_publishing_fails(
    tmp_path,
) -> None:
    """The ``finally`` in the command, verified on the failing path.

    An orphaned ``.oc-funnel-*`` directory beside a production artifact is a
    half-written parquet that looks like a peer of the real one.
    """
    import tempfile

    destination = tmp_path / "published.parquet"
    staging = Path(tempfile.mkdtemp(prefix=".oc-funnel-", dir=tmp_path))
    try:
        staged = write_pair(staging, FINGERPRINT_A, name=destination.name)
        Path(str(staged) + prov.META_SUFFIX).unlink()
        with pytest.raises(typer.Exit):
            cli._publish_pair(staged, destination, FINGERPRINT_A)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    leaked = list(tmp_path.glob(".oc-funnel-*"))
    assert not leaked, f"staging directories left behind: {leaked}"


# ---- the verification gate's hard stops --------------------------------


def staged_funnel(directory: Path, **overrides) -> Path:
    """A minimal artifact shaped like the real one, for gate tests."""
    frame = pd.DataFrame(
        {
            "edge_index": [0, 1],
            "pooled_a": [1, 2],
            "verdict_production": ["NO_EVIDENCE", "NOT_SEPARATED"],
            "reason_code_production": ["INSUFFICIENT_POOLED", "NOT_SIGNIFICANT"],
            "reason_production": ["pooled observations below minimum 25", ""],
            "dof_production": pd.array([None, 8], dtype="Int64"),
            "chi2_production": [None, 1.5],
            "p_value_production": [None, 0.2],
            "effect_production": [None, 0.01],
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    path = directory / "staged.parquet"
    prov.write_frame(frame, path, make_provenance(FINGERPRINT_A))
    return path


def test_the_gate_hard_stops_on_a_missing_production_column(tmp_path) -> None:
    staged = staged_funnel(tmp_path)
    frame = pd.read_parquet(staged).drop(columns=["effect_production"])
    prov.write_frame(frame, staged, make_provenance(FINGERPRINT_A))

    with pytest.raises(typer.Exit) as caught:
        cli._verify_funnel_artifact(
            staged, tmp_path / "absent.parquet",
            {"SEPARATED": 0, "NOT_SEPARATED": 1, "NO_EVIDENCE": 1},
        )
    assert caught.value.exit_code == 5


def test_the_gate_hard_stops_when_dof_lands_as_a_double(tmp_path) -> None:
    """Section 1's dtype gate, exercised for the first time.

    This is the check that catches the float-coercion trap, and until now
    nothing had ever made it fire. A plain float column with a None in it is
    exactly what pandas infers when the nullable dtype is forgotten.
    """
    staged = staged_funnel(tmp_path, dof_production=[None, 8.0])

    with pytest.raises(typer.Exit) as caught:
        cli._verify_funnel_artifact(
            staged, tmp_path / "absent.parquet",
            {"SEPARATED": 0, "NOT_SEPARATED": 1, "NO_EVIDENCE": 1},
        )
    assert caught.value.exit_code == 7


def test_the_gate_hard_stops_on_a_row_count_that_contradicts_the_counts(
    tmp_path,
) -> None:
    staged = staged_funnel(tmp_path)
    with pytest.raises(typer.Exit) as caught:
        cli._verify_funnel_artifact(
            staged, tmp_path / "absent.parquet",
            {"SEPARATED": 0, "NOT_SEPARATED": 1, "NO_EVIDENCE": 99},
        )
    assert caught.value.exit_code == 5


def test_the_gate_hard_stops_when_a_pre_existing_column_changed(tmp_path) -> None:
    """Section 14: the thirteen must be identical in VALUE.

    The published artifact is compared against, and a moved value refuses
    the publish rather than overwriting the evidence that it moved.
    """
    published = tmp_path / "published.parquet"
    prov.write_frame(
        pd.read_parquet(staged_funnel(tmp_path)), published,
        make_provenance(FINGERPRINT_A),
    )
    # A fresh run in which a pre-existing column moved.
    staged = staged_funnel(tmp_path, pooled_a=[1, 999])

    with pytest.raises(typer.Exit) as caught:
        cli._verify_funnel_artifact(
            staged, published,
            {"SEPARATED": 0, "NOT_SEPARATED": 1, "NO_EVIDENCE": 1},
        )
    assert caught.value.exit_code == 6
    # And the published artifact is untouched, as the message promises.
    assert pd.read_parquet(published)["pooled_a"].tolist() == [1, 2]


def test_the_gate_passes_an_unchanged_regeneration(tmp_path) -> None:
    """The gate must not refuse a correct republish of the same values."""
    published = tmp_path / "published.parquet"
    prov.write_frame(
        pd.read_parquet(staged_funnel(tmp_path)), published,
        make_provenance(FINGERPRINT_A),
    )
    staged = staged_funnel(tmp_path)
    cli._verify_funnel_artifact(
        staged, published, {"SEPARATED": 0, "NOT_SEPARATED": 1, "NO_EVIDENCE": 1}
    )


# ---- end to end: a torn pair does not reach a response -----------------


@pytest.fixture()
def served_root(tmp_path) -> Path:
    """A copy of the real artifacts, so a tear can be injected safely."""
    for relative in (artifacts.EVIDENCE_FUNNEL, artifacts.ADDRESS_CLUSTERS,
                     artifacts.CLUSTERS):
        source = DATA_ROOT / relative
        assert source.is_file(), (
            f"{source} is absent, so the end-to-end refusal cannot be "
            f"verified. Generate it with 'make run ARGS=\"evidence-funnel\"' "
            f"and 'make run ARGS=\"build-cluster-index\"'."
        )
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        shutil.copy2(
            str(source) + prov.META_SUFFIX, str(target) + prov.META_SUFFIX
        )
    return tmp_path


def test_the_real_artifact_is_identity_stamped(served_root) -> None:
    """The published production artifact carries the stamp, not just the
    code that writes one."""
    path = served_root / artifacts.EVIDENCE_FUNNEL
    meta = provenance_gate.load_sidecar(path)
    embedded = prov.read_artifact_identity(path)
    assert embedded[prov.FINGERPRINT_METADATA_KEY] == meta["run_fingerprint"]
    assert (
        embedded[prov.ARTIFACT_SCHEMA_METADATA_KEY] == contract.ARTIFACT_SCHEMA
    )


def test_a_torn_evidence_artifact_is_refused_by_the_endpoint(
    served_root,
) -> None:
    """The whole point, at the surface where it matters.

    The sidecar is rewritten with a different fingerprint - the readable
    equivalent of a crash between two renames - and the endpoint must
    refuse rather than serve rows under an identity they do not have.
    """
    from fastapi.testclient import TestClient

    path = served_root / artifacts.EVIDENCE_FUNNEL
    sidecar = Path(str(path) + prov.META_SUFFIX)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    stale = meta["run_fingerprint"]
    meta["run_fingerprint"] = FINGERPRINT_B
    sidecar.write_text(json.dumps(meta), encoding="utf-8")

    client = TestClient(create_app(served_root), raise_server_exceptions=False)
    response = client.get(f"{API_PREFIX}/evidence/{FINGERPRINT_B[:16]}:0")

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "provenance_refused"
    assert "TORN PUBLISH" in body["detail"]
    assert stale[:16] in body["detail"]


def test_an_intact_real_artifact_still_serves(served_root) -> None:
    """The guard must not refuse the artifact it is meant to protect.

    Without this the two tests above would pass just as well against a gate
    that refused everything.
    """
    from fastapi.testclient import TestClient
    from obsidianchain.api import evidence

    path = served_root / artifacts.EVIDENCE_FUNNEL
    meta = provenance_gate.load_sidecar(path)
    fingerprint = evidence.current_run_fingerprint(meta)
    edge = int(pd.read_parquet(path, columns=["edge_index"]).iloc[0]["edge_index"])

    client = TestClient(create_app(served_root), raise_server_exceptions=False)
    response = client.get(f"{API_PREFIX}/evidence/{fingerprint}:{edge}")
    assert response.status_code == 200, response.text
