"""Durable artifacts must carry their own provenance.

Phase 4 audit finding 6.1: every terminal renderer in this project carries a
SYNTHETIC banner and no output file carried anything. A row of
``phase33_decisions.csv`` opened in a spreadsheet, detached from the run that
produced it, was indistinguishable from a measurement about two real Bitcoin
entities.

Every test here **reads the artifact back off disk**. Asserting on the frame
in memory would pass while the file on disk stayed bare, which is the exact
failure being fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain import provenance as prov


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"decision_id": [1, 2], "chi2": [64.4, 2.1], "blocked": [True, False]}
    )


def synthetic_control(**overrides) -> prov.Provenance:
    defaults = dict(
        provenance_type=prov.ProvenanceType.SYNTHETIC_CONTROL,
        dataset_id="worlds/D",
        dataset_sha256="deadbeef",
        world="D",
        synthetic_network=True,
        generator_version="2.0.0",
        production_rule={"min_pooled_observations": 25, "alpha": 1e-4},
    )
    defaults.update(overrides)
    return prov.Provenance(**defaults)


# ---- the three types are distinguishable -------------------------------


def test_there_are_exactly_three_provenance_types() -> None:
    """DEMO alone is not enough, which is the whole point of the enum.

    Phase 3.3 controlled worlds are synthetic but are not demonstration
    scenarios. Collapsing them into a single ``demo`` boolean would
    misrepresent both.
    """
    assert {t.value for t in prov.ProvenanceType} == {
        "PRODUCTION",
        "SYNTHETIC_CONTROL",
        "DEMO",
    }


def test_synthetic_control_is_not_demo_and_not_a_measurement() -> None:
    record = synthetic_control()
    assert record.demo is False
    assert record.is_measurement is False


def test_demo_is_flagged_both_ways() -> None:
    record = prov.Provenance(
        provenance_type=prov.ProvenanceType.DEMO, dataset_id="demo"
    )
    assert record.demo is True
    assert record.is_measurement is False


def test_production_over_synthetic_network_is_not_a_measurement() -> None:
    """The frozen dataset's chain is real; its announcements are generated.

    So a chi-square computed from it is not a measurement of Bitcoin, and
    ``is_measurement`` is derived rather than settable so nobody can claim
    otherwise by filling in a field.
    """
    record = prov.Provenance(
        provenance_type=prov.ProvenanceType.PRODUCTION,
        dataset_id="elliptic++/frozen-september-2026",
        synthetic_network=True,
    )
    assert record.is_measurement is False


def test_production_over_chain_only_is_a_measurement() -> None:
    record = prov.Provenance(
        provenance_type=prov.ProvenanceType.PRODUCTION,
        dataset_id="elliptic++/cospend",
        synthetic_network=None,
    )
    assert record.is_measurement is True


# ---- what lands on disk ------------------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_every_row_carries_the_marker_on_disk(
    tmp_path: Path, frame: pd.DataFrame, suffix: str
) -> None:
    """A row is the unit that gets copied into an email or a slide."""
    path = prov.write_frame(
        frame, tmp_path / f"decisions{suffix}", synthetic_control()
    )
    reloaded = (
        pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
    )
    assert prov.PROVENANCE_COLUMN in reloaded.columns
    assert (reloaded[prov.PROVENANCE_COLUMN] == "SYNTHETIC_CONTROL").all()
    assert reloaded[prov.PROVENANCE_COLUMN].notna().all()
    assert len(reloaded) == len(frame)


def test_the_marker_is_the_first_column(tmp_path: Path, frame) -> None:
    """Visible before any number, not buried past the last one."""
    path = prov.write_frame(frame, tmp_path / "d.csv", synthetic_control())
    assert pd.read_csv(path).columns[0] == prov.PROVENANCE_COLUMN


def test_network_derived_rows_also_say_the_network_was_synthetic(
    tmp_path: Path, frame
) -> None:
    """Two columns, two different facts - not duplicated metadata.

    ``provenance_type`` says which pipeline produced the row;
    ``synthetic_network`` says whether the announcements behind the statistic
    were generated. A PRODUCTION row over synthetic announcements needs both.
    """
    path = prov.write_frame(frame, tmp_path / "d.csv", synthetic_control())
    reloaded = pd.read_csv(path)
    assert prov.SYNTHETIC_NETWORK_COLUMN in reloaded.columns
    assert reloaded[prov.SYNTHETIC_NETWORK_COLUMN].all()


def test_chain_only_artifacts_omit_the_network_column(
    tmp_path: Path, frame
) -> None:
    """Optional means optional: no column for a question that never arises."""
    record = prov.Provenance(
        provenance_type=prov.ProvenanceType.PRODUCTION,
        dataset_id="elliptic++/cospend",
        synthetic_network=None,
    )
    reloaded = pd.read_csv(prov.write_frame(frame, tmp_path / "d.csv", record))
    assert prov.SYNTHETIC_NETWORK_COLUMN not in reloaded.columns
    assert prov.PROVENANCE_COLUMN in reloaded.columns


def test_the_sidecar_holds_the_full_record(tmp_path: Path, frame) -> None:
    path = prov.write_frame(frame, tmp_path / "d.csv", synthetic_control())
    meta = json.loads(Path(str(path) + prov.META_SUFFIX).read_text())
    for key in (
        "schema",
        "provenance_type",
        "demo",
        "is_measurement",
        "dataset_id",
        "dataset_sha256",
        "world",
        "synthetic_network",
        "generator_version",
        "production_rule",
        "git_revision",
    ):
        assert key in meta, f"sidecar is missing {key}"
    assert meta["provenance_type"] == "SYNTHETIC_CONTROL"
    assert meta["dataset_id"] == "worlds/D"
    assert meta["dataset_sha256"] == "deadbeef"
    assert meta["world"] == "D"
    assert meta["generator_version"] == "2.0.0"
    assert meta["production_rule"]["min_pooled_observations"] == 25
    assert meta["is_measurement"] is False


def test_the_sidecar_records_the_rule_that_was_in_force(
    tmp_path: Path, frame
) -> None:
    """The rule is settable from the CLI, so a result is meaningless without it.

    A ``merges BLOCKED = 47`` figure produced under ``--min-pooled 2`` was
    previously byte-identical to one produced under the real rule of 25.
    """
    from obsidianchain.network.separation import SeparationConfig

    record = synthetic_control(
        production_rule=prov.rule_config(SeparationConfig())
    )
    path = prov.write_frame(frame, tmp_path / "d.csv", record)
    rule = prov.read_meta(path)["production_rule"]
    assert rule == {
        "min_pooled_observations": 25,
        "min_observer_observations": 5,
        "alpha": 1e-4,
        "min_effect": 0.05,
    }


def test_provenance_is_non_empty_everywhere_it_appears(
    tmp_path: Path, frame
) -> None:
    """"Present" is not enough - a blank marker is no marker."""
    path = prov.write_frame(frame, tmp_path / "d.csv", synthetic_control())
    reloaded = pd.read_csv(path)
    assert reloaded[prov.PROVENANCE_COLUMN].map(bool).all()
    meta = prov.read_meta(path)
    for key in ("provenance_type", "dataset_id", "schema"):
        assert str(meta[key]).strip(), f"{key} is blank"


def test_an_existing_column_of_the_same_name_is_replaced_not_duplicated(
    tmp_path: Path,
) -> None:
    """Re-stamping must not leave two conflicting markers in one row."""
    frame = pd.DataFrame(
        {prov.PROVENANCE_COLUMN: ["PRODUCTION"], "value": [1]}
    )
    reloaded = pd.read_csv(
        prov.write_frame(frame, tmp_path / "d.csv", synthetic_control())
    )
    assert list(reloaded.columns).count(prov.PROVENANCE_COLUMN) == 1
    assert reloaded[prov.PROVENANCE_COLUMN].iloc[0] == "SYNTHETIC_CONTROL"


def test_optional_fields_stay_absent_rather_than_invented(
    tmp_path: Path, frame
) -> None:
    """A dataset with no hash records null, not a placeholder."""
    record = prov.Provenance(
        provenance_type=prov.ProvenanceType.PRODUCTION, dataset_id="somewhere"
    )
    meta = prov.read_meta(prov.write_frame(frame, tmp_path / "d.csv", record))
    assert meta["dataset_sha256"] is None
    assert meta["world"] is None
    assert meta["generator_version"] is None
    assert meta["production_rule"] is None


def test_git_revision_is_absent_rather_than_wrong(tmp_path: Path) -> None:
    """No repository to read means None, never a fabricated hash."""
    assert prov.git_revision(tmp_path / "nowhere" / "deeper") is None


# ---- the writers actually route through it -----------------------------


def test_funnel_records_land_with_provenance(tmp_path: Path) -> None:
    from obsidianchain.eval import evidence_funnel

    result = evidence_funnel.FunnelResult(
        records=pd.DataFrame(
            {c: [0.0, 1.0] for c in evidence_funnel.RECORD_COLUMNS}
        )
    )
    path = tmp_path / "funnel.parquet"
    rows = evidence_funnel.write_records(
        result, path, provenance=synthetic_control()
    )
    assert rows == 2
    reloaded = pd.read_parquet(path)
    assert (reloaded[prov.PROVENANCE_COLUMN] == "SYNTHETIC_CONTROL").all()
    assert prov.read_meta(path)["dataset_id"] == "worlds/D"


def test_arrival_vectors_land_with_provenance(tmp_path: Path) -> None:
    from obsidianchain.network import arrivals

    observations = pd.DataFrame(
        {
            "txid": [1, 1, 2, 2],
            "observer_id": ["obs-00", "obs-01", "obs-00", "obs-01"],
            "timestamp_ms": [0.0, 900.0, 10.0, 1200.0],
        }
    )
    vectors = arrivals.build(observations)
    path = tmp_path / "arrivals.csv"
    arrivals.write(vectors, path, fmt="csv", provenance=synthetic_control())
    reloaded = pd.read_csv(path)
    assert (reloaded[prov.PROVENANCE_COLUMN] == "SYNTHETIC_CONTROL").all()
    assert prov.read_meta(path)["synthetic_network"] is True


def test_entity_labels_land_with_provenance(tmp_path: Path) -> None:
    from obsidianchain.io import entity_labels

    frame = pd.DataFrame({"address": ["a", "b"], "entity": ["x", "y"]})
    path = tmp_path / "labels.csv"
    assert entity_labels.write(
        frame,
        path,
        provenance=prov.Provenance(
            provenance_type=prov.ProvenanceType.PRODUCTION,
            dataset_id="elliptic++/address_labels",
        ),
    ) == 2
    assert (pd.read_csv(path)[prov.PROVENANCE_COLUMN] == "PRODUCTION").all()
    assert prov.read_meta(path)["is_measurement"] is True


def test_writers_without_provenance_are_unchanged(tmp_path: Path) -> None:
    """The parameter is additive: the old call shape still writes a bare frame.

    This is what keeps the change safe to land - no existing caller had to
    move at the same time as the writers.
    """
    from obsidianchain.io import entity_labels

    frame = pd.DataFrame({"address": ["a"], "entity": ["x"]})
    path = tmp_path / "bare.csv"
    entity_labels.write(frame, path)
    assert list(pd.read_csv(path).columns) == ["address", "entity"]
    assert not Path(str(path) + prov.META_SUFFIX).exists()


# ---- the CLI leaves nothing bare ---------------------------------------


def test_no_cli_command_writes_a_frame_without_provenance() -> None:
    """Source-level: a bare ``to_csv`` in the CLI is how this regressed.

    The check is on the CLI rather than on the library because the library
    writers keep an unprovenanced path for tests and for callers that only
    want a frame on disk. Every *command* must pass one.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "obsidianchain" / "cli.py"
    ).read_text(encoding="utf-8")
    executable = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines()
    )
    for bare in ("to_csv(", "to_parquet("):
        assert bare not in executable, (
            f"cli.py calls {bare} directly; route it through "
            f"provenance.write_frame so the artifact carries its own record"
        )


def test_the_demo_envelope_uses_the_shared_vocabulary() -> None:
    """The demo's own flag system and the shared enum must not drift."""
    from obsidianchain.demo import api

    assert api.PROVENANCE == "SYNTHETIC_DEMONSTRATION"
    envelope = api.build_envelope(
        scenarios=[], fixture=api.record(), rule=api.record(),
        totals=api.record(), seed=1, namespace="/tmp/demo",
    )
    assert envelope["provenance_type"] == prov.ProvenanceType.DEMO.value
