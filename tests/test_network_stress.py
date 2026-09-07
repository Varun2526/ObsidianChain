"""Phase 2 stress cases and hand-verified arrival representation.

TEST CLASSIFICATION
-------------------
[INSTRUMENTATION] - determinism and correct reshaping. Must pass exactly.
[MECHANISM]       - shows what the synthetic model was built to represent.
                    NOT evidence about Bitcoin.

SCIENTIFIC VALIDATION is absent here by design. Phase 2 cannot supply it.
It needs real mainnet observations, controlled nodes we operate, and known
ground truth. A stress scenario that degrades the signal proves the
instrumentation is honest, not that any inference is accurate.

The point of the stress cases is to guarantee Phase 3 meets hard inputs. A
scenario producing fewer usable transactions has done its job.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from obsidianchain.network import arrivals, audit, synthetic

TXIDS = np.arange(1, 801)
BASE = replace(synthetic.FROZEN_SEPTEMBER_2026, n_origins=24, n_observers=6)


def vectors_for(config, txids=TXIDS):
    observations, nodes, observers, truth = synthetic.generate(txids, config)
    ids = sorted(observers["observer_id"].tolist())
    v = arrivals.build(observations, observer_ids=ids)
    bc_ips = synthetic.known_broadcaster_ips(nodes)
    bc = set(observations.loc[observations["peer_ip"].isin(bc_ips), "txid"])
    return v, bc, truth, observations


# =====================================================================
# Section 3: arrival-vector representation, hand verified
# =====================================================================


def test_observer_ordering_is_deterministic() -> None:
    """[INSTRUMENTATION] Columns follow the supplied axis, not input order."""
    shuffled = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["obs-02", "obs-00", "obs-01"],
            "timestamp_ms": [300.0, 100.0, 200.0],
        }
    )
    axis = ["obs-00", "obs-01", "obs-02"]
    v = arrivals.build(shuffled, observer_ids=axis)
    assert v.observer_ids == axis
    assert list(v.absolute_ms[0]) == [100.0, 200.0, 300.0]


def test_timestamps_map_to_the_correct_observer() -> None:
    """[INSTRUMENTATION] obs-01 saw it at 250 ms; that value must land in its column."""
    records = pd.DataFrame(
        {
            "txid": [7, 7, 7],
            "observer_id": ["obs-00", "obs-01", "obs-02"],
            "timestamp_ms": [1000.0, 1250.0, 1010.0],
        }
    )
    v = arrivals.build(records)
    col = v.observer_ids.index("obs-01")
    assert v.absolute_ms[0, col] == 1250.0
    assert v.offsets_ms[0, col] == 250.0
    assert v.ranks[0, col] == 2
    assert v.first_observer[0] == "obs-00"


def test_missing_observation_is_explicit_never_zero() -> None:
    """[INSTRUMENTATION] A missing sighting must not look like a fast one."""
    records = pd.DataFrame(
        {
            "txid": [1, 1],
            "observer_id": ["a", "c"],
            "timestamp_ms": [500.0, 900.0],
        }
    )
    v = arrivals.build(records, observer_ids=["a", "b", "c"])
    assert np.isnan(v.absolute_ms[0, 1]), "absent must be NaN, not 0"
    assert np.isnan(v.offsets_ms[0, 1])
    assert v.ranks[0, 1] == -1
    assert v.n_observed[0] == 2
    assert v.spread_ms[0] == 400.0, "spread uses only what was actually seen"


def test_duplicate_observations_resolve_deterministically() -> None:
    """[INSTRUMENTATION] Earliest wins, regardless of row order."""
    forward = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["a", "a", "a"],
            "timestamp_ms": [900.0, 100.0, 500.0],
        }
    )
    reversed_rows = forward.iloc[::-1].reset_index(drop=True)
    a = arrivals.build(forward)
    b = arrivals.build(reversed_rows)
    assert a.absolute_ms[0, 0] == 100.0
    np.testing.assert_array_equal(a.absolute_ms, b.absolute_ms)
    assert a.duplicates_collapsed == 2


def test_no_timestamp_is_silently_discarded() -> None:
    """[INSTRUMENTATION] Every collapsed record is counted, not dropped quietly."""
    records = pd.DataFrame(
        {
            "txid": [1, 1, 1, 2],
            "observer_id": ["a", "a", "b", "a"],
            "timestamp_ms": [10.0, 20.0, 30.0, 40.0],
        }
    )
    v = arrivals.build(records)
    distinct_pairs = len(records.drop_duplicates(["txid", "observer_id"]))
    assert v.duplicates_collapsed == len(records) - distinct_pairs
    assert int(np.count_nonzero(~np.isnan(v.absolute_ms))) == distinct_pairs


def test_vector_carries_no_origin_information() -> None:
    """[INSTRUMENTATION] Two identical timing patterns are indistinguishable.

    Whatever origin produced them, the vectors must be equal - the
    representation holds timing and nothing else.
    """
    pattern = [100.0, 340.0, 220.0]
    frames = []
    for txid in (1, 2):
        frames.append(
            pd.DataFrame(
                {
                    "txid": txid,
                    "observer_id": ["a", "b", "c"],
                    "timestamp_ms": [p + txid * 10_000 for p in pattern],
                }
            )
        )
    v = arrivals.build(pd.concat(frames, ignore_index=True))
    np.testing.assert_allclose(v.offsets_ms[0], v.offsets_ms[1])
    np.testing.assert_array_equal(v.ranks[0], v.ranks[1])


# =====================================================================
# Section 2: broadcaster handling, the six required cases
# =====================================================================


def test_broadcaster_transaction_is_no_evidence() -> None:
    """[INSTRUMENTATION] Case 1: a broadcaster transaction."""
    v, bc, _, _ = vectors_for(replace(BASE, broadcaster_fraction=0.3))
    labels = arrivals.classify_evidence(v, bc)
    blocked = labels[labels["txid"].isin(bc)]
    assert len(blocked) > 0
    assert (blocked["evidence"] == arrivals.Evidence.NO_EVIDENCE.value).all()
    assert (blocked["no_evidence_reason"] == "known_broadcaster").all()


def test_ordinary_transaction_is_usable() -> None:
    """[MECHANISM] Case 2: an ordinary transaction retains ordering."""
    v, bc, _, _ = vectors_for(replace(BASE, broadcaster_fraction=0.3))
    labels = arrivals.classify_evidence(v, bc)
    ordinary = labels[~labels["txid"].isin(bc)]
    assert (ordinary["evidence"] == arrivals.Evidence.USABLE.value).mean() > 0.85


def test_observer_seeing_a_broadcaster_still_yields_no_evidence() -> None:
    """[INSTRUMENTATION] Case 3: one observer's sighting cannot rescue it."""
    records = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["a", "b", "c"],
            "timestamp_ms": [100.0, 8000.0, 100.0],  # wide spread on purpose
        }
    )
    v = arrivals.build(records)
    labels = arrivals.classify_evidence(v, broadcaster_txids={1})
    assert labels.iloc[0]["evidence"] == arrivals.Evidence.NO_EVIDENCE.value
    assert labels.iloc[0]["no_evidence_reason"] == "known_broadcaster"


def test_missing_observer_case() -> None:
    """[INSTRUMENTATION] Case 4: a gap does not change the classification."""
    records = pd.DataFrame(
        {
            "txid": [1, 1],
            "observer_id": ["a", "c"],
            "timestamp_ms": [100.0, 6000.0],
        }
    )
    v = arrivals.build(records, observer_ids=["a", "b", "c"])
    labels = arrivals.classify_evidence(v)
    assert labels.iloc[0]["evidence"] == arrivals.Evidence.USABLE.value
    assert v.n_observed[0] == 2


def test_single_observation_is_no_evidence() -> None:
    """[INSTRUMENTATION] Case 5: one sighting cannot order anything."""
    records = pd.DataFrame(
        {"txid": [1], "observer_id": ["a"], "timestamp_ms": [100.0]}
    )
    v = arrivals.build(records, observer_ids=["a", "b"])
    labels = arrivals.classify_evidence(v)
    assert labels.iloc[0]["evidence"] == arrivals.Evidence.NO_EVIDENCE.value
    assert labels.iloc[0]["no_evidence_reason"] == "too_few_observers"


def test_duplicate_observation_case() -> None:
    """[INSTRUMENTATION] Case 6: duplicates do not inflate observer count."""
    records = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["a", "a", "b"],
            "timestamp_ms": [100.0, 150.0, 4000.0],
        }
    )
    v = arrivals.build(records)
    assert v.n_observed[0] == 2
    labels = arrivals.classify_evidence(v)
    assert labels.iloc[0]["evidence"] == arrivals.Evidence.USABLE.value


def test_no_evidence_can_never_become_a_constraint() -> None:
    """[INSTRUMENTATION] The guard converts a silent error into a loud one."""
    v, bc, _, _ = vectors_for(replace(BASE, broadcaster_fraction=0.3))
    labels = arrivals.classify_evidence(v, bc)
    with pytest.raises(ValueError, match="NO_EVIDENCE"):
        arrivals.assert_no_constraint_permitted(labels, list(bc)[:3])
    usable = labels.loc[
        labels["evidence"] == arrivals.Evidence.USABLE.value, "txid"
    ].tolist()
    arrivals.assert_no_constraint_permitted(labels, usable[:3])  # must not raise


# =====================================================================
# Section 4: stress scenarios
# =====================================================================


@pytest.mark.parametrize("jitter_sd", [50.0, 150.0, 500.0])
def test_A_increased_clock_jitter(jitter_sd: float) -> None:
    """[INSTRUMENTATION] Higher jitter must not break the representation."""
    v, bc, _, _ = vectors_for(replace(BASE, clock_jitter_sd_ms=jitter_sd))
    assert v.n_transactions == len(TXIDS)
    assert np.isfinite(v.spread_ms).all()
    labels = arrivals.classify_evidence(v, bc)
    assert set(labels["evidence"]) <= {
        arrivals.Evidence.USABLE.value, arrivals.Evidence.NO_EVIDENCE.value
    }


def test_A_jitter_reduces_usable_share_monotonically_enough() -> None:
    """[MECHANISM] Ordering degrades as jitter approaches propagation delay."""
    shares = []
    for sd in (25.0, 250.0, 2000.0):
        v, bc, _, _ = vectors_for(replace(BASE, clock_jitter_sd_ms=sd))
        labels = arrivals.classify_evidence(v, bc)
        shares.append(
            float((labels["evidence"] == arrivals.Evidence.USABLE.value).mean())
        )
    assert shares[0] >= shares[-1], "extreme jitter must not make things cleaner"


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.6])
def test_B_missing_observations(rate: float) -> None:
    """[INSTRUMENTATION] Dropped sightings stay explicitly absent."""
    v, _, _, observations = vectors_for(replace(BASE, missing_observation_rate=rate))
    assert len(observations) < len(TXIDS) * BASE.n_observers
    assert np.isnan(v.absolute_ms).any()
    assert (v.n_observed <= BASE.n_observers).all()
    assert not (v.absolute_ms == 0).any(), "absent must never be zero"


def test_C_observer_failure_removes_the_column_contents() -> None:
    """[INSTRUMENTATION] A dead vantage point records nothing at all."""
    config = replace(BASE, failed_observers=("obs-00", "obs-01"))
    observations, nodes, observers, _ = synthetic.generate(TXIDS, config)
    assert not observations["observer_id"].isin(["obs-00", "obs-01"]).any()

    v = arrivals.build(observations, observer_ids=sorted(observers["observer_id"]))
    dead = v.observer_ids.index("obs-00")
    assert np.isnan(v.absolute_ms[:, dead]).all()
    assert (v.ranks[:, dead] == -1).all()
    assert (v.n_observed <= BASE.n_observers - 2).all()


def test_D_propagation_overlap_produces_similar_vectors() -> None:
    """[MECHANISM] Different origins, deliberately similar arrival patterns."""
    flat = synthetic.PropagationModel(
        same_asn_factor=0.97, same_region_factor=0.99
    )
    v_flat, _, truth_flat, _ = vectors_for(replace(BASE, propagation=flat))
    v_base, _, truth_base, _ = vectors_for(BASE)

    def separation(v, truth):
        report = audit.audit_identifiability(v, truth, BASE, sample=400)
        return report.separation_ratio

    overlapped = separation(v_flat, truth_flat)
    normal = separation(v_base, truth_base)
    assert np.isfinite(overlapped) and np.isfinite(normal)
    assert overlapped <= normal + 0.05, (
        f"removing proximity contrast should not increase separability "
        f"({overlapped:.3f} vs {normal:.3f})"
    )


def test_E_same_origin_vectors_are_not_identical() -> None:
    """[MECHANISM] One origin's own transactions must scatter, not clone."""
    v, _, truth, _ = vectors_for(BASE)
    lookup = truth.set_index("txid")["true_origin_id"].reindex(v.txids).to_numpy()
    biggest = pd.Series(lookup).value_counts().idxmax()
    rows = np.flatnonzero(lookup == biggest)[:50]
    assert rows.size >= 5
    block = v.offsets_ms[rows]
    distinct = {tuple(np.round(r, 3)) for r in block}
    assert len(distinct) > 1, "same origin must not yield identical vectors"


def test_F_broadcasters_excluded_under_every_scenario() -> None:
    """[INSTRUMENTATION] Broadcaster suppression survives all stress settings."""
    for config in (
        BASE,
        replace(BASE, clock_jitter_sd_ms=500.0),
        replace(BASE, missing_observation_rate=0.5),
        replace(BASE, failed_observers=("obs-00",)),
        replace(BASE, broadcaster_fraction=0.5),
    ):
        v, bc, _, _ = vectors_for(config)
        if not bc:
            continue
        labels = arrivals.classify_evidence(v, bc)
        blocked = labels[labels["txid"].isin(bc)]
        assert (
            blocked["evidence"] == arrivals.Evidence.NO_EVIDENCE.value
        ).all(), f"broadcaster leaked into USABLE under {config}"


def test_stress_suite_runs_and_covers_every_scenario_class() -> None:
    """[INSTRUMENTATION] The audit's scenario set is complete."""
    results = audit.run_stress_scenarios(TXIDS[:300], BASE)
    names = {r.name for r in results}
    assert "baseline" in names
    assert any(n.startswith("jitter-") for n in names)          # A
    assert any(n.startswith("missing-") for n in names)         # B
    assert any(n.startswith("observers-down-") for n in names)  # C
    assert "overlapping-origins" in names                        # D
    assert "same-origin-scatter" in names                        # E
    assert "broadcaster-heavy" in names                          # F
    assert all(r.n_records > 0 for r in results)


# =====================================================================
# Section 5: identifiability audit
# =====================================================================


def test_identifiability_reports_every_required_measure() -> None:
    """[INSTRUMENTATION] The audit must not quietly omit a measure."""
    v, _, truth, _ = vectors_for(BASE)
    report = audit.audit_identifiability(v, truth, BASE, sample=500)
    assert report.n_origins > 0
    assert report.n_observers == BASE.n_observers
    assert report.tx_per_origin_max >= report.tx_per_origin_min >= 0
    assert 0.0 <= report.broadcaster_fraction <= 1.0
    assert report.missing_observation_rate == BASE.missing_observation_rate
    assert report.jitter_sd_ms == BASE.clock_jitter_sd_ms
    assert report.delay_percentiles and "p50" in report.delay_percentiles
    assert report.within_origin_distance and report.between_origin_distance
    assert np.isfinite(report.separation_ratio)
    assert report.near_duplicate_vectors >= 0


def test_leakage_audit_flags_a_polluted_directory(tmp_path) -> None:
    """[INSTRUMENTATION] The audit must actually detect a leak."""
    directory = tmp_path / synthetic.OBSERVATIONS_DIR
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "txid": [1], "observer_id": ["a"], "peer_ip": ["192.0.2.1"],
            "peer_port": [8333], "peer_asn": [64512], "timestamp_ms": [1.0],
            "true_origin_id": ["origin-001"],
        }
    ).to_csv(directory / "observations.csv", index=False)
    report = audit.audit_leakage(tmp_path)
    assert not report.clean
    assert report.boundary_error is not None
