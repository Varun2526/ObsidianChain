"""Evidence fusion (pipeline/alerts.py): noisy-OR, budget severity, stacking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obsidianchain import geoip
from obsidianchain.ml.stacking import apply_stacker, load_stacker
from obsidianchain.pipeline import alerts as al
from obsidianchain.pipeline.orchestrator import run_pipeline

CAPTURE = Path("tests/data/synthetic_acceptance_capture.json")


def test_noisy_or_is_neutral_to_missing_lines_and_compounds_agreement() -> None:
    assert al.noisy_or([]) == 0.0
    one = al.noisy_or([(0.6, 1.0)])
    # Adding a silent line does not dilute, as a weighted mean would.
    assert al.noisy_or([(0.6, 1.0), (0.0, 0.5)]) == pytest.approx(one)
    # Two agreeing lines beat either alone.
    assert al.noisy_or([(0.6, 1.0), (0.6, 0.5)]) > one
    assert 0.0 <= al.noisy_or([(1.0, 1.0), (1.0, 1.0)]) <= 1.0


def _alerts(scores):
    return [al.AlertItem(alert_id=f"a{i}", cluster_id=str(i), primary_address=str(i),
                         member_addresses=[str(i)], fused_risk_score=s, severity="")
            for i, s in enumerate(scores)]


def test_severity_is_a_budget_over_rank_with_a_floor() -> None:
    items = _alerts(np.linspace(0.99, 0.5, 100))
    al.assign_budget_severity(items)
    sev = [a.severity for a in items]
    assert sev.count("CRITICAL") == 2 and sev.count("HIGH") == 6 and sev.count("MEDIUM") == 12
    quiet = _alerts([0.1, 0.05, 0.01])
    al.assign_budget_severity(quiet)
    assert {a.severity for a in quiet} == {"INFORMATIONAL"}


def test_the_stacker_is_monotone_in_both_inputs() -> None:
    stacker = load_stacker()
    if stacker is None:
        pytest.skip("stacker not built")
    raw = np.array([0.05, 0.05, 0.5])
    prop = np.array([0.0, 0.2, 0.0])
    p = apply_stacker(stacker, raw, prop)
    assert p[1] > p[0] and p[2] > p[0]


def test_propagation_is_counted_once(tmp_path) -> None:
    """With a seed present, propagation enters through the stacker and is not
    added again as its own fusion line."""
    out = run_pipeline(CAPTURE, runs_dir=tmp_path, geoip_provider=geoip.TestFixtureProvider(),
                       seed_addresses=["1PeelSource"])
    for a in out.alert_result.alerts:
        model = next(e for e in a.evidence if e.category == al.MODEL_SIGNAL)
        assert model.details["combined_with_propagation"] is (load_stacker() is not None)
        assert "corroborating_evidence_lines" in a.summary
        assert a.summary["confidence"] == pytest.approx(a.fused_risk_score, abs=1e-4)


def test_ranking_is_deterministic_and_dense(tmp_path) -> None:
    out = run_pipeline(CAPTURE, runs_dir=tmp_path, geoip_provider=geoip.TestFixtureProvider())
    ranks = [a.rank for a in out.alert_result.alerts]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [a.fused_risk_score for a in out.alert_result.alerts]
    assert scores == sorted(scores, reverse=True)
