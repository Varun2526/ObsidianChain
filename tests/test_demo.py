"""Tests for the demonstration scenarios.

SCIENTIFIC VALIDATION NOTICE: the fixtures under test are synthetic and were
written by the same people as the engine. Every test here checks that the
demonstration is honestly labelled and behaves as it claims. None is evidence
about Bitcoin.

Two properties matter more than the rest and are tested hardest:

* **the frozen production namespace is untouched** by anything the demo does;
* **the DEMO flag reaches the UI**, and an unflagged payload does not render.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from obsidianchain.demo import api, audit, report, runner, scenarios
from obsidianchain.demo.scenarios import DemoOutcome
from obsidianchain.network import boundary, separation, synthetic


# ---- shared fixtures ---------------------------------------------------


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Path:
    """A data root holding a stand-in production namespace and a demo root.

    The production side is a stand-in rather than the real dataset so this
    runs without the 1.6M-record capture present, but it sits exactly where
    the real one does - ``processed/network`` and ``processed/network_truth``
    under the same data root the demo is pointed at.
    """
    root = tmp_path_factory.mktemp("workspace")
    obs = root / "processed" / synthetic.OBSERVATIONS_DIR
    truth = root / "processed" / synthetic.TRUTH_DIR
    obs.mkdir(parents=True)
    truth.mkdir(parents=True)
    pd.DataFrame(
        {
            "txid": [1, 2],
            "observer_id": ["obs-00", "obs-01"],
            "peer_ip": ["192.0.2.1", "192.0.2.2"],
            "peer_port": [8333, 8333],
            "peer_asn": [64512, 64513],
            "timestamp_ms": [1.0, 2.0],
        }
    ).to_csv(obs / "observations.csv", index=False)
    (obs / "manifest.json").write_text(
        json.dumps({"dataset_sha256": "frozen-production-anchor"}),
        encoding="utf-8",
    )
    (truth / "ground_truth.csv").write_text("txid,true_origin_id\n1,origin-000\n")
    return root


def _tree_digest(root: Path) -> dict[str, str]:
    """Path -> content hash for every file under ``root``."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def demo_run(workspace: Path) -> runner.DemoRun:
    return runner.run(scenarios.default_demo_root(workspace), rebuild=True)


@pytest.fixture(scope="module")
def envelope(demo_run: runner.DemoRun) -> dict:
    return runner.to_envelope(demo_run)


# ---- the frozen production namespace is untouched ----------------------


def test_demo_writes_nothing_into_the_production_namespace(
    workspace: Path,
) -> None:
    """THE assertion. A demo run must leave processed/ byte-identical.

    The failure this guards against is a ``--data-root`` that reaches the
    generator: it would overwrite the frozen announcement records, and the
    frozen dataset hash anchors every Phase 2 and Phase 3 number in the
    project. Content is hashed rather than mtimes compared, because a
    rewrite with identical bytes is fine and a silent edit is not.
    """
    production = workspace / "processed"
    before = _tree_digest(production)
    assert before, "the stand-in production namespace should not be empty"

    runner.run(scenarios.default_demo_root(workspace), rebuild=True)

    after = _tree_digest(production)
    assert after == before, (
        "the demonstration modified the production namespace: "
        f"changed={sorted(set(before) ^ set(after))}"
    )


def test_demo_output_stays_inside_the_demo_namespace(workspace: Path) -> None:
    demo_root = scenarios.default_demo_root(workspace)
    result = runner.run(demo_root, rebuild=True)
    payload = runner.to_envelope(result)
    json_path = api.write_json(payload, demo_root / "output" / "scenarios.json")
    html_path = report.write_html(payload, demo_root / "output" / "index.html")

    for path in (json_path, html_path):
        assert demo_root in path.parents
    for path in workspace.rglob("*"):
        if path.is_file() and scenarios.DEMO_DIRNAME not in path.parts:
            assert "processed" in path.parts, (
                f"{path} was written outside both the demo namespace and the "
                f"stand-in production one"
            )


def test_frozen_generator_config_is_unchanged() -> None:
    """The demo must not have edited the frozen production configuration."""
    frozen = synthetic.FROZEN_SEPTEMBER_2026
    assert frozen.seed == 0
    assert frozen.n_observers == 8
    assert frozen.n_origins == 64
    assert frozen.broadcaster_fraction == 0.15
    assert frozen.missing_observation_rate == 0.02
    assert frozen.clock_bias_range_ms == 40.0
    assert frozen.clock_jitter_sd_ms == 25.0
    assert frozen.propagation.sigma == pytest.approx(
        synthetic.DECKER_WATTENHOFER_SIGMA
    )
    assert round(synthetic.DECKER_WATTENHOFER_SIGMA, 4) == 1.1506


def test_demo_network_config_differs_from_frozen_only_in_seed() -> None:
    """sigma and the topology are held; only the seed moves.

    A demonstration that also changed the physics would be demonstrating a
    different system, and the difference would be invisible in the output.
    """
    frozen = synthetic.FROZEN_SEPTEMBER_2026.describe()
    demo = scenarios.network_config().describe()
    differing = {k for k in frozen if frozen[k] != demo[k]}
    assert differing == {"seed"}, f"unexpected divergence: {sorted(differing)}"
    assert demo["seed"] == scenarios.DEMO_SEED


def test_production_rule_is_unchanged_and_guarded() -> None:
    """25 pooled per side, alpha 1e-4, effect 0.05 - and a guard that says so."""
    config = separation.SeparationConfig()
    assert config.min_pooled_observations == 25
    assert config.min_observer_observations == 5
    assert config.alpha == 1e-4
    assert config.min_effect == 0.05
    runner.assert_production_rule(config)  # must not raise


def test_a_loosened_rule_stops_the_demonstration() -> None:
    """Tuning the threshold to make a scenario fire must fail loudly."""
    loosened = separation.SeparationConfig(min_pooled_observations=5)
    with pytest.raises(runner.DemoRuleError, match="production rule"):
        runner.assert_production_rule(loosened)


# ---- namespace guard ---------------------------------------------------


def test_namespace_guard_refuses_a_root_that_is_not_named_demo(
    tmp_path: Path,
) -> None:
    with pytest.raises(scenarios.DemoNamespaceError, match="not a demonstration"):
        scenarios.assert_demo_namespace(tmp_path / "processed")


def test_namespace_guard_refuses_a_root_inside_production_directories(
    tmp_path: Path,
) -> None:
    inside = tmp_path / "processed" / synthetic.OBSERVATIONS_DIR / "demo"
    with pytest.raises(scenarios.DemoNamespaceError, match="production"):
        scenarios.assert_demo_namespace(inside)


def test_building_outside_the_demo_namespace_is_refused(tmp_path: Path) -> None:
    with pytest.raises(scenarios.DemoNamespaceError):
        scenarios.build_fixture(tmp_path / "processed")
    assert not (tmp_path / "processed").exists()


def test_default_demo_root_is_a_sibling_of_processed(tmp_path: Path) -> None:
    root = scenarios.default_demo_root(tmp_path)
    assert root == tmp_path / scenarios.DEMO_DIRNAME
    assert "processed" not in root.parts


# ---- determinism -------------------------------------------------------


def test_the_fixture_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    first = scenarios.build_fixture(tmp_path / "a" / "demo")
    second = scenarios.build_fixture(tmp_path / "b" / "demo")
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["record_count"] == second["record_count"]
    assert (
        (tmp_path / "a" / "demo" / "raw" / "AddrTx_edgelist.csv").read_bytes()
        == (tmp_path / "b" / "demo" / "raw" / "AddrTx_edgelist.csv").read_bytes()
    )


def test_two_runs_produce_the_same_payload(workspace: Path) -> None:
    demo_root = scenarios.default_demo_root(workspace)
    first = runner.to_envelope(runner.run(demo_root, rebuild=True))
    second = runner.to_envelope(runner.run(demo_root, rebuild=True))
    assert first == second


# ---- the DEMO flag reaches the UI --------------------------------------


def test_every_object_in_the_payload_is_flagged(envelope: dict) -> None:
    api.assert_demo_flagged(envelope)  # must not raise
    assert envelope[api.DEMO_FLAG_FIELD] is True
    assert envelope["provenance"] == api.PROVENANCE
    assert envelope["not_a_measurement"] is True
    for scenario in envelope["scenarios"]:
        assert scenario[api.DEMO_FLAG_FIELD] is True
        for decision in scenario["decisions"]:
            assert decision[api.DEMO_FLAG_FIELD] is True


def test_a_missing_flag_anywhere_is_caught(envelope: dict) -> None:
    """Depth is not an excuse: a nested object must be flagged too."""
    damaged = copy.deepcopy(envelope)
    del damaged["scenarios"][2]["decisions"][0][api.DEMO_FLAG_FIELD]
    with pytest.raises(api.DemoFlagError, match=r"scenarios\[2\]"):
        api.assert_demo_flagged(damaged)


def test_a_wrong_provenance_is_caught(envelope: dict) -> None:
    damaged = copy.deepcopy(envelope)
    damaged["scenarios"][0]["provenance"] = "PRODUCTION"
    with pytest.raises(api.DemoFlagError, match="provenance"):
        api.assert_demo_flagged(damaged)


def test_the_renderer_refuses_an_unflagged_payload(envelope: dict) -> None:
    """The flag is load-bearing: strip it and nothing renders at all."""
    damaged = copy.deepcopy(envelope)
    damaged[api.DEMO_FLAG_FIELD] = False
    with pytest.raises(api.DemoFlagError):
        report.render_html(damaged)
    with pytest.raises(api.DemoFlagError):
        report.format_terminal(damaged)


def test_writing_and_reading_json_both_enforce_the_flag(
    envelope: dict, tmp_path: Path
) -> None:
    path = api.write_json(envelope, tmp_path / "demo" / "scenarios.json")
    assert api.read_json(path) == envelope

    unflagged = copy.deepcopy(envelope)
    unflagged[api.DEMO_FLAG_FIELD] = False
    with pytest.raises(api.DemoFlagError):
        api.write_json(unflagged, tmp_path / "demo" / "bad.json")
    (tmp_path / "demo" / "raw.json").write_text(json.dumps(unflagged))
    with pytest.raises(api.DemoFlagError):
        api.read_json(tmp_path / "demo" / "raw.json")


def test_the_page_is_marked_everywhere_a_viewer_looks(envelope: dict) -> None:
    page = report.render_html(envelope)
    banner = envelope["banner"]
    assert page.count(banner) >= 3, "banner must survive a partial screenshot"
    assert f"<title>{banner}" in page
    assert page.count("SYNTHETIC DEMONSTRATION</span>") == len(
        envelope["scenarios"]
    ), "every scenario card carries its own badge"
    assert "class=\"hazard\"" in page


def test_the_page_states_what_the_demonstration_does_not_show(
    envelope: dict,
) -> None:
    """Both statements, same visual weight. Omitting the second is the lie."""
    page = report.render_html(envelope)
    assert envelope["statements"]["mechanism"] in page
    assert envelope["statements"]["frozen_dataset"] in page
    assert "What this does not show" in page
    assert "253,429" in page and "40 reached the pooled minimum" in page


def test_the_page_needs_no_network(envelope: dict) -> None:
    """Air-gapped: no external stylesheet, script, font or image."""
    page = report.render_html(envelope)
    for forbidden in ("http://", "https://", "<script", "src="):
        assert forbidden not in page, f"page reaches outside itself: {forbidden}"


def test_the_terminal_report_is_marked(envelope: dict) -> None:
    text = report.format_terminal(envelope)
    assert text.count(envelope["banner"]) >= 2
    assert envelope["statements"]["frozen_dataset"].split(".")[0] in " ".join(
        text.split()
    )


# ---- the scenarios behave as described ---------------------------------


def test_all_five_scenarios_reach_their_stated_outcome(
    demo_run: runner.DemoRun,
) -> None:
    observed = {r.spec.key: r.outcome for r in demo_run.results}
    assert observed == {
        "A": DemoOutcome.CANDIDATE,
        "B": DemoOutcome.MERGED,
        "C": DemoOutcome.BLOCKED,
        "D": DemoOutcome.CONTESTED,
        "E": DemoOutcome.ABSTAINED,
    }
    assert all(r.matches_expectation for r in demo_run.results)


def test_a_scenario_that_stops_behaving_as_described_fails_the_run(
    workspace: Path, monkeypatch
) -> None:
    """The demonstration asserts its own claims rather than displaying them."""
    misdescribed = tuple(
        spec if spec.key != "C"
        else scenarios.ScenarioSpec(
            **{
                **{
                    field: getattr(spec, field)
                    for field in spec.__dataclass_fields__
                },
                "expected_outcome": DemoOutcome.MERGED,
                "expected_verdict": "NOT_SEPARATED",
            }
        )
        for spec in scenarios.SCENARIOS
    )
    monkeypatch.setattr(scenarios, "SCENARIOS", misdescribed)
    with pytest.raises(runner.DemoExpectationError, match="scenario C"):
        runner.run(scenarios.default_demo_root(workspace))


def test_b_and_c_differ_only_in_origin(demo_run: runner.DemoRun) -> None:
    """The contrast is the demonstration. Same volume, opposite decisions."""
    by_key = {r.spec.key: r for r in demo_run.results}
    b, c = by_key["B"], by_key["C"]
    assert sorted(b.pooled.values()) == sorted(c.pooled.values())
    assert b.observations_seen == c.observations_seen
    assert b.usable_observations == c.usable_observations
    assert b.decisions[0].merged is True
    assert c.decisions[0].merged is False
    assert c.decisions[0].p_value < separation.SeparationConfig().alpha
    assert b.decisions[0].p_value > separation.SeparationConfig().alpha


def test_only_c_blocks_and_only_d_is_contested(demo_run: runner.DemoRun) -> None:
    blocked = {r.spec.key for r in demo_run.results if r.outcome is DemoOutcome.BLOCKED}
    contested = {r.spec.key for r in demo_run.results if r.contested}
    assert blocked == {"C"}
    assert contested == {"D"}
    assert demo_run.blocked == 1
    assert demo_run.contested == 1


def test_the_veto_changes_the_clustering(demo_run: runner.DemoRun) -> None:
    """Chain-only merges everything; the fused run keeps C's two sides apart."""
    assert demo_run.fused_clusters == demo_run.chain_only_clusters + 1
    by_key = {r.spec.key: r for r in demo_run.results}
    assert by_key["C"].chain_only_components == 1
    assert by_key["C"].fused_components == 2
    for key in ("A", "B", "D", "E"):
        assert by_key[key].chain_only_components == by_key[key].fused_components


def test_d_abstains_on_every_decision_yet_ends_contested(
    demo_run: runner.DemoRun,
) -> None:
    """The blind spot, stated as a test.

    Every merge decision has a side below the pooled minimum, so the veto
    never had a chance to act. The contradiction only exists once the
    component is complete.
    """
    d = next(r for r in demo_run.results if r.spec.key == "D")
    assert d.decisions, "D must actually propose merges"
    assert all(x.verdict == "NO_EVIDENCE" for x in d.decisions)
    assert all(x.merged for x in d.decisions)
    assert min(min(x.pooled_a, x.pooled_b) for x in d.decisions) < 25
    assert d.contradiction is not None
    assert d.contradiction.evidence.verdict is separation.Verdict.SEPARATED
    assert d.contradiction.evidence.n_a >= 25
    assert d.contradiction.evidence.n_b >= 25


def test_a_and_e_are_different_kinds_of_silence(demo_run: runner.DemoRun) -> None:
    """Never seen is not the same as seen and unusable."""
    by_key = {r.spec.key: r for r in demo_run.results}
    a, e = by_key["A"], by_key["E"]
    assert a.observations_seen == 0
    assert e.observations_seen > 0
    assert a.usable_observations == 0 and e.usable_observations == 0
    assert a.verdict == e.verdict == "NO_EVIDENCE"
    assert a.outcome is DemoOutcome.CANDIDATE
    assert e.outcome is DemoOutcome.ABSTAINED


def test_no_scenario_produces_a_must_link(demo_run: runner.DemoRun) -> None:
    """Cannot-link only. There is no verdict that forces a merge."""
    verdicts = {d.verdict for r in demo_run.results for d in r.decisions}
    assert verdicts <= {v.value for v in separation.Verdict}
    assert not any("MUST" in v or "SAME_ORIGIN" in v for v in verdicts)


# ---- the boundary still holds ------------------------------------------


def test_the_demo_fixture_carries_only_the_six_allowed_columns(
    workspace: Path, demo_run: runner.DemoRun
) -> None:
    processed = scenarios.default_demo_root(workspace) / "processed"
    observations = boundary.load_observations(processed)
    assert set(observations.columns) == boundary.PHASE3_ALLOWED_FIELDS


def test_the_demo_truth_directory_is_refused_like_any_other(
    workspace: Path, demo_run: runner.DemoRun
) -> None:
    """The demo writes ground truth, and the boundary refuses it just the same."""
    processed = scenarios.default_demo_root(workspace) / "processed"
    truth = processed / synthetic.TRUTH_DIR / "ground_truth.csv"
    assert truth.is_file()
    with pytest.raises(boundary.GroundTruthLeakError):
        boundary.load_observations(
            processed, filename=f"../{synthetic.TRUTH_DIR}/ground_truth.csv"
        )


def test_the_demo_marker_file_says_what_the_directory_is(
    workspace: Path, demo_run: runner.DemoRun
) -> None:
    marker = (
        scenarios.default_demo_root(workspace) / scenarios.DEMO_MARKER
    ).read_text(encoding="utf-8")
    assert "NOT a measurement" in marker
    assert str(scenarios.DEMO_SEED) in marker


def test_the_origin_roster_assumptions_are_checked(demo_run: runner.DemoRun) -> None:
    """Scenario C separating depends on a property of a generated roster."""
    nodes = synthetic.build_nodes(scenarios.network_config())
    scenarios.assert_origin_roster(nodes)  # must not raise
    broken = nodes.copy()
    broken.loc[scenarios.ORIGIN_RIGHT, "region"] = broken.loc[
        scenarios.ORIGIN_LEFT, "region"
    ]
    with pytest.raises(RuntimeError, match="scenario C"):
        scenarios.assert_origin_roster(broken)


# ---- the post-hoc audit ------------------------------------------------


def test_the_audit_reports_one_contradiction_per_component(
    demo_run: runner.DemoRun,
) -> None:
    """Not one per member. Counting members would repeat the 166-edge trap."""
    contradictions = [
        r.contradiction for r in demo_run.results if r.contradiction is not None
    ]
    assert len(contradictions) == 1
    assert len({c.root for c in contradictions}) == 1


def test_the_audit_uses_the_production_rule_on_both_sides(
    demo_run: runner.DemoRun,
) -> None:
    config = demo_run.config
    contradiction = next(
        r.contradiction for r in demo_run.results if r.contradiction is not None
    )
    evidence = contradiction.evidence
    assert evidence.n_a >= config.min_pooled_observations
    assert evidence.n_b >= config.min_pooled_observations
    assert evidence.p_value < config.alpha
    assert evidence.effect >= config.min_effect


def test_pooling_an_empty_member_set_is_the_empty_statistic(
    demo_run: runner.DemoRun,
) -> None:
    assert audit._pool(demo_run.oracle, []).count == 0
