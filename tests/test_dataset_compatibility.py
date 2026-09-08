"""A chain and a network dataset must belong to the same experiment.

The failure this guards against is silent, not loud. ``build_oracle`` maps
observations onto the chain and drops whatever does not resolve, so pairing
the wrong two produces an oracle with no statistics, a 100% abstention rate,
and a report that reads as a clean negative result.

Measured on the real pair: 0 of 6,000 reach-stress addresses exist in
Elliptic++, and 57 of 60,000 txids collide - and those 57 are DIFFERENT
transactions that happen to share an integer id, so a statistic built from
them is meaningless rather than empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.eval import phase33
from obsidianchain.network import synthetic


def write_chain(root: Path, txids, addresses=None) -> None:
    raw = Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    addresses = addresses or [f"addr{i}" for i in range(len(txids))]
    pd.DataFrame({"input_address": addresses, "txId": list(txids)}).to_csv(
        raw / "AddrTx_edgelist.csv", index=False
    )
    pd.DataFrame({"address": sorted(set(addresses)), "class": 3}).to_csv(
        raw / "wallets_classes.csv", index=False
    )


def write_network(root: Path, txids, world: str = "A") -> None:
    directory = Path(root) / "processed" / synthetic.WORLDS_DIR / world
    directory.mkdir(parents=True, exist_ok=True)
    n = len(txids)
    pd.DataFrame(
        {
            "txid": list(txids),
            "observer_id": ["obs-00"] * n,
            "peer_ip": ["192.0.2.1"] * n,
            "peer_port": [8333] * n,
            "peer_asn": [64512] * n,
            "timestamp_ms": np.arange(n, dtype=float),
        }
    ).to_parquet(directory / "observations.parquet", index=False)


# ---- matching pairs are accepted --------------------------------------


def test_identical_chain_and_network_pass(tmp_path: Path) -> None:
    write_chain(tmp_path, range(1, 51))
    write_network(tmp_path, range(1, 51))
    assert phase33.assert_datasets_compatible(tmp_path, tmp_path, "A") == 1.0


def test_network_subset_of_the_chain_passes(tmp_path: Path) -> None:
    """A network dataset may cover only some of the chain's transactions."""
    write_chain(tmp_path, range(1, 101))
    write_network(tmp_path, range(1, 21))
    assert phase33.assert_datasets_compatible(tmp_path, tmp_path, "A") == 1.0


def test_separate_roots_pass_when_the_data_matches(tmp_path: Path) -> None:
    """Chain and network may live in different roots if they belong together."""
    chain, network = tmp_path / "chain", tmp_path / "network"
    write_chain(chain, range(1, 41))
    write_network(network, range(1, 41))
    assert phase33.assert_datasets_compatible(chain, network, "A") == 1.0


# ---- mismatched pairs are refused --------------------------------------


def test_disjoint_datasets_are_refused(tmp_path: Path) -> None:
    chain, network = tmp_path / "chain", tmp_path / "network"
    write_chain(chain, range(1, 101))
    write_network(network, range(100_000, 100_050))
    with pytest.raises(phase33.DatasetMismatchError, match="not the same experiment"):
        phase33.assert_datasets_compatible(chain, network, "A")


def test_coincidental_id_overlap_is_still_refused(tmp_path: Path) -> None:
    """The real case: 57 of 60,000 ids collided by accident.

    A handful of shared integers is worse than none - those are different
    transactions wearing the same number.
    """
    chain, network = tmp_path / "chain", tmp_path / "network"
    write_chain(chain, range(1, 1001))
    network_ids = list(range(990, 1000)) + list(range(50_000, 50_990))
    write_network(network, network_ids)
    with pytest.raises(phase33.DatasetMismatchError):
        phase33.assert_datasets_compatible(chain, network, "A")


def test_error_names_both_roots_and_the_remedy(tmp_path: Path) -> None:
    chain, network = tmp_path / "chain", tmp_path / "network"
    write_chain(chain, range(1, 101))
    write_network(network, range(500_000, 500_100))
    with pytest.raises(phase33.DatasetMismatchError) as caught:
        phase33.assert_datasets_compatible(chain, network, "A")
    message = str(caught.value)
    assert "chain" in message and "network dataset" in message
    assert "--chain-root" in message, "the error must say how to fix it"
    assert "silent" in message or "no evidence" in message


def test_empty_network_dataset_is_refused(tmp_path: Path) -> None:
    write_chain(tmp_path, range(1, 51))
    write_network(tmp_path, [])
    with pytest.raises(phase33.DatasetMismatchError, match="no transactions"):
        phase33.assert_datasets_compatible(tmp_path, tmp_path, "A")


@pytest.mark.parametrize("shared", [0, 1, 10, 49])
def test_overlap_below_the_floor_is_refused(tmp_path: Path, shared: int) -> None:
    chain, network = tmp_path / f"c{shared}", tmp_path / f"n{shared}"
    write_chain(chain, range(1, 101))
    network_ids = list(range(1, shared + 1)) + list(
        range(900_000, 900_000 + (100 - shared))
    )
    write_network(network, network_ids)
    with pytest.raises(phase33.DatasetMismatchError):
        phase33.assert_datasets_compatible(chain, network, "A")


def test_overlap_at_the_floor_is_accepted(tmp_path: Path) -> None:
    """The floor is a floor, not a trap: exactly at it must pass."""
    chain, network = tmp_path / "chain", tmp_path / "network"
    write_chain(chain, range(1, 101))
    network_ids = list(range(1, 51)) + list(range(900_000, 900_050))
    write_network(network, network_ids)
    resolved = phase33.assert_datasets_compatible(chain, network, "A")
    assert resolved == pytest.approx(phase33.MIN_TXID_OVERLAP)


# ---- the guard is wired into the runner -------------------------------


def test_run_regime_takes_explicit_roots() -> None:
    """Naming both roots is what makes a mismatch visible rather than implied."""
    import inspect

    parameters = inspect.signature(phase33.run_regime).parameters
    assert "chain_root" in parameters
    assert "network_root" in parameters
    assert parameters["network_root"].default is None, (
        "network_root must default to the chain root so A-E behaviour is "
        "unchanged when only --data-root is given"
    )


def test_build_oracle_refuses_a_mismatched_pair(tmp_path: Path) -> None:
    """THE guard, on the production fusion path rather than beside it.

    Behavioural, not a source-index check: the previous version of this test
    asserted that ``assert_datasets_compatible`` appeared textually before
    ``build_oracle`` inside ``run_regime``, which only ever protected the
    evaluation harness. ``run --mode fused``, ``fusion-summary`` and
    ``evidence-funnel`` all call ``build_oracle`` directly and were
    unguarded. The check now lives inside ``build_oracle``, so this asserts
    the outcome an operator would actually see.
    """
    from obsidianchain.io import elliptic
    from obsidianchain.network import separation

    write_chain(tmp_path, range(1, 101))
    write_network(tmp_path, range(500_000, 500_100))
    graph = elliptic.load_cospend_graph(tmp_path, keep_labels=True)

    with pytest.raises(separation.DatasetMismatchError) as caught:
        separation.build_oracle(
            graph,
            processed_root=tmp_path / "processed",
            data_root=tmp_path,
            world="A",
        )
    message = str(caught.value)
    assert "not the same experiment" in message
    assert "--chain-root" in message
    assert "silent" in message or "no evidence" in message


def test_the_mismatch_raises_instead_of_abstaining(tmp_path: Path) -> None:
    """No oracle at all, rather than an empty one reporting 100% abstention.

    This is the failure being prevented: an oracle with no statistics is
    indistinguishable in every report from a genuine negative result.
    """
    from obsidianchain.io import elliptic
    from obsidianchain.network import separation

    write_chain(tmp_path, range(1, 101))
    write_network(tmp_path, range(500_000, 500_100))
    graph = elliptic.load_cospend_graph(tmp_path, keep_labels=True)

    oracle = None
    try:
        oracle = separation.build_oracle(
            graph,
            processed_root=tmp_path / "processed",
            data_root=tmp_path,
            world="A",
        )
    except separation.DatasetMismatchError:
        pass
    assert oracle is None, (
        "an oracle was returned for an incompatible pair; every downstream "
        "report would read it as a clean 100% abstention"
    )


def test_a_compatible_pair_still_builds_an_oracle(tmp_path: Path) -> None:
    """The guard must not become a wall: a matching pair behaves as before."""
    from obsidianchain.io import elliptic
    from obsidianchain.network import separation

    write_chain(tmp_path, range(1, 101))
    write_network(tmp_path, range(1, 101))
    graph = elliptic.load_cospend_graph(tmp_path, keep_labels=True)

    oracle = separation.build_oracle(
        graph,
        processed_root=tmp_path / "processed",
        data_root=tmp_path,
        world="A",
    )
    assert oracle is not None


def test_run_regime_does_not_duplicate_the_guard() -> None:
    """One rule in one place. A second copy is a thing that can drift."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "obsidianchain" / "eval" / "phase33.py"
    ).read_text(encoding="utf-8")
    body = source[source.index("def run_regime"):source.index("def _score_against_truth")]
    executable = "\n".join(
        line.split("#", 1)[0] for line in body.split("\n")
    )
    assert "assert_datasets_compatible" not in executable, (
        "the guard belongs in build_oracle; a copy here can drift from it"
    )


# ---- the report keeps the two fixtures distinct ------------------------


def test_reach_stress_report_is_labelled_as_a_mechanism_test() -> None:
    from obsidianchain.network.separation import SeparationConfig

    outcomes = {"A": phase33.RegimeOutcome(regime="A", name="INDEPENDENT")}
    text = phase33.format_experiment(
        outcomes, SeparationConfig(), fixture="reach-stress",
        chain_description="6,000 addresses (synthetic reach-stress chain)",
    )
    assert "REACH-STRESS" in text
    assert "NOT AN ELLIPTIC++ RESULT" in text
    assert "569,513" in text, "must say the cluster count is not the baseline's"


def test_normal_world_report_is_not_labelled_reach_stress() -> None:
    from obsidianchain.network.separation import SeparationConfig

    outcomes = {"A": phase33.RegimeOutcome(regime="A", name="INDEPENDENT")}
    text = phase33.format_experiment(outcomes, SeparationConfig())
    assert "REACH-STRESS" not in text
    assert "Elliptic++" in text


def test_production_rule_unchanged_by_this_work() -> None:
    from obsidianchain.network.separation import SeparationConfig

    config = SeparationConfig()
    assert config.min_pooled_observations == 25
    assert config.alpha == 1e-4
    assert config.min_effect == 0.05
