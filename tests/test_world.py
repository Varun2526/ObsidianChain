"""The coherent TXID-correlated synthetic world.

Three things this file is really checking.

That the world is COHERENT: the chain layer and the network layer describe
the same transactions, because both came from the same ones. If that fails,
every experiment built on the world is measuring a coincidence.

That TRUTH IS ISOLATED: no observable file carries entity identity, origin
identity or the behaviour that produced a transaction. A leak here would make
every metric meaningless while leaving every test green, which is the worst
failure mode available.

That the DETECTORS SEE WHAT IS THERE, and only that: the mixing detector must
find every generated mixing round and none of the five benign shapes chosen
to resemble one, and PEEL-1 must find the generated chains.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from obsidianchain.features import mixing, peel
from obsidianchain.features.incidence import load_incidence
from obsidianchain.world import behaviours as bx
from obsidianchain.world import overlap as overlap_mod
from obsidianchain.world.generate import (
    TRUTH_DIR,
    WorldConfig,
    build_world,
    write_world,
)

#: Columns that exist only in truth. Their appearance in an observable file
#: is the leak this suite exists to catch.
TRUTH_COLUMNS = (
    "true_entity", "true_origin_id", "scenario", "entity_id", "origin_id",
    "ground_truth", "scenario_label", "band",
)

OBSERVABLE_RAW = (
    "AddrTx_edgelist.csv", "TxAddr_edgelist.csv", "txs_features.csv",
)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("world")
    write_world(root, WorldConfig())
    return root


@pytest.fixture(scope="module")
def truth(world) -> dict:
    return {
        name: pd.read_csv(world / TRUTH_DIR / f"{name}.csv")
        for name in ("entities", "addresses", "transactions")
    }


@pytest.fixture(scope="module")
def incidence(world):
    return load_incidence(world)


# ---- the world is internally consistent --------------------------------


def test_the_existing_loader_reads_the_world_unmodified(incidence) -> None:
    """The strongest form of "the pipeline runs on it": the same loader."""
    assert incidence.n_addresses > 0
    assert len(incidence.transactions) > 0
    assert len(incidence.frame) > 0


def test_every_txid_is_unique(world) -> None:
    features = pd.read_csv(world / "raw" / "txs_features.csv")
    assert features["txId"].is_unique


def test_every_address_belongs_to_exactly_one_entity(truth) -> None:
    assert truth["addresses"]["address"].is_unique


def test_every_referenced_address_exists(world, truth) -> None:
    known = set(truth["addresses"]["address"])
    spends = pd.read_csv(world / "raw" / "AddrTx_edgelist.csv")
    receives = pd.read_csv(world / "raw" / "TxAddr_edgelist.csv")
    assert set(spends["input_address"]) <= known
    assert set(receives["output_address"]) <= known


def test_every_edge_references_a_real_transaction(world) -> None:
    features = pd.read_csv(world / "raw" / "txs_features.csv")
    known = set(features["txId"])
    for name in ("AddrTx_edgelist.csv", "TxAddr_edgelist.csv"):
        edges = pd.read_csv(world / "raw" / name)
        assert set(edges["txId"]) <= known


def test_transaction_values_are_internally_consistent(world) -> None:
    """Outputs never exceed inputs, and the summary matches the counts."""
    features = pd.read_csv(world / "raw" / "txs_features.csv")
    assert (features["out_BTC_total"] <= features["in_BTC_total"] + 1e-9).all()
    assert (features["in_BTC_min"] <= features["in_BTC_max"]).all()
    assert (features["out_BTC_min"] <= features["out_BTC_max"]).all()
    assert (features["num_input_addresses"] > 0).all()
    assert (features["num_output_addresses"] > 0).all()
    assert (features["fees"] >= 0).all()


def test_the_edge_counts_match_the_declared_cardinalities(world) -> None:
    features = pd.read_csv(world / "raw" / "txs_features.csv").set_index("txId")
    spends = pd.read_csv(world / "raw" / "AddrTx_edgelist.csv")
    counted = spends.groupby("txId").size()
    declared = features.loc[counted.index, "num_input_addresses"]
    assert (counted.to_numpy() == declared.to_numpy()).all()


def test_every_timestep_is_inside_the_production_split_range(world) -> None:
    features = pd.read_csv(world / "raw" / "txs_features.csv")
    assert features["Time step"].between(1, 49).all()


# ---- the two layers share transactions ---------------------------------


def test_the_network_layer_describes_the_chain_layers_transactions(
    world, incidence
) -> None:
    """The whole point of the world: one txid space, two layers."""
    observations = pd.read_parquet(
        world / "processed" / "network" / "observations.parquet"
    )
    chain_txids = set(incidence.transactions.index.astype(str))
    network_txids = set(observations["txid"].astype(str))

    assert network_txids, "the network layer observed nothing"
    assert network_txids <= chain_txids, (
        "the network layer announced transactions the chain layer never "
        "produced; the two layers do not share a txid space"
    )
    # And the correlation is near-total in the un-thinned world.
    assert len(network_txids) / len(chain_txids) > 0.9


def test_network_observations_carry_only_observable_fields(world) -> None:
    observations = pd.read_parquet(
        world / "processed" / "network" / "observations.parquet"
    )
    assert set(observations.columns) == {
        "txid", "observer_id", "peer_ip", "peer_port", "peer_asn",
        "timestamp_ms",
    }


# ---- truth isolation ---------------------------------------------------


@pytest.mark.parametrize("name", OBSERVABLE_RAW)
def test_no_observable_chain_file_carries_a_truth_column(world, name) -> None:
    columns = set(pd.read_csv(world / "raw" / name, nrows=1).columns)
    leaked = columns & set(TRUTH_COLUMNS)
    assert not leaked, f"{name} leaks {leaked}"


def test_the_network_observations_carry_no_truth_column(world) -> None:
    observations = pd.read_parquet(
        world / "processed" / "network" / "observations.parquet"
    )
    leaked = set(observations.columns) & set(TRUTH_COLUMNS)
    assert not leaked, f"observations leak {leaked}"


def test_no_observable_value_encodes_an_entity_name(world, truth) -> None:
    """A subtler leak: an identifier that CONTAINS the entity name.

    Column names are the obvious channel and the easy one to check. An
    address string like ``ADDR-E-tr-PEELING-0-3`` would pass every column
    check above and hand the entity to any model that hashed it.
    """
    entities = set(truth["entities"]["true_entity"])
    addresses = pd.read_csv(world / "raw" / "AddrTx_edgelist.csv")
    sample = addresses["input_address"].astype(str).head(200)
    for entity in list(entities)[:20]:
        assert not sample.str.contains(entity, regex=False).any(), (
            f"an observable address embeds the entity name {entity!r}"
        )


def test_the_truth_directory_is_marked_for_evaluation_only(world) -> None:
    readme = (world / TRUTH_DIR / "README.txt").read_text(encoding="utf-8")
    assert "FOR_EVALUATION_ONLY" in readme


def test_truth_lives_outside_every_directory_the_pipeline_reads(world) -> None:
    assert (world / TRUTH_DIR).is_dir()
    assert not (world / "raw" / TRUTH_DIR).exists()
    assert not (world / "processed" / TRUTH_DIR).exists()


def test_the_label_file_is_the_only_bridge_and_carries_no_identity(world) -> None:
    """Labels are legitimate - production has them too - but nothing else."""
    labels = pd.read_csv(world / "raw" / "wallets_classes.csv")
    assert set(labels.columns) == {"address", "class"}
    assert set(labels["class"]) <= {1, 2}


# ---- determinism -------------------------------------------------------


def test_the_same_seed_produces_the_same_world() -> None:
    first, first_truth = build_world(WorldConfig())
    second, second_truth = build_world(WorldConfig())
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    for key in first_truth:
        pd.testing.assert_frame_equal(first_truth[key], second_truth[key])


def test_a_different_seed_produces_a_different_world() -> None:
    a, _ = build_world(WorldConfig())
    b, _ = build_world(WorldConfig(seed=999))
    assert not a["tx_features"].equals(b["tx_features"])


def test_the_config_digest_changes_with_the_config() -> None:
    assert WorldConfig().digest() != WorldConfig(seed=1).digest()


# ---- provenance --------------------------------------------------------


def test_the_manifest_records_what_a_reader_needs(world) -> None:
    manifest = json.loads((world / "world_manifest.json").read_text())
    for key in ("generator_version", "seed", "config_sha256",
                "world_fingerprint", "created_at", "provenance_type",
                "behaviours", "positive_class"):
        assert manifest.get(key) is not None, f"manifest omits {key}"


def test_the_world_is_marked_synthetic_control(world) -> None:
    manifest = json.loads((world / "world_manifest.json").read_text())
    assert manifest["provenance_type"] == "SYNTHETIC_CONTROL"
    assert any("not of Bitcoin" in note for note in manifest["notes"])


def test_the_positive_class_is_declared_a_convention(world) -> None:
    """It must not read as a claim that a behaviour is unlawful."""
    manifest = json.loads((world / "world_manifest.json").read_text())
    meaning = manifest["positive_class_meaning"].lower()
    assert "convention" in meaning
    assert "not a claim" in meaning


def test_the_fingerprint_does_not_depend_on_the_clock(world) -> None:
    """A world regenerated from one seed is the same world."""
    manifest = json.loads((world / "world_manifest.json").read_text())
    from obsidianchain.world import generate as gen

    import hashlib
    expected = hashlib.sha256(
        (WorldConfig().digest() + gen.GENERATOR_VERSION).encode("utf-8")
    ).hexdigest()
    assert manifest["world_fingerprint"] == expected


# ---- the behaviours produce the shapes they claim ----------------------


@pytest.fixture(scope="module")
def scored(incidence, truth) -> pd.DataFrame:
    frame = mixing.score_transactions(incidence.transactions)
    frame = frame.assign(txid=incidence.transactions.index.to_numpy())
    return frame.merge(truth["transactions"][["txid", "scenario"]], on="txid")


def test_every_behaviour_produced_transactions(scored) -> None:
    present = set(scored["scenario"])
    assert present == set(bx.BEHAVIOURS), (
        f"missing: {set(bx.BEHAVIOURS) - present}"
    )


def test_every_generated_mixing_round_is_detected(scored) -> None:
    rounds = scored[scored["scenario"] == bx.MIXING_LIKE]
    assert len(rounds) > 0
    assert (rounds["mixing_class"] == mixing.MIXING_PATTERN).all()


def test_no_adversarial_behaviour_is_called_a_mixing_pattern(scored) -> None:
    """The headline false-positive guarantee, on generated data.

    These five shapes were chosen BECAUSE they resemble mixing. A detector
    that cannot separate them from a collaborative spend would bury an
    investigator.
    """
    adversarial = scored[scored["scenario"].isin(bx.ADVERSARIAL)]
    assert len(adversarial) > 0
    offenders = adversarial[
        adversarial["mixing_class"] == mixing.MIXING_PATTERN
    ]
    assert offenders.empty, (
        offenders[["txid", "scenario", "mixing_score"]].to_dict("records")
    )


@pytest.mark.parametrize(
    "behaviour,expected", sorted(bx.EXPECTED_MIXING_CLASS.items())
)
def test_a_behaviour_produces_its_expected_class(scored, behaviour, expected) -> None:
    rows = scored[scored["scenario"] == behaviour]
    assert len(rows) > 0
    assert (rows["mixing_class"] == expected).all(), (
        rows["mixing_class"].value_counts().to_dict()
    )


def test_peeling_produces_a_chain_the_detector_finds(incidence, truth) -> None:
    cutoff = incidence.last_timestep()
    chains = peel.build_chain_graph(incidence)
    m2 = peel.build(incidence, cutoff, chains=chains,
                    min_depth=peel.PRIMARY_MIN_DEPTH)
    m2 = m2.assign(address=incidence.addresses[m2.index.to_numpy()])
    joined = (
        m2.merge(truth["addresses"], on="address")
        .merge(truth["entities"][["true_entity", "scenario"]], on="true_entity")
    )
    peeling = joined[joined["scenario"] == bx.PEELING]
    assert peeling["in_chain"].sum() > 0
    assert peeling["chain_depth_max"].max() >= peel.PRIMARY_MIN_DEPTH


def test_a_flat_behaviour_produces_no_chain(incidence, truth) -> None:
    cutoff = incidence.last_timestep()
    chains = peel.build_chain_graph(incidence)
    m2 = peel.build(incidence, cutoff, chains=chains,
                    min_depth=peel.PRIMARY_MIN_DEPTH)
    m2 = m2.assign(address=incidence.addresses[m2.index.to_numpy()])
    joined = (
        m2.merge(truth["addresses"], on="address")
        .merge(truth["entities"][["true_entity", "scenario"]], on="true_entity")
    )
    for behaviour in (bx.CONSOLIDATION, bx.EXCHANGE_BATCH, bx.MIXING_LIKE):
        rows = joined[joined["scenario"] == behaviour]
        assert rows["in_chain"].sum() == 0, behaviour


def test_the_behaviour_repertoire_declares_its_own_limits() -> None:
    assert "not claims" in bx.MEANING
    assert set(bx.ADVERSARIAL) <= set(bx.BEHAVIOURS)
    assert set(bx.POSITIVE_CLASS) <= set(bx.BEHAVIOURS)


# ---- overlap / degradation ---------------------------------------------


@pytest.fixture(scope="module")
def sweep(world) -> dict:
    return overlap_mod.run(world, levels=(1.0, 0.5, 0.0))


def test_coverage_falls_as_overlap_falls(sweep) -> None:
    coverage = [level["coverage"] for level in sweep["levels"]]
    assert coverage == sorted(coverage, reverse=True), coverage
    assert coverage[0] > coverage[-1]


def test_abstention_is_the_complement_of_coverage(sweep) -> None:
    for level in sweep["levels"]:
        assert level["coverage"] + level["abstention"] == pytest.approx(1.0)


def test_no_overlap_reports_evidence_unavailable_not_zero_evidence(sweep) -> None:
    """The property that matters: missing evidence stays missing.

    At zero overlap the real M3 builder REFUSES rather than returning zeros,
    and the experiment records that refusal as a condition. A system that
    turned an empty observation set into a confident zero would be
    manufacturing a finding out of a collector outage.
    """
    zero = [lv for lv in sweep["levels"] if lv["requested_overlap"] == 0.0][0]
    assert zero["evidence_available"] is False
    assert zero["unavailable_reason"]
    assert zero["addresses_with_evidence"] == 0
    assert zero["abstention"] == 1.0


def test_full_overlap_still_leaves_most_addresses_abstaining(sweep) -> None:
    """Honest about the world's own limits.

    Even at 100% txid overlap most addresses do not reach the production
    pooling minimum, because a small world produces few observations per
    address. That is a property of this generator, and recording it stops
    the coverage figure from being read as a system capability.
    """
    full = [lv for lv in sweep["levels"] if lv["requested_overlap"] == 1.0][0]
    assert full["evidence_available"] is True
    assert 0.0 < full["coverage"] < 1.0


def test_the_sweep_is_deterministic(world) -> None:
    first = overlap_mod.run(world, levels=(0.5,))
    second = overlap_mod.run(world, levels=(0.5,))
    assert first["levels"] == second["levels"]


def test_thinning_removes_whole_transactions(world) -> None:
    """Not random rows: a missed transaction is missed entirely."""
    observations = pd.read_parquet(
        world / "processed" / "network" / "observations.parquet"
    )
    txids = observations["txid"].unique()
    thinned = overlap_mod._thin(observations, txids, 0.5, seed=7)
    kept = set(thinned["txid"])
    for txid in list(kept)[:10]:
        original = (observations["txid"] == txid).sum()
        remaining = (thinned["txid"] == txid).sum()
        assert original == remaining


def test_the_experiment_states_what_it_does_not_claim(sweep) -> None:
    assert "does not claim" in sweep["meaning"]
    # Both halves of the distinction must be stated: an abstention is not
    # evidence of separation AND not evidence of a link. Asserting only one
    # would let the other quietly disappear.
    abstention = sweep["abstention_meaning"].lower()
    assert "not evidence of separation" in abstention
    assert "not evidence of a link" in abstention
    assert sweep["provenance_type"] == "SYNTHETIC_CONTROL"


# ---- chain-only vs network-only vs fused -------------------------------


@pytest.fixture(scope="module")
def comparison(world) -> dict:
    return overlap_mod.layer_comparison(world)


def test_the_comparison_reports_all_three_layers(comparison) -> None:
    for layer in ("chain", "network", "fused"):
        assert layer in comparison


def test_the_chain_layer_counts_what_the_ledger_yields(comparison) -> None:
    chain = comparison["chain"]
    assert chain["addresses"] > 0
    assert chain["transactions"] > 0
    assert chain["cospend_components"] > 0
    assert chain["transactions_with_mixing_pattern"] > 0
    assert chain["addresses_in_a_peel_chain"] > 0


def test_false_merges_are_counted_against_known_truth(comparison) -> None:
    """Countable only because this world's entities are known.

    A component spanning two entities is a false merge. This is the metric
    the whole network-constraint idea exists to reduce, and it can be
    measured here and nowhere in production.
    """
    chain = comparison["chain"]
    assert "components_spanning_more_than_one_entity" in chain
    assert chain["components_spanning_more_than_one_entity"] >= 0
    assert (
        chain["components_spanning_more_than_one_entity"]
        <= chain["cospend_components"]
    )


def test_the_network_layer_reports_abstention_alongside_coverage(comparison) -> None:
    network = comparison["network"]
    assert network["available"] is True
    assert (
        network["addresses_with_evidence"] + network["addresses_abstaining"]
        == comparison["chain"]["addresses"]
    )


def test_the_comparison_names_what_it_does_not_measure(comparison) -> None:
    """Absent rather than estimated, and said to be absent."""
    text = comparison["not_measured_here"].lower()
    assert "false splits" in text
    assert "constraint precision" in text
    assert "not perform" in text or "does not" in text


def test_the_comparison_is_labelled_synthetic(comparison) -> None:
    assert comparison["provenance_type"] == "SYNTHETIC_CONTROL"
    assert any("not a measurement of Bitcoin" in n for n in comparison["notes"])


def test_the_comparison_denies_being_three_models(comparison) -> None:
    meaning = comparison["meaning"]
    assert "not a comparison of three models" in meaning
    assert "the network layer is not a model" in meaning


def test_the_fused_view_repeats_the_network_layers_limit(comparison) -> None:
    note = comparison["fused"]["note"]
    assert "never merges" in note
    assert "never attributes ownership" in note


def test_the_comparison_is_deterministic(world) -> None:
    assert overlap_mod.layer_comparison(world) == overlap_mod.layer_comparison(world)
