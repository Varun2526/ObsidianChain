"""The five demonstration scenarios, and the seeded fixture that produces them.

WHAT THIS IS
------------
Five hand-designed situations, each built from a seed, each exercising one
state of the constraint engine end to end:

===  ==========================================================  ============
key  situation                                                   outcome
===  ==========================================================  ============
A    blockchain evidence proposes a merge                        CANDIDATE
B    network evidence present but insufficient                   MERGED
C    network evidence strong and separating                      BLOCKED
D    contradiction appears after transitive clustering           CONTESTED
E    no usable network evidence                                  ABSTAINED
===  ==========================================================  ============

WHAT THIS IS NOT
----------------
It is not a measurement, and no number here says anything about Bitcoin. The
fixture was written by the same people as the engine and contains exactly the
structure the engine looks for. Its only claim is "the mechanism is wired up
and behaves as described when the situation arises".

On the frozen Elliptic++ dataset the situation does not arise: 40 of 253,429
proposed unions reached the pooled minimum and none separated. That is a
finding, not a bug, and the demonstration must never be quoted as evidence
against it.

THE THRESHOLD IS NOT LOWERED
----------------------------
``SeparationConfig()`` is used exactly as production defines it - 25 pooled
observations per side, alpha 1e-4, effect floor 0.05 - and sigma stays at the
Decker & Wattenhofer 1.1506. What the fixture changes is the *evidence*, not
the rule: scenario C accumulates 160 usable announcements per side because
that is what it takes for a genuine origin difference to clear a 1e-4 gate at
this sigma. Measured, not guessed - at 30 and 60 pooled the same two origins
return NOT_SEPARATED. Lowering the bar to meet thin evidence would demonstrate
a different rule, which is the one thing this fixture must not do.

That number is also the honest reason the frozen dataset never fires: its
median component pools zero usable observations, and its best pools 40.

HOW EACH SCENARIO IS CONSTRUCTED
--------------------------------
One chain, five disjoint sub-graphs, one engine run. Evidence is carried by
**single-input transactions**, which contribute an announcement to exactly one
address and produce no co-spend edge; merges are proposed by separate
**multi-input transactions**. Separating the two lets each address's pooled
count be set exactly, without the co-spend structure and the evidence volume
being tangled together.

``A`` two addresses, three co-spend transactions, and **no announcement
    records at all**. The chain proposes a merge and the network layer has
    nothing to say about it - not "the evidence was unusable", but "this
    capture never saw these transactions". That is the candidate stage, and
    it is what every other scenario starts from.

``B`` both sides accumulate 160 usable announcements from the *same* origin.
    The rule looks, finds no difference beyond its noise, and the merge
    proceeds. Evidence present, insufficient to act on.

``C`` identical to B in every respect except that the two sides broadcast
    from *different* origins. The merge is refused. B and C differing in one
    variable is the point: the veto tracks the origin difference, not the
    volume of data.

``D`` the false-negative trap, and the reason ``CONTESTED`` exists. A hub
    address with 160 announcements from one origin is joined to eight arm
    addresses carrying 20 each from another, through a thin link address.
    Every merge decision has one side below the pooled minimum, so every one
    abstains and the merge goes through. Only once clustering has finished
    does the component hold two pools large enough to compare - and by then
    union-find has no split. The contradiction is recorded, not resolved.

``E`` 160 announcements per side, every one of them relayed by a known
    broadcaster, so every one is classified NO_EVIDENCE and discarded before
    any statistic is computed. The rule abstains. This is deliberately the
    same volume as B and C: abstention here is about the *quality* of the
    evidence, and a system that acted on it would be manufacturing a link
    out of a vector that carries no ordering information.

DETERMINISM
-----------
Everything derives from :data:`DEMO_SEED`. The chain is written by
:func:`build_chain`, the announcements by :mod:`obsidianchain.network.synthetic`
with a fixed ``origin_idx``, and the fixture manifest records a SHA-256 of the
observations so a rebuild can be shown to be byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.network import synthetic

#: Bump when the fixture layout or its generation semantics change.
DEMO_FIXTURE_VERSION = "1.0.0"

#: Everything the demo produces derives from this. NTRO PS 26146; the value
#: is arbitrary, its being fixed is not.
DEMO_SEED = 26146

#: The one directory name a demo root may end in. See
#: :func:`assert_demo_namespace` - this is what keeps the demonstration out
#: of ``data/processed/``.
DEMO_DIRNAME = "demo"

#: Written at the demo root so a stray directory is identifiable on sight.
DEMO_MARKER = "SYNTHETIC_DEMONSTRATION"

#: Usable announcements accumulated per side where the rule must be able to
#: answer. Chosen by measurement, not by taste. With sigma = 1.1506 the two
#: origins below return NOT_SEPARATED at 30 and at 60 pooled, and separate
#: from about 120. Scenario D needs more than C does: its comparison is a
#: single address against a *mixture* of the rest of its component, which
#: dilutes the contrast, and at 160 it landed at p = 1.2e-4 - on the wrong
#: side of a 1e-4 gate by a hair. 320 clears it by four orders of magnitude,
#: so the demonstration does not turn on a coin flip.
#:
#: The production minimum of 25 is untouched throughout. This is the evidence
#: rising to meet the rule, never the rule coming down to meet the evidence.
EVIDENCE_TX = 320

#: Scenario D's arms. Each carries fewer than the pooled minimum, so no merge
#: decision involving one is ever answerable, yet together they out-pool the
#: hub. That gap is the whole trap.
D_ARM_COUNT = 12
D_ARM_TX = 20
D_LINK_TX = 4

#: Scenario A's co-spend transactions. Three rather than one so the report
#: can show a re-proposed boundary being recognised as one decision.
A_MERGE_TX = 3

#: Origin indices into the roster built from ``NetworkConfig(seed=DEMO_SEED)``.
#: Picked, not invented: 002 and 007 are the pair whose per-observer
#: proximity profiles differ most (eu-west against latam, no shared ASN), and
#: 003 is a known broadcaster. Selection is asserted in
#: :func:`assert_origin_roster` so a generator change cannot silently turn
#: scenario C into scenario B.
ORIGIN_LEFT = 2
ORIGIN_RIGHT = 7
ORIGIN_BROADCASTER = 3


class DemoNamespaceError(RuntimeError):
    """Raised when a demo run is pointed outside its own namespace."""


class DemoOutcome(str, Enum):
    """What the demonstration reports for one scenario.

    Five names for five distinguishable engine states. ``CANDIDATE`` and
    ``ABSTAINED`` both leave the merge in place and both return the engine
    verdict ``NO_EVIDENCE``, and separating them is deliberate: one means the
    capture never saw the transactions, the other means it saw them and they
    carried no ordering information. An operator reads those differently -
    the first is a coverage problem, the second is not.
    """

    CANDIDATE = "CANDIDATE"
    """Co-spend proposed the merge; no announcement record covers it."""

    MERGED = "MERGED"
    """The rule looked, found no separation, and let the merge proceed."""

    BLOCKED = "BLOCKED"
    """The rule refused the merge. The cannot-link fired."""

    CONTESTED = "CONTESTED"
    """Evidence says separate, but the pair is already merged transitively."""

    ABSTAINED = "ABSTAINED"
    """Observations existed and none of them were usable."""


@dataclass(frozen=True)
class ScenarioSpec:
    """One scenario: how to build it, and what it must produce.

    ``expected_outcome`` and ``expected_verdict`` are checked by the runner
    against what the engine actually did. A demonstration that quietly
    displays whatever happened would be worth nothing to a judge, so a
    scenario that stops behaving as described fails the run instead.
    """

    key: str
    title: str
    situation: str
    chain_story: str
    network_story: str
    expected_outcome: DemoOutcome
    expected_verdict: str
    reads: str
    """One line on what the viewer should take away."""


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        key="A",
        title="Blockchain evidence proposes a merge",
        situation="A co-spend transaction spends two addresses together.",
        chain_story=(
            f"{A_MERGE_TX} transactions spend both addresses as inputs. "
            "Common-input-ownership proposes the merge."
        ),
        network_story=(
            "No announcement record covers these transactions. The capture "
            "never saw them, so the network layer is not consulted at all."
        ),
        expected_outcome=DemoOutcome.CANDIDATE,
        expected_verdict="NO_EVIDENCE",
        reads=(
            "This is where every decision starts. Plain co-spend clustering "
            "stops here and merges - permanently."
        ),
    ),
    ScenarioSpec(
        key="B",
        title="Network evidence present but insufficient",
        situation="Both sides broadcast from the same origin.",
        chain_story="One co-spend transaction proposes the merge.",
        network_story=(
            f"{EVIDENCE_TX} usable announcements pooled on each side, well "
            "past the minimum of 25. The two arrival centroids differ by "
            "less than their own standard error."
        ),
        expected_outcome=DemoOutcome.MERGED,
        expected_verdict="NOT_SEPARATED",
        reads=(
            "Having enough evidence is not the same as having evidence that "
            "says something. The merge proceeds."
        ),
    ),
    ScenarioSpec(
        key="C",
        title="Network evidence strong and separating",
        situation="The two sides broadcast from different origins.",
        chain_story="One co-spend transaction proposes the merge.",
        network_story=(
            f"{EVIDENCE_TX} usable announcements pooled on each side - the "
            "same volume as B. The centroids differ by more than the "
            "measurement precision allows, past both gates."
        ),
        expected_outcome=DemoOutcome.BLOCKED,
        expected_verdict="SEPARATED",
        reads=(
            "The differentiator. A wrong co-spend merge is prevented rather "
            "than cleaned up afterwards, because union-find has no undo."
        ),
    ),
    ScenarioSpec(
        key="D",
        title="Contradiction appears after transitive clustering",
        situation=(
            f"A hub and {D_ARM_COUNT} arms, joined one at a time through a "
            f"thin link."
        ),
        chain_story=(
            f"{D_ARM_COUNT + 1} co-spend transactions build one component. "
            "Every single decision has one side below the pooled minimum."
        ),
        network_story=(
            f"The hub pools {EVIDENCE_TX} announcements from one origin; the "
            f"arms pool {D_ARM_COUNT * D_ARM_TX} from another, but never "
            f"more than {D_ARM_TX} at a time. The two pools only exist as "
            "pools once clustering has finished."
        ),
        expected_outcome=DemoOutcome.CONTESTED,
        expected_verdict="SEPARATED",
        reads=(
            "The veto acts before a union; this contradiction only becomes "
            "visible after one. It is recorded, not resolved - unmerging "
            "would mean discarding a cryptographically-backed co-spend edge."
        ),
    ),
    ScenarioSpec(
        key="E",
        title="No usable network evidence",
        situation="Every announcement is relayed by a known broadcaster.",
        chain_story="One co-spend transaction proposes the merge.",
        network_story=(
            f"{EVIDENCE_TX} announcements per side exist and every one is "
            "discarded before any statistic is computed: a broadcaster is "
            "heard directly by every observer, so the arrival order carries "
            "no information about who sent it."
        ),
        expected_outcome=DemoOutcome.ABSTAINED,
        expected_verdict="NO_EVIDENCE",
        reads=(
            "Abstention, not weak evidence. One Electrum server broadcasts "
            "for tens of thousands of users; a link drawn from that would be "
            "manufactured."
        ),
    ),
)

SCENARIOS_BY_KEY = {spec.key: spec for spec in SCENARIOS}


# ---- namespace guard ---------------------------------------------------


def assert_demo_namespace(root) -> Path:
    """Refuse any root that is not a demonstration namespace.

    The failure this prevents is not malice, it is a convenient default: a
    ``--data-root`` left pointing at ``data/`` would have the demo generator
    write announcement records straight over the frozen production dataset,
    and the frozen hash is the anchor for every Phase 2 and Phase 3 number in
    the project. So the check is structural rather than a convention - the
    final path component must be ``demo``, and no production directory name
    may appear anywhere in the path.
    """
    path = Path(root).resolve()
    if path.name != DEMO_DIRNAME:
        raise DemoNamespaceError(
            f"{path} is not a demonstration namespace: the demo root's final "
            f"path component must be {DEMO_DIRNAME!r}. Demonstration fixtures "
            f"are never written alongside production data."
        )
    forbidden = {
        synthetic.OBSERVATIONS_DIR,
        synthetic.TRUTH_DIR,
        synthetic.WORLDS_DIR,
        synthetic.WORLDS_TRUTH_DIR,
    }
    clash = forbidden & set(path.parts)
    if clash:
        raise DemoNamespaceError(
            f"{path} sits inside {sorted(clash)}, which holds production or "
            f"controlled-world data. Pick a root outside it."
        )
    return path


def default_demo_root(data_root) -> Path:
    """``<data_root>/demo`` - the one place the demonstration writes."""
    return Path(data_root) / DEMO_DIRNAME


# ---- the chain ---------------------------------------------------------


def _address(key: str, index: int, role: str = "") -> str:
    """A self-identifying address label.

    Deliberately not shaped like a Bitcoin address. These strings end up in
    JSON, in the HTML report and in any terminal output a judge might
    screenshot, and every one of them should say what it is without a caption.
    """
    tail = f"{role}-{index:02d}" if role else f"{index:02d}"
    return f"DEMO-{key}-{tail}"


@dataclass
class DemoChain:
    """The synthetic chain, plus the origin each transaction broadcast from.

    ``origin_of_txid`` is generator input, not observable. It is written to
    the demo's own truth directory and never crosses
    :mod:`obsidianchain.network.boundary`.
    """

    addr_tx: pd.DataFrame
    wallets: pd.DataFrame
    origin_of_txid: dict[int, int]
    scenario_addresses: dict[str, tuple[str, ...]]
    scenario_txids: dict[str, tuple[int, ...]]
    merge_txids: dict[str, tuple[int, ...]]

    @property
    def observed_txids(self) -> np.ndarray:
        """Transactions the capture saw, sorted. Scenario A's are absent."""
        return np.sort(np.array(sorted(self.origin_of_txid), dtype=np.int64))


def build_chain() -> DemoChain:
    """Lay out all five sub-graphs in one deterministic chain.

    Transaction ids are assigned in blocks, in scenario order, and rows are
    written in that order too. That matters in exactly one place: co-spend
    edges are replayed in transaction order, and scenario D depends on its
    nine merge transactions arriving hub-first so that each decision has a
    thin side. Everywhere else the order is irrelevant and fixed only so the
    fixture is reproducible.
    """
    rows_addr: list[str] = []
    rows_tx: list[int] = []
    addresses: list[str] = []
    origin_of_txid: dict[int, int] = {}
    scenario_addresses: dict[str, tuple[str, ...]] = {}
    scenario_txids: dict[str, list[int]] = {}
    merge_txids: dict[str, list[int]] = {}

    txid = 1

    def spend(tx: int, inputs: list[str]) -> None:
        rows_addr.extend(inputs)
        rows_tx.extend([tx] * len(inputs))

    # ---- A: proposed, never observed ---------------------------------
    a0, a1 = _address("A", 0), _address("A", 1)
    addresses += [a0, a1]
    scenario_addresses["A"] = (a0, a1)
    scenario_txids["A"], merge_txids["A"] = [], []
    for _ in range(A_MERGE_TX):
        spend(txid, [a0, a1])
        scenario_txids["A"].append(txid)
        merge_txids["A"].append(txid)
        # No entry in origin_of_txid: this transaction is never announced.
        txid += 1

    # ---- B and C: same shape, one variable apart ----------------------
    for key, right_origin in (("B", ORIGIN_LEFT), ("C", ORIGIN_RIGHT)):
        left, right = _address(key, 0), _address(key, 1)
        addresses += [left, right]
        scenario_addresses[key] = (left, right)
        scenario_txids[key], merge_txids[key] = [], []
        for address, origin in ((left, ORIGIN_LEFT), (right, right_origin)):
            for _ in range(EVIDENCE_TX):
                spend(txid, [address])  # single input: evidence, not an edge
                origin_of_txid[txid] = origin
                scenario_txids[key].append(txid)
                txid += 1
        spend(txid, [left, right])
        origin_of_txid[txid] = ORIGIN_LEFT
        scenario_txids[key].append(txid)
        merge_txids[key].append(txid)
        txid += 1

    # ---- D: the pools only exist once clustering has finished ---------
    hub = _address("D", 0, "hub")
    link = _address("D", 0, "link")
    arms = [_address("D", i, "arm") for i in range(D_ARM_COUNT)]
    addresses += [hub, link, *arms]
    scenario_addresses["D"] = (hub, link, *arms)
    scenario_txids["D"], merge_txids["D"] = [], []
    for address, origin, count in (
        (hub, ORIGIN_LEFT, EVIDENCE_TX),
        (link, ORIGIN_RIGHT, D_LINK_TX),
        *[(arm, ORIGIN_RIGHT, D_ARM_TX) for arm in arms],
    ):
        for _ in range(count):
            spend(txid, [address])
            origin_of_txid[txid] = origin
            scenario_txids["D"].append(txid)
            txid += 1
    # Hub first, then one arm at a time. Every decision now has a side below
    # the pooled minimum, which is what makes the contradiction unreachable
    # until the component is complete.
    for partner in (hub, *arms):
        spend(txid, [link, partner])
        origin_of_txid[txid] = ORIGIN_RIGHT
        scenario_txids["D"].append(txid)
        merge_txids["D"].append(txid)
        txid += 1

    # ---- E: seen, and unusable ---------------------------------------
    e0, e1 = _address("E", 0), _address("E", 1)
    addresses += [e0, e1]
    scenario_addresses["E"] = (e0, e1)
    scenario_txids["E"], merge_txids["E"] = [], []
    for address in (e0, e1):
        for _ in range(EVIDENCE_TX):
            spend(txid, [address])
            origin_of_txid[txid] = ORIGIN_BROADCASTER
            scenario_txids["E"].append(txid)
            txid += 1
    spend(txid, [e0, e1])
    origin_of_txid[txid] = ORIGIN_BROADCASTER
    scenario_txids["E"].append(txid)
    merge_txids["E"].append(txid)
    txid += 1

    return DemoChain(
        addr_tx=pd.DataFrame({"input_address": rows_addr, "txId": rows_tx}),
        wallets=pd.DataFrame({"address": addresses, "class": 3}),
        origin_of_txid=origin_of_txid,
        scenario_addresses=scenario_addresses,
        scenario_txids={k: tuple(v) for k, v in scenario_txids.items()},
        merge_txids={k: tuple(v) for k, v in merge_txids.items()},
    )


# ---- the network layer -------------------------------------------------


def network_config() -> synthetic.NetworkConfig:
    """The frozen propagation model, on the demonstration seed.

    Only the seed differs from :data:`synthetic.FROZEN_SEPTEMBER_2026`. sigma,
    the observer count, the clock jitter and bias, the broadcaster fraction
    and the missing-observation rate are all held at their production values,
    because a demonstration that also changed the physics would not be
    demonstrating this system.
    """
    frozen = synthetic.FROZEN_SEPTEMBER_2026
    return synthetic.NetworkConfig(
        n_observers=frozen.n_observers,
        n_origins=frozen.n_origins,
        broadcaster_fraction=frozen.broadcaster_fraction,
        peers_per_observer=frozen.peers_per_observer,
        clock_jitter_sd_ms=frozen.clock_jitter_sd_ms,
        clock_bias_range_ms=frozen.clock_bias_range_ms,
        missing_observation_rate=frozen.missing_observation_rate,
        seed=DEMO_SEED,
        propagation=frozen.propagation,
    )


def assert_origin_roster(nodes: pd.DataFrame) -> None:
    """Check the three chosen origins still have the properties assumed.

    Scenario C separates from B only because ORIGIN_LEFT and ORIGIN_RIGHT sit
    in different regions, and E abstains only because ORIGIN_BROADCASTER is a
    known broadcaster. Both are properties of a generated roster. If the
    generator changes and these stop holding, C quietly becomes a second copy
    of B and the demonstration shows a mechanism that is not running - which
    is the exact failure mode this project keeps finding. Fail instead.
    """
    broadcaster = nodes["is_known_broadcaster"].to_numpy()
    region = nodes["region"].to_numpy()
    for index in (ORIGIN_LEFT, ORIGIN_RIGHT):
        if broadcaster[index]:
            raise RuntimeError(
                f"demo origin {index} is a known broadcaster; its "
                f"announcements would be discarded and scenarios B/C/D would "
                f"silently abstain."
            )
    if region[ORIGIN_LEFT] == region[ORIGIN_RIGHT]:
        raise RuntimeError(
            f"demo origins {ORIGIN_LEFT} and {ORIGIN_RIGHT} now share region "
            f"{region[ORIGIN_LEFT]!r}; scenario C would stop separating and "
            f"would duplicate scenario B."
        )
    if not broadcaster[ORIGIN_BROADCASTER]:
        raise RuntimeError(
            f"demo origin {ORIGIN_BROADCASTER} is no longer a known "
            f"broadcaster; scenario E would stop abstaining."
        )


def generate_observations(chain: DemoChain):
    """Announcement records for every transaction the capture saw.

    Scenario A's transactions are simply not passed in, which is how "the
    capture never saw them" is expressed - there is no flag for it and there
    should not be, because a real capture has no record of what it missed.
    """
    config = network_config()
    nodes = synthetic.build_nodes(config)
    observers = synthetic.build_observers(config, nodes)
    assert_origin_roster(nodes)

    txids = chain.observed_txids
    origin_idx = np.array(
        [chain.origin_of_txid[int(t)] for t in txids], dtype=np.int64
    )
    observations, nodes, observers, ground_truth = synthetic.generate(
        txids, config, nodes=nodes, observers=observers, origin_idx=origin_idx
    )
    return observations, nodes, observers, ground_truth, config


# ---- writing -----------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fixture(demo_root, chain: DemoChain | None = None) -> dict:
    """Write the whole demonstration fixture under ``demo_root``.

    Layout mirrors a real data root - ``raw/`` for the chain, ``processed/``
    for what inference may read, and a quarantined truth directory - so the
    ordinary loaders and :mod:`obsidianchain.network.boundary` work unchanged.
    Nothing in the engine needs to know this fixture exists.

    Returns the manifest.
    """
    root = assert_demo_namespace(demo_root)
    chain = chain or build_chain()

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    chain.addr_tx.to_csv(raw / "AddrTx_edgelist.csv", index=False)
    chain.wallets.to_csv(raw / "wallets_classes.csv", index=False)

    observations, nodes, observers, ground_truth, config = generate_observations(
        chain
    )

    processed = root / "processed"
    obs_dir = processed / synthetic.OBSERVATIONS_DIR
    truth_dir = processed / synthetic.TRUTH_DIR
    obs_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    obs_path = obs_dir / "observations.parquet"
    observations.to_parquet(obs_path, index=False)
    observers.loc[:, ["observer_id", "asn", "region"]].to_csv(
        obs_dir / "observers.csv", index=False
    )
    nodes.loc[nodes["is_known_broadcaster"], ["ip"]].to_csv(
        obs_dir / "broadcaster_ips.csv", index=False
    )

    ground_truth.to_csv(truth_dir / "ground_truth.csv", index=False)
    nodes.to_csv(truth_dir / "origin_nodes.csv", index=False)
    (truth_dir / "README.txt").write_text(
        "DEMONSTRATION GROUND TRUTH - EVALUATION ONLY\n"
        "Synthetic. Never loaded by the inference stage; "
        "network.boundary refuses this directory.\n",
        encoding="utf-8",
    )

    manifest = {
        "fixture": "demo",
        "demo": True,
        "provenance": "SYNTHETIC_DEMONSTRATION",
        # The shared vocabulary from obsidianchain.provenance, so a demo
        # artifact is classifiable alongside every other durable output
        # rather than only by its own local flag.
        "provenance_type": "DEMO",
        "not_a_measurement": True,
        "demo_fixture_version": DEMO_FIXTURE_VERSION,
        "seed": DEMO_SEED,
        "dataset_sha256": _sha256(obs_path),
        "record_count": int(len(observations)),
        "transaction_count": int(observations["txid"].nunique()),
        "chain_transaction_count": int(chain.addr_tx["txId"].nunique()),
        "unobserved_transaction_count": int(
            chain.addr_tx["txId"].nunique() - observations["txid"].nunique()
        ),
        "address_count": int(len(chain.wallets)),
        "observer_count": int(observations["observer_id"].nunique()),
        "record_columns": list(synthetic.RECORD_COLUMNS),
        "sigma": synthetic.DECKER_WATTENHOFER_SIGMA,
        "production_min_pooled": 25,
        "evidence_transactions_per_side": EVIDENCE_TX,
        "scenarios": [spec.key for spec in SCENARIOS],
        "configuration": config.describe(),
    }
    (obs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / DEMO_MARKER).write_text(
        "Synthetic demonstration fixture.\n"
        "Everything under this directory is generated, deterministic from "
        f"seed {DEMO_SEED}, and is NOT a measurement.\n"
        "The frozen production dataset lives elsewhere and is untouched by "
        "anything here.\n",
        encoding="utf-8",
    )
    return manifest


def fixture_exists(demo_root) -> bool:
    """Whether a demo fixture has already been written under ``demo_root``."""
    root = Path(demo_root)
    return (
        (root / "raw" / "AddrTx_edgelist.csv").is_file()
        and (
            root / "processed" / synthetic.OBSERVATIONS_DIR
            / "observations.parquet"
        ).is_file()
    )
