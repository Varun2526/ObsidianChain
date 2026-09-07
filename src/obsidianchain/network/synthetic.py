"""Synthetic Bitcoin network-layer announcement generator.

===========================================================================
THIS DATA IS SYNTHETIC. READ THIS BEFORE QUOTING ANY RESULT DERIVED FROM IT.
===========================================================================

Every record this module produces is generated from a random model. No packet
was captured, no peer was contacted, no mainnet node was observed. The IP
addresses are drawn from RFC 5737 documentation ranges and belong to nobody.

In one sentence: this data is synthetic, any result derived from it
demonstrates the mechanism rather than validating the method, and real
validation requires mainnet capture plus a controlled ground-truth set.

What a result computed on this data can show:

    that the *mechanism* works end to end - that announcement timings can be
    turned into arrival vectors, that arrival vectors can be turned into
    constraints, and that those constraints change clustering output in the
    direction the design predicts.

What it CANNOT show:

    that the method works on Bitcoin. Nothing here validates the approach.
    The generator was written by the same people as the analysis, so it
    necessarily contains the structure the analysis looks for. A result on
    synthetic data of this kind is a demonstration, never evidence.

Real validation requires two things this phase has neither of:

    1. Mainnet capture - actual `inv`/`tx` announcements recorded from real
       peers at several vantage points, with real clock discipline.
    2. A controlled ground-truth set - transactions we broadcast ourselves
       from known origins, so that the correct answer is known independently
       of the inference being tested.

Until both exist, any accuracy figure computed against this generator is
measuring the generator, not Bitcoin. Say so plainly if asked.

---------------------------------------------------------------------------
Propagation parameters
---------------------------------------------------------------------------
The lognormal *shape* is taken from published measurement, not invented:

    Decker, C. and Wattenhofer, R. (2013), "Information Propagation in the
    Bitcoin Network", IEEE P2P 2013. Reports block propagation with a median
    of 6.5 s and a mean of 12.6 s.

    For a lognormal, median = exp(mu) and mean = exp(mu + sigma^2 / 2), so
        sigma = sqrt(2 * ln(mean / median)) = sqrt(2 * ln(12.6 / 6.5)) = 1.15

That sigma is the one parameter here with a citable provenance, and it is
what gives the delay distribution its heavy right tail.

The *scale* is NOT from that paper and is flagged accordingly. Decker and
Wattenhofer measured blocks (~1 MB in 2013); transactions are ~250 bytes and
propagate far faster, so reusing 6.5 s would be wrong by an order of
magnitude. ``DEFAULT_TX_MEDIAN_MS`` below is an assumption, not a
measurement. Replace it with a figure from your own October capture; it is
isolated in one constant precisely so that swap is a one-line change.

Further reading on why timing carries topology information at all:

    Neudecker, T., Andelfinger, P. and Hartenstein, H. (2016), "Timing
    Analysis for Inferring the Topology of the Bitcoin Peer-to-Peer
    Network", IEEE UIC/ATC/ScalCom.

---------------------------------------------------------------------------
Deliberate modelling choices
---------------------------------------------------------------------------
``peer_ip`` does not identify the origin for ordinary transactions. In a real
capture the peer that relays an announcement to you is your own neighbour,
not the transaction's origin, so here it is drawn from the observer's own
peer table. This is deliberate: if ``peer_ip`` leaked the origin, the next
phase could recover the answer by string comparison and would appear to work
without using timing at all.

Known broadcasters are the exception, and that is the point. Exchange and
wallet-server infrastructure connects to a large share of the network, so
every observer hears such a transaction directly from the broadcaster, at
almost the same instant. The resulting arrival vector is flat and its
``peer_ip`` is identical everywhere - which is exactly why these
transactions must yield NO constraint downstream. A flat vector carries no
ordering information, and treating it as if it did would manufacture
evidence.
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

#: Bump when the output layout or generation semantics change. Recorded in
#: the manifest so a Phase 3 result can be tied to the exact generator.
GENERATOR_VERSION = "2.0.0"

#: Directory names. The split is structural, not a convention: anything the
#: inference stage may read lives under OBSERVATIONS_DIR, anything it may not
#: lives under TRUTH_DIR, and the loader in network.boundary refuses to cross.
OBSERVATIONS_DIR = "network"
TRUTH_DIR = "network_truth"

# ---- propagation model ------------------------------------------------

#: Lognormal shape from Decker & Wattenhofer (2013); see module docstring.
DECKER_WATTENHOFER_SIGMA = float(np.sqrt(2 * np.log(12.6 / 6.5)))

#: ASSUMPTION, not a measurement. Median transaction propagation delay.
#: Replace from real capture in October.
DEFAULT_TX_MEDIAN_MS = 1_200.0

#: Known broadcasters are directly connected, so their announcements arrive
#: fast and with little spread. Also an assumption.
DEFAULT_BROADCAST_MEDIAN_MS = 45.0
DEFAULT_BROADCAST_SIGMA = 0.25

RECORD_COLUMNS = [
    "txid",
    "observer_id",
    "peer_ip",
    "peer_port",
    "peer_asn",
    "timestamp_ms",
]

REGIONS = ("na-east", "na-west", "eu-west", "eu-central", "apac", "latam")

#: RFC 5737 TEST-NET blocks. Reserved for documentation, routable by nobody.
_TEST_NETS = ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")


@dataclass(frozen=True)
class PropagationModel:
    """Lognormal delay parameters, in milliseconds."""

    median_ms: float = DEFAULT_TX_MEDIAN_MS
    sigma: float = DECKER_WATTENHOFER_SIGMA
    broadcast_median_ms: float = DEFAULT_BROADCAST_MEDIAN_MS
    broadcast_sigma: float = DEFAULT_BROADCAST_SIGMA

    #: Multipliers applied to the median when the observer is near the origin.
    #: These create the differential signal the next phase looks for.
    same_asn_factor: float = 0.45
    same_region_factor: float = 0.70

    @property
    def mean_ms(self) -> float:
        return float(self.median_ms * np.exp(self.sigma**2 / 2))


@dataclass(frozen=True)
class NetworkConfig:
    """Everything the generator needs. Seeded, so runs are reproducible."""

    n_observers: int = 8
    n_origins: int = 64
    broadcaster_fraction: float = 0.15
    peers_per_observer: int = 12
    clock_jitter_sd_ms: float = 25.0

    #: Fixed per-observer clock offset. ASSUMPTION: observers are
    #: NTP-disciplined, which typically holds a host within a few tens of
    #: milliseconds over the internet. An earlier +/-250 ms default was
    #: unrealistic and swamped the broadcaster signal with clock error
    #: alone - worth remembering when the October capture defines this for
    #: real, because the value directly sets how flat a "flat" vector is.
    clock_bias_range_ms: float = 40.0
    base_epoch_ms: int = 1_500_000_000_000  # arbitrary synthetic origin
    window_ms: int = 90 * 24 * 3_600 * 1_000
    seed: int = 0

    #: Probability that any single (transaction, observer) sighting is simply
    #: absent. Real captures drop announcements - the peer never relayed to
    #: this vantage point, the process was restarting, the packet was lost.
    #: A dataset with perfect coverage would let Phase 3 assume a complete
    #: vector, which no real capture supplies.
    missing_observation_rate: float = 0.0

    #: Observers that record nothing at all, simulating a dead vantage point.
    failed_observers: tuple[str, ...] = ()

    propagation: PropagationModel = field(default_factory=PropagationModel)

    def __post_init__(self) -> None:
        if self.n_observers < 1:
            raise ValueError("n_observers must be >= 1")
        if self.n_origins < 1:
            raise ValueError("n_origins must be >= 1")
        if not 0.0 <= self.broadcaster_fraction <= 1.0:
            raise ValueError("broadcaster_fraction must be in [0, 1]")
        if not 0.0 <= self.missing_observation_rate <= 1.0:
            raise ValueError("missing_observation_rate must be in [0, 1]")

    def describe(self) -> dict:
        """Configuration as plain data, for the reproducibility manifest."""
        out = asdict(self)
        out["failed_observers"] = list(self.failed_observers)
        return out


# ---- topology ---------------------------------------------------------


def _synthetic_ips(rng: np.random.Generator, n: int) -> list[str]:
    """Distinct addresses from RFC 5737 documentation ranges."""
    pool: list[str] = []
    for block in _TEST_NETS:
        pool.extend(str(ip) for ip in ipaddress.ip_network(block).hosts())
    if n > len(pool):
        raise ValueError(
            f"need {n} addresses but the documentation ranges hold {len(pool)}"
        )
    picked = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in picked]


def build_nodes(config: NetworkConfig) -> pd.DataFrame:
    """Origin nodes: the simulated sources of transactions."""
    rng = np.random.default_rng([config.seed, 1])
    n = config.n_origins
    ips = _synthetic_ips(rng, n)
    n_broadcasters = int(round(n * config.broadcaster_fraction))

    is_broadcaster = np.zeros(n, dtype=bool)
    is_broadcaster[rng.choice(n, size=n_broadcasters, replace=False)] = True

    return pd.DataFrame(
        {
            "node_id": [f"origin-{i:03d}" for i in range(n)],
            "ip": ips,
            "port": rng.choice([8333, 8333, 8333, 8334, 18333], size=n),
            "asn": rng.integers(64512, 65535, size=n),  # RFC 6996 private ASNs
            "region": rng.choice(REGIONS, size=n),
            "is_known_broadcaster": is_broadcaster,
        }
    )


def build_observers(config: NetworkConfig, nodes: pd.DataFrame) -> pd.DataFrame:
    """Observer vantage points, each with a clock bias and a peer table.

    ``clock_bias_ms`` is a fixed per-observer offset - an unsynchronised
    clock - as distinct from the per-record jitter. The next phase should be
    robust to bias, since it shifts a whole arrival vector without changing
    the ordering.
    """
    rng = np.random.default_rng([config.seed, 2])
    n = config.n_observers

    # Peer tables are drawn from ordinary relays only. If a broadcaster IP
    # could also appear as somebody's ordinary relay, then seeing that IP in
    # a record would no longer tell you the transaction came from a
    # broadcaster, and the suppression rule downstream would be ambiguous.
    ordinary = np.flatnonzero(~nodes["is_known_broadcaster"].to_numpy())
    if ordinary.size == 0:  # degenerate config: every origin is a broadcaster
        ordinary = np.arange(len(nodes))
    peers = min(config.peers_per_observer, int(ordinary.size))

    return pd.DataFrame(
        {
            "observer_id": [f"obs-{i:02d}" for i in range(n)],
            "asn": rng.integers(64512, 65535, size=n),
            "region": rng.choice(REGIONS, size=n),
            "clock_bias_ms": rng.uniform(
                -config.clock_bias_range_ms, config.clock_bias_range_ms, size=n
            ).round(3),
            "peer_indices": [
                rng.choice(ordinary, size=peers, replace=False).tolist()
                for _ in range(n)
            ],
        }
    )


# ---- generation -------------------------------------------------------


def load_transaction_ids(data_root=None) -> np.ndarray:
    """Distinct transaction ids from AddrTx_edgelist.csv, sorted.

    AddrTx is used rather than txs_features so the announcement set matches
    the transactions the clustering actually consumes.
    """
    from obsidianchain.io import elliptic

    frame = elliptic.load_input_edges(data_root)
    return np.sort(frame["txId"].unique())


def generate(
    txids: np.ndarray,
    config: NetworkConfig | None = None,
    nodes: pd.DataFrame | None = None,
    observers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce announcement records for every transaction.

    Returns ``(observations, nodes, observers, ground_truth)``. Ground truth
    is returned as a *separate* frame precisely so it cannot be mistaken for
    part of the observation set; :func:`write_outputs` puts it in a different
    directory, and :mod:`obsidianchain.network.boundary` refuses to load it.

    Delay is lognormal with the shape cited in the module docstring, scaled
    down when the observer shares an ASN or region with the origin - that
    proximity effect is the entire synthetic signal. Known-broadcaster
    origins bypass it: every observer hears them directly, at nearly the
    same moment, from the same peer.
    """
    config = config or NetworkConfig()
    nodes = build_nodes(config) if nodes is None else nodes
    observers = build_observers(config, nodes) if observers is None else observers

    txids = np.asarray(txids)
    n_tx = int(txids.size)
    if n_tx == 0:
        empty_truth = pd.DataFrame(
            columns=["txid", "true_origin_id", "broadcaster_flag"]
        )
        return pd.DataFrame(columns=RECORD_COLUMNS), nodes, observers, empty_truth

    rng = np.random.default_rng([config.seed, 3])
    prop = config.propagation

    # Each transaction gets an origin and a synthetic broadcast instant.
    origin_idx = rng.integers(0, len(nodes), size=n_tx)
    base_ms = config.base_epoch_ms + rng.integers(0, config.window_ms, size=n_tx)

    node_asn = nodes["asn"].to_numpy()
    node_region = nodes["region"].to_numpy()
    node_ip = nodes["ip"].to_numpy()
    node_port = nodes["port"].to_numpy()
    is_broadcaster = nodes["is_known_broadcaster"].to_numpy()

    origin_asn = node_asn[origin_idx]
    origin_region = node_region[origin_idx]
    broadcast_tx = is_broadcaster[origin_idx]

    # GROUND TRUTH. Generated because the model needs it; never written to
    # the observations directory and never returned to the inference stage.
    ground_truth = pd.DataFrame(
        {
            "txid": txids,
            "true_origin_id": nodes["node_id"].to_numpy()[origin_idx],
            "broadcaster_flag": broadcast_tx,
        }
    )

    failed = set(config.failed_observers)
    frames = []
    for row, observer in observers.iterrows():
        if observer["observer_id"] in failed:
            continue  # dead vantage point: records nothing at all
        peer_pool = np.asarray(observer["peer_indices"], dtype=np.int64)

        # Proximity shortens the median; this is the differential signal.
        factor = np.ones(n_tx, dtype=np.float64)
        factor[origin_region == observer["region"]] = prop.same_region_factor
        factor[origin_asn == observer["asn"]] = prop.same_asn_factor

        mu = np.log(prop.median_ms * factor)
        delay = rng.lognormal(mean=mu, sigma=prop.sigma)

        # Broadcasters: direct connection, fast and tight.
        b_delay = rng.lognormal(
            mean=np.log(prop.broadcast_median_ms),
            sigma=prop.broadcast_sigma,
            size=n_tx,
        )
        delay = np.where(broadcast_tx, b_delay, delay)

        jitter = rng.normal(0.0, config.clock_jitter_sd_ms, size=n_tx)
        timestamp = base_ms + delay + float(observer["clock_bias_ms"]) + jitter

        # Ordinary transactions arrive via one of this observer's own peers,
        # which deliberately does NOT identify the origin. Broadcasters are
        # heard directly, so there the peer is the origin itself.
        relay_idx = peer_pool[rng.integers(0, peer_pool.size, size=n_tx)]
        peer_idx = np.where(broadcast_tx, origin_idx, relay_idx)

        frame = pd.DataFrame(
            {
                "txid": txids,
                "observer_id": observer["observer_id"],
                "peer_ip": node_ip[peer_idx],
                "peer_port": node_port[peer_idx].astype(np.int32),
                "peer_asn": node_asn[peer_idx].astype(np.int32),
                "timestamp_ms": np.round(timestamp, 3),
            }
        )
        if config.missing_observation_rate > 0:
            kept = rng.random(n_tx) >= config.missing_observation_rate
            frame = frame[kept]
        frames.append(frame)

    if frames:
        observations = pd.concat(frames, ignore_index=True)
    else:
        observations = pd.DataFrame(columns=RECORD_COLUMNS)
    observations = observations.sort_values(
        ["txid", "observer_id"], kind="stable"
    ).reset_index(drop=True)
    return observations.loc[:, RECORD_COLUMNS], nodes, observers, ground_truth


def known_broadcaster_ips(nodes: pd.DataFrame) -> set[str]:
    """IPs whose announcements must never produce a constraint.

    Accepts either the observable ``broadcaster_ips.csv`` (an ``ip`` column)
    or the ground-truth origin roster. Prefer the former: it is what a real
    deployment has, being a public list of exchange and wallet-server
    infrastructure.
    """
    if "is_known_broadcaster" in nodes.columns:
        return set(nodes.loc[nodes["is_known_broadcaster"], "ip"])
    return set(nodes["ip"])


#: The frozen September 2026 prototype configuration.
#:
#: Phase 3 consumes the dataset produced by exactly this configuration. It is
#: not to be tuned against Phase 3 results - the whole point of freezing it is
#: that a change in Phase 3 output means a change in Phase 3, not in its
#: input. The missing-observation rate is deliberately non-zero: a capture
#: with perfect coverage would let the inference stage assume a complete
#: arrival vector, which no real deployment ever has.
FROZEN_SEPTEMBER_2026 = NetworkConfig(
    n_observers=8,
    n_origins=64,
    broadcaster_fraction=0.15,
    peers_per_observer=12,
    clock_jitter_sd_ms=25.0,
    clock_bias_range_ms=40.0,
    missing_observation_rate=0.02,
    failed_observers=(),
    seed=0,
    propagation=PropagationModel(),
)


def write_outputs(
    observations: pd.DataFrame,
    nodes: pd.DataFrame,
    observers: pd.DataFrame,
    ground_truth: pd.DataFrame,
    processed_root,
    config: NetworkConfig,
    fmt: str = "parquet",
) -> dict[str, object]:
    """Write the two halves into two directories.

    ``processed/network/`` holds everything the inference stage may read:
    the announcement records, the observer roster a deployment would know
    about its own vantage points, and the public broadcaster IP list.

    ``processed/network_truth/`` holds everything it may not: which origin
    actually sent each transaction, the full origin roster with region and
    ASN, and the observers' true clock offsets. A real deployment has none
    of these; it estimates clock offset and never sees the origin roster at
    all.
    """
    from pathlib import Path

    processed_root = Path(processed_root)
    obs_dir = processed_root / OBSERVATIONS_DIR
    truth_dir = processed_root / TRUTH_DIR
    obs_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        obs_path = obs_dir / "observations.parquet"
        observations.to_parquet(obs_path, index=False)
    elif fmt == "csv":
        obs_path = obs_dir / "observations.csv"
        observations.to_csv(obs_path, index=False)
    else:
        raise ValueError(f"unsupported format {fmt!r}; use 'parquet' or 'csv'")

    # --- observable side ------------------------------------------------
    # An operator knows which vantage points they run and roughly where, but
    # not their true clock error - that is estimated, never known.
    observers.loc[:, ["observer_id", "asn", "region"]].to_csv(
        obs_dir / "observers.csv", index=False
    )
    # Exchange and wallet-server IPs are publicly documented, so a plain list
    # of them is realistic evidence. The origin *roster* is not.
    broadcaster_path = obs_dir / "broadcaster_ips.csv"
    nodes.loc[nodes["is_known_broadcaster"], ["ip"]].to_csv(
        broadcaster_path, index=False
    )

    # --- ground truth ---------------------------------------------------
    truth_path = truth_dir / "ground_truth.csv"
    ground_truth.to_csv(truth_path, index=False)
    nodes.to_csv(truth_dir / "origin_nodes.csv", index=False)
    observers.loc[:, ["observer_id", "clock_bias_ms"]].to_csv(
        truth_dir / "observer_clocks.csv", index=False
    )
    (truth_dir / "README.txt").write_text(
        "GROUND TRUTH - EVALUATION ONLY\n"
        "==============================\n"
        "Nothing in this directory may be read by the inference stage.\n"
        "It exists so an evaluation harness can score Phase 3 output after\n"
        "the fact. A real Bitcoin capture supplies none of it.\n",
        encoding="utf-8",
    )

    manifest = build_manifest(observations, config, obs_path)
    manifest_path = obs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "observations": obs_path,
        "observers": obs_dir / "observers.csv",
        "broadcaster_ips": broadcaster_path,
        "ground_truth": truth_path,
        "manifest": manifest_path,
        "rows": int(len(observations)),
        "bytes": int(obs_path.stat().st_size),
        "manifest_data": manifest,
    }


def build_manifest(
    observations: pd.DataFrame, config: NetworkConfig, observations_path
) -> dict:
    """Reproducibility manifest for the observation set.

    Deliberately free of machine-specific detail - no hostname, no absolute
    path, no generation timestamp - so two people running the same command on
    different machines produce identical manifests. If they do not, the input
    genuinely differs and a Phase 3 comparison is invalid.
    """
    import hashlib
    from pathlib import Path

    observations_path = Path(observations_path)
    digest = hashlib.sha256()
    with observations_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return {
        "generator_version": GENERATOR_VERSION,
        "dataset_sha256": digest.hexdigest(),
        "observations_filename": observations_path.name,
        "record_count": int(len(observations)),
        "transaction_count": int(observations["txid"].nunique())
        if len(observations)
        else 0,
        "observer_count": int(observations["observer_id"].nunique())
        if len(observations)
        else 0,
        "record_columns": list(RECORD_COLUMNS),
        "configuration": config.describe(),
    }


def format_summary(
    observations: pd.DataFrame,
    nodes: pd.DataFrame,
    observers: pd.DataFrame,
    config: NetworkConfig,
) -> str:
    """Render a generation summary, warning banner included."""
    width = 78
    out: list[str] = []
    add = out.append
    add("=" * width)
    add("obsidianchain :: synthetic network announcements")
    add("=" * width)
    add("  *** SYNTHETIC DATA - DEMONSTRATES THE MECHANISM, VALIDATES NOTHING ***")
    add("  No packet was captured. Results computed on this data measure the")
    add("  generator, not Bitcoin. Real validation needs mainnet capture plus")
    add("  a controlled ground-truth set; both are October work.")
    add("")
    n_broadcast = int(nodes["is_known_broadcaster"].sum())
    n_missing_expected = config.missing_observation_rate
    add("-- configuration " + "-" * (width - 18))
    add(f"  seed                            {config.seed:>12}")
    add(f"  observers                       {len(observers):>12,}")
    add(f"  origin nodes                    {len(nodes):>12,}")
    add(f"  known broadcasters              {n_broadcast:>12,}"
        f"   {n_broadcast / len(nodes) * 100:.1f}% of origins")
    add(f"  peers per observer              {config.peers_per_observer:>12,}")
    add("")
    prop = config.propagation
    add("-- propagation " + "-" * (width - 16))
    add(f"  lognormal sigma                 {prop.sigma:>12.4f}   Decker & Wattenhofer 2013")
    add(f"  median delay (ms)               {prop.median_ms:>12,.0f}   ASSUMPTION, not measured")
    add(f"  implied mean (ms)               {prop.mean_ms:>12,.0f}")
    add(f"  broadcaster median (ms)         {prop.broadcast_median_ms:>12,.0f}   ASSUMPTION")
    add(f"  same-ASN / same-region factor   {prop.same_asn_factor:>6.2f} /{prop.same_region_factor:>6.2f}")
    add(f"  clock jitter sd (ms)            {config.clock_jitter_sd_ms:>12,.0f}")
    add(f"  clock bias range (ms)           +/-{config.clock_bias_range_ms:>9,.0f}")
    add(f"  missing-observation rate        {n_missing_expected:>12.3f}")
    add(f"  failed observers                {len(config.failed_observers):>12,}")
    add("")
    n_tx = int(observations["txid"].nunique()) if len(observations) else 0
    add("-- output " + "-" * (width - 11))
    add(f"  transactions                    {n_tx:>12,}")
    add(f"  announcement records            {len(observations):>12,}")
    add(f"  distinct peer IPs seen          {observations['peer_ip'].nunique():>12,}")
    add("=" * width)
    return "\n".join(out)
