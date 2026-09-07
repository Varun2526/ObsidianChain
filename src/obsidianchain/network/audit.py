"""Phase 2 audit: information boundary, identifiability, and stress cases.

Three separate questions, kept separate on purpose.

**Is the boundary intact?** Can anything the generator knows reach an
inference stage? This is a structural check on files and schemas, and its
answer is binary.

**Is the synthetic data unrealistically easy?** A generator writing data for
its own analysis can make the problem trivial without anyone noticing. The
identifiability section measures how separable origins actually are, so the
answer is on the record rather than assumed. If it says the data is too easy,
that is a limitation to publish, not a number to improve.

**Does the instrumentation survive difficulty?** The stress scenarios add
jitter, drop observations, kill observers, and construct deliberately
overlapping cases. They check that the *instrumentation* stays correct as the
signal degrades - not that any inference stays accurate, which is Phase 3's
problem and cannot be settled on synthetic data at all.

Nothing here infers an origin. No similarity, no distance-based decision, no
threshold that could later be tuned. The identifiability section computes
distances only to describe the dataset, and it uses ground truth openly and
exclusively for that description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.network import arrivals, boundary, synthetic

#: Offsets are bucketed to this resolution when counting near-duplicate
#: vectors. Coarser than clock error, so two vectors that collide really are
#: indistinguishable at the precision the measurement supports.
NEAR_DUPLICATE_BUCKET_MS = 250.0

#: Transactions sampled for the pairwise distance comparison. The full
#: pairwise problem is quadratic and unnecessary - a sample answers the
#: question, which is about distributions rather than any individual pair.
IDENTIFIABILITY_SAMPLE = 4000


@dataclass
class LeakageReport:
    origin_id_exposed: bool = False
    entity_labels_exposed: bool = False
    broadcaster_metadata_exposed: bool = False
    clock_bias_exposed: bool = False
    observation_columns: list[str] = field(default_factory=list)
    truth_files_present: list[str] = field(default_factory=list)
    boundary_error: str | None = None

    @property
    def clean(self) -> bool:
        return not (
            self.origin_id_exposed
            or self.entity_labels_exposed
            or self.broadcaster_metadata_exposed
            or self.clock_bias_exposed
            or self.boundary_error
        )


@dataclass
class IdentifiabilityReport:
    n_origins: int = 0
    n_transactions: int = 0
    rank_separation_ratio: float = float("nan")
    first_observer_lift: float = float("nan")
    centroid_separation_ratio: float = float("nan")
    tx_per_origin_min: int = 0
    tx_per_origin_median: float = 0.0
    tx_per_origin_max: int = 0
    n_observers: int = 0
    broadcaster_fraction: float = 0.0
    missing_observation_rate: float = 0.0
    jitter_sd_ms: float = 0.0
    clock_bias_range_ms: float = 0.0
    delay_percentiles: dict = field(default_factory=dict)
    within_origin_distance: dict = field(default_factory=dict)
    between_origin_distance: dict = field(default_factory=dict)
    separation_ratio: float = float("nan")
    near_duplicate_vectors: int = 0
    near_duplicate_groups: int = 0
    near_duplicates_cross_origin: int = 0


def audit_leakage(processed_root) -> LeakageReport:
    """Check what an inference stage can actually reach on disk."""
    report = LeakageReport()
    processed_root = Path(processed_root)

    try:
        observations = boundary.load_observations(processed_root)
        report.observation_columns = list(observations.columns)
    except boundary.GroundTruthLeakError as exc:
        report.boundary_error = str(exc)
        return report
    except FileNotFoundError as exc:
        report.boundary_error = str(exc)
        return report

    lowered = {c.lower() for c in report.observation_columns}
    report.origin_id_exposed = any("origin" in c for c in lowered)
    report.entity_labels_exposed = any("entity" in c for c in lowered)
    report.broadcaster_metadata_exposed = any("broadcaster" in c for c in lowered)

    observers = boundary.load_observers(processed_root)
    report.clock_bias_exposed = any(
        "clock_bias" in c.lower() for c in observers.columns
    )

    truth_dir = processed_root / synthetic.TRUTH_DIR
    if truth_dir.is_dir():
        report.truth_files_present = sorted(p.name for p in truth_dir.iterdir())
    return report


def _vector_matrix(vectors: arrivals.ArrivalVectors) -> np.ndarray:
    """Offsets with missing entries filled by the row's own maximum.

    A missing sighting is not zero delay. Filling with the row maximum keeps
    an absent observer from masquerading as an early one, which would invent
    exactly the signal this dataset is supposed to leave ambiguous.
    """
    matrix = vectors.offsets_ms.copy()
    row_max = np.nanmax(
        np.where(np.isnan(matrix), -np.inf, matrix), axis=1, keepdims=True
    )
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    return np.where(np.isnan(matrix), row_max, matrix)


def audit_identifiability(
    vectors: arrivals.ArrivalVectors,
    ground_truth: pd.DataFrame,
    config: synthetic.NetworkConfig,
    sample: int = IDENTIFIABILITY_SAMPLE,
    seed: int = 0,
) -> IdentifiabilityReport:
    """Measure how separable the synthetic origins actually are.

    Uses ground truth openly: this is the evaluation harness describing the
    dataset, not an inference stage consuming it.
    """
    report = IdentifiabilityReport(
        n_observers=vectors.n_observers,
        n_transactions=vectors.n_transactions,
        missing_observation_rate=config.missing_observation_rate,
        jitter_sd_ms=config.clock_jitter_sd_ms,
        clock_bias_range_ms=config.clock_bias_range_ms,
    )

    truth = ground_truth.set_index("txid")
    aligned = truth.reindex(vectors.txids)
    origins = aligned["true_origin_id"].to_numpy()
    is_broadcaster = aligned["broadcaster_flag"].fillna(False).to_numpy().astype(bool)

    report.n_origins = int(pd.Series(origins).nunique())
    report.broadcaster_fraction = float(is_broadcaster.mean())

    per_origin = pd.Series(origins).value_counts()
    if len(per_origin):
        report.tx_per_origin_min = int(per_origin.min())
        report.tx_per_origin_median = float(per_origin.median())
        report.tx_per_origin_max = int(per_origin.max())

    spread = vectors.spread_ms[~np.isnan(vectors.spread_ms)]
    if spread.size:
        report.delay_percentiles = {
            f"p{p}": float(np.percentile(spread, p)) for p in (5, 25, 50, 75, 95, 99)
        }

    # --- pairwise distances, ordinary transactions only -----------------
    matrix = _vector_matrix(vectors)
    ordinary = np.flatnonzero(~is_broadcaster & (vectors.n_observed >= 2))
    rng = np.random.default_rng(seed)
    if ordinary.size > sample:
        ordinary = rng.choice(ordinary, size=sample, replace=False)

    if ordinary.size >= 2:
        from scipy.spatial.distance import pdist

        sub = matrix[ordinary]
        sub_origin = origins[ordinary]
        # Normalise each vector by its own spread: absolute delay differs
        # between transactions for reasons unrelated to origin, and comparing
        # raw magnitudes would measure that instead.
        scale = np.maximum(sub.max(axis=1, keepdims=True), 1.0)
        unit = sub / scale

        # pdist keeps the condensed form: a 4,000-point squareform costs a
        # gigabyte of intermediate, which is pure waste for a median.
        # Origin ids are strings, so compare integer codes instead.
        origin_codes = pd.factorize(pd.Series(sub_origin))[0].astype(np.float64)
        same_pairs = pdist(origin_codes.reshape(-1, 1), metric="hamming") == 0
        dist = pdist(unit)
        within, between = dist[same_pairs], dist[~same_pairs]
        if within.size:
            report.within_origin_distance = _describe(within)
        if between.size:
            report.between_origin_distance = _describe(between)
        if within.size and between.size and np.median(within) > 0:
            report.separation_ratio = float(
                np.median(between) / np.median(within)
            )

        # A second, ordering-only view. If offsets and ranks agree that
        # origins do not separate, the finding is about the data rather than
        # about the choice of distance.
        rank_dist = pdist(vectors.ranks[ordinary].astype(float), metric="hamming")
        r_within, r_between = rank_dist[same_pairs], rank_dist[~same_pairs]
        if r_within.size and r_between.size and np.median(r_within) > 0:
            report.rank_separation_ratio = float(
                np.median(r_between) / np.median(r_within)
            )

        # The bluntest diagnostic: does sharing an origin make two
        # transactions more likely to be seen first by the same observer?
        first = vectors.ranks[ordinary].argmin(axis=1).astype(float)
        agree = pdist(first.reshape(-1, 1), metric="hamming") == 0
        p_same = float(agree[same_pairs].mean()) if same_pairs.any() else float("nan")
        p_diff = float(agree[~same_pairs].mean()) if (~same_pairs).any() else float("nan")
        if p_diff and np.isfinite(p_diff) and p_diff > 0:
            report.first_observer_lift = p_same / p_diff

        # Aggregate view. A signal too weak to read from one transaction can
        # still be plain across thousands, so average each origin's vectors
        # and ask whether the centroids separate.
        #
        # The denominator is the centroid's own standard error, not the
        # scatter of individual transactions. Averaging n samples shrinks the
        # estimate's error by sqrt(n), so dividing by raw scatter would make
        # the ratio fall as more data is added - which is exactly backwards
        # for a question about whether pooling helps.
        # Computed over EVERY usable transaction, not the distance sample.
        # Group means are cheap, and restricting them to the sample would
        # inflate the standard error by the square root of the sampling
        # fraction - understating the aggregate signal by roughly sixfold
        # here, purely as an artifact of how many rows we happened to draw.
        full = np.flatnonzero(~is_broadcaster & (vectors.n_observed >= 2))
        full_matrix = matrix[full]
        full_scale = np.maximum(full_matrix.max(axis=1, keepdims=True), 1.0)
        full_unit = full_matrix / full_scale
        frame = pd.DataFrame(full_unit)
        frame["origin"] = pd.factorize(pd.Series(origins[full]))[0]
        grouped = frame.groupby("origin")
        centroids = grouped.mean().to_numpy()
        per_origin_n = grouped.size().to_numpy()
        if len(centroids) >= 2 and per_origin_n.min() > 0:
            between_c = float(np.median(pdist(centroids)))
            scatter = float(
                np.median(
                    np.linalg.norm(
                        full_unit - grouped.transform("mean").to_numpy(), axis=1
                    )
                )
            )
            standard_error = scatter / np.sqrt(float(np.median(per_origin_n)))
            if standard_error > 0:
                report.centroid_separation_ratio = between_c / standard_error

    # --- near-duplicate vectors ------------------------------------------
    usable = ~is_broadcaster & (vectors.n_observed >= 2)
    if usable.any():
        bucketed = np.round(matrix[usable] / NEAR_DUPLICATE_BUCKET_MS).astype(np.int64)
        keys = [tuple(row) for row in bucketed]
        counts = pd.Series(keys).value_counts()
        duplicated = counts[counts > 1]
        report.near_duplicate_groups = int(len(duplicated))
        report.near_duplicate_vectors = int(duplicated.sum())

        frame = pd.DataFrame({"key": keys, "origin": origins[usable]})
        per_key = frame.groupby("key")["origin"].nunique()
        report.near_duplicates_cross_origin = int((per_key > 1).sum())

    return report


def _describe(values: np.ndarray) -> dict:
    return {
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


# ---- stress scenarios -------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    description: str
    n_transactions: int
    n_records: int
    median_spread_ms: float
    usable: int
    no_evidence: int
    reasons: dict = field(default_factory=dict)
    note: str = ""

    @property
    def usable_fraction(self) -> float:
        total = self.usable + self.no_evidence
        return self.usable / total if total else 0.0


def _run_scenario(
    name: str,
    description: str,
    config: synthetic.NetworkConfig,
    txids: np.ndarray,
    note: str = "",
) -> ScenarioResult:
    observations, nodes, observers, truth = synthetic.generate(txids, config)
    if observations.empty:
        return ScenarioResult(name, description, 0, 0, float("nan"), 0, 0, note=note)

    vectors = arrivals.build(
        observations, observer_ids=sorted(observers["observer_id"].tolist())
    )
    broadcaster_ips = synthetic.known_broadcaster_ips(nodes)
    bc_txids = set(
        observations.loc[observations["peer_ip"].isin(broadcaster_ips), "txid"]
    )
    labels = arrivals.classify_evidence(vectors, bc_txids)
    reasons = (
        labels.loc[labels["no_evidence_reason"] != "", "no_evidence_reason"]
        .value_counts()
        .to_dict()
    )
    spread = vectors.spread_ms[~np.isnan(vectors.spread_ms)]
    return ScenarioResult(
        name=name,
        description=description,
        n_transactions=vectors.n_transactions,
        n_records=int(len(observations)),
        median_spread_ms=float(np.median(spread)) if spread.size else float("nan"),
        usable=int((labels["evidence"] == arrivals.Evidence.USABLE.value).sum()),
        no_evidence=int(
            (labels["evidence"] == arrivals.Evidence.NO_EVIDENCE.value).sum()
        ),
        reasons=reasons,
        note=note,
    )


def run_stress_scenarios(
    txids: np.ndarray, base: synthetic.NetworkConfig | None = None
) -> list[ScenarioResult]:
    """Degrade the signal deliberately and check the instrumentation holds.

    These exist so Phase 3 meets hard cases rather than only clean ones. A
    scenario that produces fewer usable transactions has not failed - it has
    done its job.
    """
    from dataclasses import replace

    base = base or synthetic.FROZEN_SEPTEMBER_2026
    results: list[ScenarioResult] = []

    results.append(
        _run_scenario("baseline", "frozen configuration", base, txids)
    )

    # A. clock jitter sweep
    for sd in (50.0, 150.0, 500.0):
        results.append(
            _run_scenario(
                f"jitter-{int(sd)}ms",
                f"clock jitter sd raised to {sd:.0f} ms",
                replace(base, clock_jitter_sd_ms=sd),
                txids,
                note="ordering degrades as jitter approaches propagation delay",
            )
        )

    # B. missing observations
    for rate in (0.10, 0.30, 0.60):
        results.append(
            _run_scenario(
                f"missing-{int(rate * 100)}pct",
                f"{rate:.0%} of sightings dropped",
                replace(base, missing_observation_rate=rate),
                txids,
                note="incomplete vectors must stay explicitly incomplete",
            )
        )

    # C. observer failure
    for n_failed in (1, 3):
        failed = tuple(f"obs-{i:02d}" for i in range(n_failed))
        results.append(
            _run_scenario(
                f"observers-down-{n_failed}",
                f"{n_failed} vantage point(s) offline",
                replace(base, failed_observers=failed),
                txids,
                note="a dead observer must be absent, never zero",
            )
        )

    # D. propagation overlap: shrink proximity contrast so different origins
    #    produce near-identical arrival patterns.
    flat_prop = synthetic.PropagationModel(
        median_ms=base.propagation.median_ms,
        sigma=base.propagation.sigma,
        same_asn_factor=0.97,
        same_region_factor=0.99,
    )
    results.append(
        _run_scenario(
            "overlapping-origins",
            "proximity contrast removed; origins look alike",
            replace(base, propagation=flat_prop),
            txids,
            note="D: different origins, deliberately similar arrival patterns",
        )
    )

    # E. same-origin variance: heavy tail widened so one origin's own
    #    transactions scatter.
    wide_prop = synthetic.PropagationModel(
        median_ms=base.propagation.median_ms,
        sigma=base.propagation.sigma * 2.0,
        same_asn_factor=base.propagation.same_asn_factor,
        same_region_factor=base.propagation.same_region_factor,
    )
    results.append(
        _run_scenario(
            "same-origin-scatter",
            "delay variance doubled; one origin's vectors diverge",
            replace(base, propagation=wide_prop),
            txids,
            note="E: same origin, deliberately dissimilar arrival vectors",
        )
    )

    # F. broadcaster dominance
    results.append(
        _run_scenario(
            "broadcaster-heavy",
            "half of all origins are known broadcasters",
            replace(base, broadcaster_fraction=0.5),
            txids,
            note="F: broadcaster transactions must stay NO_EVIDENCE throughout",
        )
    )
    return results


# ---- report -----------------------------------------------------------

_W = 78


def format_audit(
    leakage: LeakageReport,
    identifiability: IdentifiabilityReport,
    scenarios: list[ScenarioResult],
    manifest: dict,
    checks: dict[str, bool],
) -> str:
    """Render the Phase 2 audit."""
    out: list[str] = []
    add = out.append
    ok = lambda flag: "PASS" if flag else "FAIL"  # noqa: E731
    yn = lambda flag: "YES" if flag else "NO"  # noqa: E731

    add("OBSIDIANCHAIN - PHASE 2 NETWORK AUDIT")
    add("=" * _W)
    add("")
    add("DATA TYPE")
    add("  Synthetic - mechanism demonstration only.")
    add("  Nothing here is evidence about Bitcoin. Real validation needs")
    add("  mainnet capture plus controlled wallets with known ground truth.")
    add("")

    add(f"  {'Transactions:':<32}{identifiability.n_transactions:>14,}")
    add(f"  {'Observers:':<32}{identifiability.n_observers:>14,}")
    add(f"  {'Observations:':<32}{manifest.get('record_count', 0):>14,}")
    add(f"  {'Origins:':<32}{identifiability.n_origins:>14,}")
    add(f"  {'Broadcaster fraction:':<32}{identifiability.broadcaster_fraction:>14.3f}")
    add(f"  {'Missing-observation rate:':<32}"
        f"{identifiability.missing_observation_rate:>14.3f}")
    add("")

    add("GROUND-TRUTH LEAKAGE")
    add(f"  {'Origin ID exposed to Phase 3:':<38}{yn(leakage.origin_id_exposed):>8}")
    add(f"  {'Entity labels exposed:':<38}{yn(leakage.entity_labels_exposed):>8}")
    add(f"  {'Broadcaster metadata exposed:':<38}"
        f"{yn(leakage.broadcaster_metadata_exposed):>8}")
    add(f"  {'Observer clock bias exposed:':<38}{yn(leakage.clock_bias_exposed):>8}")
    add(f"  observation columns: {leakage.observation_columns}")
    if leakage.truth_files_present:
        add(f"  ground truth quarantined in {synthetic.TRUTH_DIR}/: "
            f"{leakage.truth_files_present}")
    if leakage.boundary_error:
        add(f"  BOUNDARY ERROR: {leakage.boundary_error}")
    add("")

    add("REPRODUCIBILITY")
    add(f"  {'Seed fixed:':<38}"
        f"{yn('seed' in manifest.get('configuration', {})):>8}")
    add(f"  {'Configuration frozen:':<38}{yn(checks.get('config_frozen', False)):>8}")
    add(f"  generator version   {manifest.get('generator_version', '?')}")
    add(f"  dataset sha256      {str(manifest.get('dataset_sha256', '?'))[:32]}...")
    add("")

    add("IDENTIFIABILITY - IS THE SYNTHETIC DATA TOO EASY?")
    add(f"  transactions per origin   min {identifiability.tx_per_origin_min:,} / "
        f"median {identifiability.tx_per_origin_median:,.0f} / "
        f"max {identifiability.tx_per_origin_max:,}")
    if identifiability.delay_percentiles:
        pct = identifiability.delay_percentiles
        add(f"  arrival spread (ms)       p25 {pct.get('p25', 0):,.0f} / "
            f"median {pct.get('p50', 0):,.0f} / p95 {pct.get('p95', 0):,.0f}")
    if identifiability.within_origin_distance:
        w = identifiability.within_origin_distance
        b = identifiability.between_origin_distance
        add(f"  within-origin distance    median {w['median']:.4f}"
            f"   (p25 {w['p25']:.4f} / p75 {w['p75']:.4f})")
        add(f"  between-origin distance   median {b['median']:.4f}"
            f"   (p25 {b['p25']:.4f} / p75 {b['p75']:.4f})")
        add(f"  separation ratio          {identifiability.separation_ratio:.3f}"
            f"   (between / within, 1.0 = indistinguishable)")
        add(f"  ...by rank vector only    "
            f"{identifiability.rank_separation_ratio:.3f}"
            f"   ordering-only view, guards against a metric artifact")
        add(f"  first-observer lift       "
            f"{identifiability.first_observer_lift:.3f}"
            f"   P(same first obs | same origin) / P(| different)")
        add(f"  per-origin centroids      "
            f"{identifiability.centroid_separation_ratio:.3f}"
            f"   between-centroid distance in standard errors")
    add(f"  near-identical vectors    {identifiability.near_duplicate_vectors:,}"
        f" in {identifiability.near_duplicate_groups:,} groups")
    add(f"    ...spanning >1 origin   {identifiability.near_duplicates_cross_origin:,}"
        f"   genuinely ambiguous cases")
    add("")
    ratio = identifiability.separation_ratio
    centroid = identifiability.centroid_separation_ratio
    if np.isfinite(ratio):
        if ratio > 3.0:
            add("  READ THIS: origins separate far more cleanly than any real")
            add("  capture would. The generator builds proximity in and the")
            add("  measurement finds it. Treat Phase 3 accuracy on this data as")
            add("  an upper bound that says nothing about mainnet.")
        elif ratio > 1.5:
            add("  Origins are separable but overlap substantially, which is")
            add("  the intended regime: hard enough that a method can fail.")
        else:
            add("  LIMITATION - READ BEFORE DESIGNING PHASE 3")
            add("  A single transaction's arrival vector carries almost no")
            add("  origin information here. Offsets, rank vectors and the")
            add("  first-observer test all land at a ratio near 1.0, so this")
            add("  is a property of the data and not of the chosen distance.")
            add("")
            add("  The cause is the propagation model, not a defect: with")
            add("  sigma = 1.15 each observer's delay varies roughly threefold")
            add("  per draw, while sharing an ASN or region only shifts the")
            add("  median by a factor of 0.45 to 0.70. Per-transaction noise")
            add("  simply swamps the systematic proximity effect.")
            add("")
            if np.isfinite(centroid) and centroid > 2.0:
                add("  The signal is nonetheless present in AGGREGATE: origin")
                add(f"  centroids sit {centroid:.1f} standard errors apart, so")
                add("  averaging an origin's transactions recovers position. A")
                add("  Phase 3 method that pools evidence across many")
                add("  transactions can work here; one that decides from a")
                add("  single arrival vector cannot, and its failure would")
                add("  measure sigma rather than the method.")
            else:
                add("  No aggregate signal either. Phase 3 cannot demonstrate")
                add("  the mechanism on this dataset as configured.")
            add("")
            add("  Do NOT lower sigma to make this easier. sigma is the one")
            add("  parameter with a published source, and tuning it to obtain")
            add("  a better Phase 3 result would make the demonstration")
            add("  circular. Report the limitation instead.")
    add("")

    add("STRESS SCENARIOS")
    add(f"  {'scenario':<24}{'records':>10}{'usable':>9}{'no-ev':>8}"
        f"{'usable%':>9}{'med spread':>12}")
    for s in scenarios:
        add(f"  {s.name:<24}{s.n_records:>10,}{s.usable:>9,}{s.no_evidence:>8,}"
            f"{s.usable_fraction * 100:>8.1f}%{s.median_spread_ms:>12,.0f}")
    add("")
    add("  Fewer usable transactions under stress is the scenario working,")
    add("  not the instrumentation failing.")
    add("")

    add("INSTRUMENTATION CHECKS")
    for name, passed in checks.items():
        add(f"  {name.replace('_', ' ').capitalize():<38}{ok(passed):>8}")
    add("")

    ready = leakage.clean and all(checks.values())
    add("STATUS")
    add(f"  {'Phase 2 ready for Phase 3:':<38}{yn(ready):>8}")
    if not ready:
        add("  Resolve the failures above before starting Phase 3.")
    add("=" * _W)
    return "\n".join(out)
