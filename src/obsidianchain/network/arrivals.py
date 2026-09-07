"""Turn announcement records into per-transaction arrival-time vectors.

This is instrumentation: it reshapes observations into the form the
constraint stage consumes and computes descriptive statistics. It makes no
inference and produces no constraints.

Input is the record set from :mod:`obsidianchain.network.synthetic` (or, in
October, a real capture with the same schema). Output is one row per
transaction holding, for each observer, when that observer first saw the
transaction.

Why offsets rather than absolute times
--------------------------------------
Observers do not share a clock. Each carries an unknown fixed bias, so
absolute timestamps are not comparable between vantage points. Subtracting
the earliest arrival gives an offset vector that is invariant to any shift
applied to all observers equally, and the *rank* vector is invariant to per
observer bias as well - it survives an observer whose clock is simply wrong.
Both are emitted; the rank vector is the more robust of the two and the one
to prefer when bias is suspected.

The flatness warning
--------------------
``spread_ms`` is the difference between the last and first arrival. When it
approaches zero the observers are effectively simultaneous and the vector
carries no ordering information at all. That happens by construction for
known-broadcaster transactions, which every observer hears directly. A near
zero spread must not be read as agreement or as evidence of anything - it
means the measurement is uninformative, and downstream code should treat it
as absence of evidence rather than evidence of absence. :func:`flag_flat`
marks these rows so the next stage can drop them explicitly rather than by
accident.

Duplicate announcements
-----------------------
A real capture will record the same transaction from an observer more than
once - a peer re-advertises, or several peers relay it. Only the earliest
sighting per (transaction, observer) is meaningful for propagation timing,
so later ones are collapsed. The count of collapsed records is reported
rather than discarded silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class Evidence(str, Enum):
    """What a transaction's arrival vector is permitted to support.

    There are exactly two states here, and neither is a constraint. Deciding
    between must-link and cannot-link is Phase 3's job; this stage only says
    whether the measurement is capable of supporting any decision at all.

    ``NO_EVIDENCE`` is not a weak signal or a low-confidence one. It means
    the vector contains no ordering information, and a stage that turns it
    into a link of either kind has manufactured the link. Absence of evidence
    is not evidence of absence, and this enum exists so the distinction has
    to be handled rather than assumed.
    """

    USABLE = "USABLE"
    NO_EVIDENCE = "NO_EVIDENCE"


#: Why a transaction was ruled out. Reported so the exclusion is auditable.
NO_EVIDENCE_REASONS = (
    "known_broadcaster",  # heard directly by everyone; ordering is meaningless
    "flat_vector",        # spread within total clock error
    "too_few_observers",  # fewer than two sightings; nothing to order
)

#: A vector whose spread is below this is treated as carrying no ordering.
#: It must sit above the total clock error across observers - bias range plus
#: jitter - or genuinely simultaneous announcements will look ordered when
#: the only signal is unsynchronised clocks. With the generator defaults
#: (+/-40 ms bias, 25 ms jitter sd) that floor is roughly 150 ms.
DEFAULT_FLAT_THRESHOLD_MS = 250.0


@dataclass
class ArrivalVectors:
    """Per-transaction arrival times across a fixed observer axis."""

    txids: np.ndarray
    observer_ids: list[str]

    absolute_ms: np.ndarray
    """(n_tx, n_observers) raw timestamps; NaN where unobserved."""

    offsets_ms: np.ndarray
    """(n_tx, n_observers) arrival minus the transaction's earliest arrival."""

    ranks: np.ndarray
    """(n_tx, n_observers) 0 = first to see it; -1 where unobserved."""

    first_observer: np.ndarray
    spread_ms: np.ndarray
    n_observed: np.ndarray
    duplicates_collapsed: int = 0

    @property
    def n_transactions(self) -> int:
        return int(self.txids.size)

    @property
    def n_observers(self) -> int:
        return len(self.observer_ids)

    def flag_flat(
        self, threshold_ms: float = DEFAULT_FLAT_THRESHOLD_MS
    ) -> np.ndarray:
        """Transactions whose observers are effectively simultaneous.

        These carry no ordering information. Downstream code should skip
        them rather than infer from a vector that says nothing.
        """
        return self.spread_ms <= threshold_ms

    def to_frame(self, threshold_ms: float = DEFAULT_FLAT_THRESHOLD_MS) -> pd.DataFrame:
        """Wide table: one row per transaction, one offset column per observer."""
        data = {"txid": self.txids}
        for i, observer in enumerate(self.observer_ids):
            data[f"offset_ms__{observer}"] = self.offsets_ms[:, i]
        for i, observer in enumerate(self.observer_ids):
            data[f"rank__{observer}"] = self.ranks[:, i]
        data["first_observer"] = self.first_observer
        data["spread_ms"] = self.spread_ms
        data["n_observed"] = self.n_observed
        data["is_flat"] = self.flag_flat(threshold_ms)
        return pd.DataFrame(data)

    def with_evidence(
        self,
        broadcaster_txids: set | None = None,
        threshold_ms: float = DEFAULT_FLAT_THRESHOLD_MS,
    ) -> pd.DataFrame:
        """Wide table plus the evidence classification."""
        frame = self.to_frame(threshold_ms)
        labels = classify_evidence(self, broadcaster_txids, threshold_ms)
        return frame.merge(labels, on="txid", how="left")


def build(
    observations: pd.DataFrame, observer_ids: list[str] | None = None
) -> ArrivalVectors:
    """Build arrival vectors from announcement records.

    Args:
        observations: Records with ``txid``, ``observer_id``, ``timestamp_ms``.
        observer_ids: Fixed observer axis. Defaults to the sorted set present
            in the data. Pass it explicitly when comparing runs, so that an
            observer that happened to see nothing still gets a column.
    """
    required = {"txid", "observer_id", "timestamp_ms"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing columns: {sorted(missing)}")

    if observer_ids is None:
        observer_ids = sorted(observations["observer_id"].unique().tolist())
    if not observer_ids:
        raise ValueError("no observers: cannot build arrival vectors")

    # Earliest sighting per (transaction, observer); a peer re-advertising
    # does not tell us anything new about propagation.
    before = int(len(observations))
    earliest = (
        observations.groupby(["txid", "observer_id"], sort=True)["timestamp_ms"]
        .min()
        .reset_index()
    )
    duplicates = before - int(len(earliest))

    txids = np.sort(observations["txid"].unique())
    tx_pos = pd.Index(txids)
    obs_pos = pd.Index(observer_ids)

    absolute = np.full((txids.size, len(observer_ids)), np.nan, dtype=np.float64)
    rows = tx_pos.get_indexer(earliest["txid"])
    cols = obs_pos.get_indexer(earliest["observer_id"])
    keep = cols >= 0  # ignore observers outside the requested axis
    absolute[rows[keep], cols[keep]] = earliest["timestamp_ms"].to_numpy()[keep]

    n_observed = np.count_nonzero(~np.isnan(absolute), axis=1)
    with np.errstate(invalid="ignore"):
        first_ms = np.nanmin(
            np.where(np.isnan(absolute), np.inf, absolute), axis=1
        )
        last_ms = np.nanmax(
            np.where(np.isnan(absolute), -np.inf, absolute), axis=1
        )
    seen = n_observed > 0
    first_ms = np.where(seen, first_ms, np.nan)
    last_ms = np.where(seen, last_ms, np.nan)

    offsets = absolute - first_ms[:, None]
    spread = np.where(seen, last_ms - first_ms, np.nan)

    ranks = _rank_rows(absolute)
    first_index = np.full(txids.size, -1, dtype=np.int64)
    has_first = np.any(ranks == 0, axis=1)
    first_index[has_first] = np.argmax(ranks == 0, axis=1)[has_first]
    first_observer = np.array(
        [observer_ids[i] if i >= 0 else "" for i in first_index], dtype=object
    )

    return ArrivalVectors(
        txids=txids,
        observer_ids=list(observer_ids),
        absolute_ms=absolute,
        offsets_ms=offsets,
        ranks=ranks,
        first_observer=first_observer,
        spread_ms=spread,
        n_observed=n_observed,
        duplicates_collapsed=duplicates,
    )


def _rank_rows(absolute: np.ndarray) -> np.ndarray:
    """Rank observers by arrival within each row; -1 where unobserved.

    Ties are broken by observer index so the output is deterministic. Ties
    are common in real captures at millisecond resolution, and an arbitrary
    but stable order is preferable to a nondeterministic one.
    """
    filled = np.where(np.isnan(absolute), np.inf, absolute)
    order = np.argsort(filled, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(absolute.shape[0])[:, None]
    ranks[rows, order] = np.arange(absolute.shape[1])[None, :]
    return np.where(np.isnan(absolute), -1, ranks).astype(np.int64)


def classify_evidence(
    vectors: ArrivalVectors,
    broadcaster_txids: set | None = None,
    threshold_ms: float = DEFAULT_FLAT_THRESHOLD_MS,
) -> pd.DataFrame:
    """Label each transaction USABLE or NO_EVIDENCE, with a reason.

    Every input to this decision is observable: the spread comes from our own
    timestamps, and broadcaster membership from a public IP list. Nothing here
    consults the generator.

    A transaction is NO_EVIDENCE when any of the following holds - the checks
    are ordered so the reported reason is the most fundamental one:

    * it was relayed by a known broadcaster, so every observer heard it
      directly and near-simultaneously;
    * its spread lies within total clock error, so the apparent ordering is
      indistinguishable from unsynchronised clocks;
    * fewer than two observers saw it, so there is nothing to order.
    """
    n = vectors.n_transactions
    reason = np.full(n, "", dtype=object)

    too_few = vectors.n_observed < 2
    flat = vectors.spread_ms <= threshold_ms
    flat = np.where(np.isnan(vectors.spread_ms), True, flat)

    if broadcaster_txids:
        is_broadcaster = np.isin(vectors.txids, list(broadcaster_txids))
    else:
        is_broadcaster = np.zeros(n, dtype=bool)

    reason[flat] = "flat_vector"
    reason[too_few] = "too_few_observers"
    reason[is_broadcaster] = "known_broadcaster"

    usable = reason == ""
    return pd.DataFrame(
        {
            "txid": vectors.txids,
            "evidence": np.where(usable, Evidence.USABLE.value, Evidence.NO_EVIDENCE.value),
            "no_evidence_reason": reason,
        }
    )


def assert_no_constraint_permitted(evidence: pd.DataFrame, txids) -> None:
    """Raise if any of ``txids`` is NO_EVIDENCE.

    Call this from a constraint builder before emitting links. It converts a
    silent logical error - a constraint drawn from a vector that says nothing
    - into a loud one.
    """
    blocked = evidence.loc[
        evidence["evidence"] == Evidence.NO_EVIDENCE.value, "txid"
    ]
    offending = set(np.asarray(txids).tolist()) & set(blocked.tolist())
    if offending:
        raise ValueError(
            f"{len(offending)} transaction(s) marked NO_EVIDENCE were used to "
            f"derive a constraint, e.g. {sorted(offending)[:5]}. A vector with "
            f"no ordering information cannot support a link of either kind."
        )


def summarise(
    vectors: ArrivalVectors,
    broadcaster_txids: set | None = None,
    threshold_ms: float = DEFAULT_FLAT_THRESHOLD_MS,
) -> str:
    """Describe the arrival vectors, without inferring anything from them."""
    width = 78
    out: list[str] = []
    add = out.append
    add("=" * width)
    add("obsidianchain :: arrival-time vectors")
    add("=" * width)
    add("  Instrumentation only - reshapes observations, infers nothing.")
    add("  Built on SYNTHETIC announcements; see network.synthetic docstring.")
    add("")
    add("-- shape " + "-" * (width - 10))
    add(f"  transactions                    {vectors.n_transactions:>12,}")
    add(f"  observers                       {vectors.n_observers:>12,}")
    add(f"  duplicate sightings collapsed   {vectors.duplicates_collapsed:>12,}")
    add(f"  observed by all observers       "
        f"{int((vectors.n_observed == vectors.n_observers).sum()):>12,}")
    add("")

    spread = vectors.spread_ms[~np.isnan(vectors.spread_ms)]
    add("-- spread, last arrival minus first (ms) " + "-" * (width - 41))
    if spread.size:
        for label, value in (
            ("min", spread.min()),
            ("p25", np.percentile(spread, 25)),
            ("median", np.median(spread)),
            ("p75", np.percentile(spread, 75)),
            ("p95", np.percentile(spread, 95)),
            ("max", spread.max()),
        ):
            add(f"  {label:<8}{value:>14,.1f}")
    add("")

    flat = vectors.flag_flat(threshold_ms)
    add("-- flat vectors, no ordering information " + "-" * (width - 42))
    add(f"  threshold (ms)                  {threshold_ms:>12,.0f}")
    add(f"  flat transactions               {int(flat.sum()):>12,}"
        f"   {flat.mean() * 100:.1f}%")
    add("  These carry no timing signal and must yield no constraint.")
    if broadcaster_txids is not None:
        is_bc = np.isin(vectors.txids, list(broadcaster_txids))
        add("")
        add(f"  known-broadcaster transactions  {int(is_bc.sum()):>12,}")
        if is_bc.any():
            add(f"    ...of which flat              {int(flat[is_bc].sum()):>12,}"
                f"   {flat[is_bc].mean() * 100:.1f}%")
        if (~is_bc).any():
            add(f"    non-broadcaster flat          {int(flat[~is_bc].sum()):>12,}"
                f"   {flat[~is_bc].mean() * 100:.1f}%")
    add("")

    add("-- first-observer distribution " + "-" * (width - 31))
    counts = pd.Series(vectors.first_observer).value_counts()
    for observer, count in counts.head(12).items():
        if not observer:
            continue
        add(f"  {str(observer):<24}{count:>12,}"
            f"   {count / vectors.n_transactions * 100:5.1f}%")
    add("=" * width)
    return "\n".join(out)


def write(vectors: ArrivalVectors, path: Path, fmt: str = "parquet") -> int:
    """Write the wide arrival-vector table. Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = vectors.to_frame()
    if fmt == "parquet":
        frame.to_parquet(path, index=False)
    elif fmt == "csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported format {fmt!r}; use 'parquet' or 'csv'")
    return int(len(frame))


def read_observations(path: Path) -> pd.DataFrame:
    """Read announcement records from parquet or CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'make run ARGS=\"network-generate\"' first."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
