"""Propagation and peer context from the network observations in a capture.

Works on the observations a real upload carries (``src_ip``, ``src_port``,
``dst_ip``, ``timestamp``, ``asn``, ``geo_country``, optional ``observer_id``),
after the pipeline has joined them to transactions by txid
(``correlation/engine.py``). Nothing here reads the synthetic overlay.

What each observation is taken to mean
--------------------------------------
``src_ip`` is the peer that announced the transaction to the observer. It is
a relay vantage point: with gossip relay, NAT, Tor and VPNs it is never the
sender, and every output here says "peer", not "origin".

The observer is ``observer_id`` when the capture has one. Without it, the
observer is taken to be ``dst_ip`` (the sensor that received the
announcement), and ``observer_source`` says so. Without either, observers
are unknown and observer counts are reported as ``None``, not zero.

What is computed, per transaction
---------------------------------
first/last seen and spread (only from observations with a timestamp),
first-seen peer(s) and observer(s) (ties kept, never broken arbitrarily),
per-peer first arrival and observation count, peer and ASN diversity,
dominant-peer share, capture-supplied countries, and the share of peer IPs
that are not globally routable (RFC 1918, RFC 5737, ...).

Per alert cluster, the same quantities are pooled over its transactions.

These are investigation evidence. They are not model features and are not
fused into the risk score: no labelled data with real network fields
exists to show they predict anything (see
``docs/plans/2026-09-25-network-layer.md``).
"""

from __future__ import annotations

import datetime as _dt
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from obsidianchain import geoip as _geoip

SCHEMA = "obsidianchain.network_propagation/1"

#: Plausible epoch range for a Bitcoin capture (2009-01-01 .. 2100-01-01),
#: used to decide whether a bare number is seconds or milliseconds.
_EPOCH_S_MIN, _EPOCH_S_MAX = 1_230_768_000, 4_102_444_800


def timestamp_ms(value: Any) -> int | None:
    """Epoch milliseconds, or None when the value cannot be read as a time.

    Accepts epoch seconds or milliseconds (as numbers or numeric strings) and
    ISO-8601 strings. A number outside a plausible Bitcoin-era range in both
    units is rejected rather than guessed at.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>", "null"):
        return None
    try:
        number = float(text)
    except ValueError:
        try:
            parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return int(round(parsed.timestamp() * 1000))
    if _EPOCH_S_MIN <= number <= _EPOCH_S_MAX:
        return int(round(number * 1000))
    if _EPOCH_S_MIN * 1000 <= number <= _EPOCH_S_MAX * 1000:
        return int(round(number))
    return None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() not in ("nan", "none", "<na>", "null") else None


def _asn(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if 0 <= number < 2 ** 32 else None


@dataclass
class PeerArrival:
    peer_ip: str
    first_seen_ms: int | None
    observations: int
    observers: list[str]
    asns: list[int]
    ip_class: str
    """``global``, or the IANA special-purpose name (e.g. ``Private-Use``)."""
    country_iso: str | None = None
    """Resolved by the offline GeoIP provider, when one is installed."""

    def as_dict(self) -> dict:
        return {"peer_ip": self.peer_ip, "first_seen_ms": self.first_seen_ms,
                "observations": self.observations, "observers": self.observers,
                "asns": self.asns, "ip_class": self.ip_class, "country_iso": self.country_iso}


@dataclass
class TxPropagation:
    txid: str
    observations: int
    timed_observations: int
    observer_source: str
    observers: list[str] | None
    first_seen_ms: int | None
    last_seen_ms: int | None
    spread_ms: int | None
    first_seen_peers: list[str]
    first_seen_observers: list[str]
    peers: list[PeerArrival] = field(default_factory=list)
    asns: list[int] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    dominant_peer_ip: str | None = None
    dominant_peer_share: float | None = None
    non_routable_peer_share: float | None = None
    resolved_countries: list[str] = field(default_factory=list)
    country_resolution: str | None = None

    def as_dict(self) -> dict:
        return {
            "txid": self.txid,
            "observations": self.observations,
            "timed_observations": self.timed_observations,
            "observer_source": self.observer_source,
            "observers": self.observers,
            "observer_count": None if self.observers is None else len(self.observers),
            "first_seen_ms": self.first_seen_ms,
            "last_seen_ms": self.last_seen_ms,
            "spread_ms": self.spread_ms,
            "first_seen_peers": self.first_seen_peers,
            "first_seen_observers": self.first_seen_observers,
            "peer_count": len(self.peers),
            "peers": [p.as_dict() for p in self.peers],
            "asns": self.asns,
            "asn_count": len(self.asns),
            "countries": self.countries,
            "country_source": "capture-supplied geo_country (unverified)" if self.countries else None,
            "dominant_peer_ip": self.dominant_peer_ip,
            "dominant_peer_share": self.dominant_peer_share,
            "non_routable_peer_share": self.non_routable_peer_share,
            "resolved_countries": self.resolved_countries,
            "country_resolution": self.country_resolution,
        }


def _observer_source(observations) -> str:
    if any(_clean(o.observer_id) for o in observations):
        return "observer_id"
    if any(_clean(o.dst_ip) for o in observations):
        return "dst_ip (no observer_id in capture)"
    return "unknown"


def _observer_of(obs, source: str) -> str | None:
    if source == "observer_id":
        return _clean(obs.observer_id)
    if source.startswith("dst_ip"):
        return _clean(obs.dst_ip)
    return None


def _resolved_country(ip: str, provider) -> str | None:
    if provider is None or getattr(provider, "version", "uninstalled") == "uninstalled":
        return None
    return provider.resolve_ip(ip).country_iso


def _ip_class(ip: str) -> str:
    facts = _geoip.resolve_ip(ip)
    if not facts.valid:
        return "invalid"
    return "global" if facts.globally_routable and facts.special_purpose is None else (facts.special_purpose or "not-routable")


def analyse_transaction(txid: str, observations: list, geoip_provider=None) -> TxPropagation:
    """Propagation facts for one transaction's observations.

    ``geoip_provider`` (optional) resolves each peer's country offline. Its
    result is kept apart from the capture's own ``geo_country`` so a reader
    can see which claim came from where.
    """
    source = _observer_source(observations)
    timed = [(timestamp_ms(o.timestamp), o) for o in observations]
    times = [t for t, _ in timed if t is not None]
    first = min(times) if times else None
    last = max(times) if times else None

    by_peer: dict[str, list] = {}
    for t, o in timed:
        ip = _clean(o.src_ip)
        if ip:
            by_peer.setdefault(ip, []).append((t, o))

    peers = []
    for ip, rows in by_peer.items():
        peer_times = [t for t, _ in rows if t is not None]
        observers = sorted({x for x in (_observer_of(o, source) for _, o in rows) if x})
        peers.append(PeerArrival(
            peer_ip=ip,
            first_seen_ms=min(peer_times) if peer_times else None,
            observations=len(rows),
            observers=observers,
            asns=sorted({a for a in (_asn(o.asn) for _, o in rows) if a is not None}),
            ip_class=_ip_class(ip),
            country_iso=_resolved_country(ip, geoip_provider),
        ))
    peers.sort(key=lambda p: (p.first_seen_ms is None, p.first_seen_ms or 0, p.peer_ip))

    first_peers = sorted({_clean(o.src_ip) for t, o in timed if first is not None and t == first and _clean(o.src_ip)})
    first_observers = sorted({x for x in (_observer_of(o, source) for t, o in timed
                                          if first is not None and t == first) if x})
    observers = None if source == "unknown" else sorted({x for x in (_observer_of(o, source) for o in observations) if x})

    counts = Counter(_clean(o.src_ip) for o in observations if _clean(o.src_ip))
    dominant_ip, dominant_n = (counts.most_common(1)[0] if counts else (None, 0))
    with_ip = sum(counts.values())
    non_routable = [p for p in peers if p.ip_class != "global"]

    resolution = None
    if geoip_provider is not None and getattr(geoip_provider, "version", "uninstalled") != "uninstalled":
        resolution = f"offline GeoIP {geoip_provider.version}"
        attribution = getattr(geoip_provider, "attribution", None)
        if attribution:
            resolution += f" ({attribution})"
    return TxPropagation(
        resolved_countries=sorted({p.country_iso for p in peers if p.country_iso}),
        country_resolution=resolution,
        txid=txid,
        observations=len(observations),
        timed_observations=len(times),
        observer_source=source,
        observers=observers,
        first_seen_ms=first,
        last_seen_ms=last,
        spread_ms=(last - first) if len(times) >= 2 else None,
        first_seen_peers=first_peers,
        first_seen_observers=first_observers,
        peers=peers,
        asns=sorted({a for a in (_asn(o.asn) for o in observations) if a is not None}),
        countries=sorted({c.upper() for c in (_clean(o.geo_country) for o in observations) if c}),
        dominant_peer_ip=dominant_ip,
        dominant_peer_share=(dominant_n / with_ip) if with_ip else None,
        non_routable_peer_share=(len(non_routable) / len(peers)) if peers else None,
    )


@dataclass
class PropagationResult:
    transactions: dict[str, TxPropagation]

    def for_txids(self, txids: Iterable[str]) -> list[TxPropagation]:
        return [self.transactions[t] for t in txids if t in self.transactions]

    def pooled(self, txids: Iterable[str]) -> dict | None:
        """The same quantities pooled over a set of transactions (an alert cluster)."""
        rows = self.for_txids(txids)
        if not rows:
            return None
        peer_obs: Counter = Counter()
        peer_first: dict[str, int] = {}
        observers: set[str] = set()
        observer_known = False
        asns: set[int] = set()
        countries: set[str] = set()
        resolved: set[str] = set()
        resolution = None
        classes: dict[str, str] = {}
        for r in rows:
            for p in r.peers:
                peer_obs[p.peer_ip] += p.observations
                classes[p.peer_ip] = p.ip_class
                if p.first_seen_ms is not None:
                    peer_first[p.peer_ip] = min(peer_first.get(p.peer_ip, p.first_seen_ms), p.first_seen_ms)
            if r.observers is not None:
                observer_known = True
                observers.update(r.observers)
            asns.update(r.asns)
            countries.update(r.countries)
            resolved.update(r.resolved_countries)
            resolution = resolution or r.country_resolution
        total = sum(peer_obs.values())
        top_ip, top_n = peer_obs.most_common(1)[0] if peer_obs else (None, 0)
        spreads = sorted(r.spread_ms for r in rows if r.spread_ms is not None)
        firsts = [r.first_seen_ms for r in rows if r.first_seen_ms is not None]
        return {
            "transactions_observed": len(rows),
            "observations": sum(r.observations for r in rows),
            "peer_count": len(peer_obs),
            "observer_count": len(observers) if observer_known else None,
            "asn_count": len(asns),
            "asns": sorted(asns),
            "countries": sorted(countries),
            "country_source": "capture-supplied geo_country (unverified)" if countries else None,
            "resolved_countries": sorted(resolved),
            "country_resolution": resolution,
            "dominant_peer_ip": top_ip,
            "dominant_peer_share": (top_n / total) if total else None,
            "non_routable_peer_share": (sum(1 for c in classes.values() if c != "global") / len(classes)) if classes else None,
            "earliest_first_seen_ms": min(firsts) if firsts else None,
            "median_spread_ms": spreads[len(spreads) // 2] if spreads else None,
            "top_peers": [{"peer_ip": ip, "observations": n, "first_seen_ms": peer_first.get(ip),
                           "ip_class": classes.get(ip)} for ip, n in peer_obs.most_common(10)],
        }

    def summary(self) -> dict:
        rows = list(self.transactions.values())
        sources = Counter(r.observer_source for r in rows)
        spreads = sorted(r.spread_ms for r in rows if r.spread_ms is not None)
        return {
            "transactions_with_observations": len(rows),
            "observations": sum(r.observations for r in rows),
            "with_timing": sum(1 for r in rows if r.first_seen_ms is not None),
            "with_spread": len(spreads),
            "median_spread_ms": spreads[len(spreads) // 2] if spreads else None,
            "distinct_peers": len({p.peer_ip for r in rows for p in r.peers}),
            "distinct_asns": len({a for r in rows for a in r.asns}),
            "observer_source": dict(sources),
            "distinct_resolved_countries": len({c for r in rows for c in r.resolved_countries}),
            "country_resolution": next((r.country_resolution for r in rows if r.country_resolution), None),
        }

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "meaning": ("Per-transaction propagation from the capture's own network observations. "
                        "A peer is the IP that announced a transaction to an observer: a relay vantage "
                        "point, never the sender. Evidence only; not used by the risk model."),
            "summary": self.summary(),
            "transactions": [r.as_dict() for r in sorted(self.transactions.values(), key=lambda r: r.txid)],
        }


def analyse(correlation_result, geoip_provider=None) -> PropagationResult:
    """Propagation for every transaction that has observations (correlated or not)."""
    grouped: dict[str, list] = {}
    for obs in correlation_result.all_observations:
        grouped.setdefault(obs.txid, []).append(obs)
    return PropagationResult({txid: analyse_transaction(txid, obs, geoip_provider)
                              for txid, obs in grouped.items()})
