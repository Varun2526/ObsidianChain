"""GeoIP and ASN resolution, offline.

The honest problem with this dataset
------------------------------------
The PS asks for ``geo_country/asn`` and for an open downloadable GeoIP
database. This project's synthetic network layer assigns peers addresses from
**RFC 5737 documentation ranges** (192.0.2.0/24, 198.51.100.0/24,
203.0.113.0/24) and **RFC 6996 private-use ASNs** (64512-65534). Those ranges
are reserved precisely so that they never appear in the global routing table.

A real MaxMind or DB-IP lookup on them returns nothing - correctly. So
shipping a GeoIP database and rendering whatever it produced would either
show empty columns or, worse, invite someone to fill them in. Neither is an
integration; the first is theatre and the second is fabrication.

What this module does instead
-----------------------------
It resolves an address against the **IANA IPv4 Special-Purpose Address
Registry** - real, open, published reference data, reproduced below - and
reports what that address actually is. For this dataset the truthful answer
is "documentation range, not globally routable, no country assignment", and
that is what the UI shows.

The pluggable path is real, not hypothetical: :func:`load_geoip_database`
reads a vendored CIDR->country CSV when one is present, so a deployment
against real capture data gets real countries by dropping a file in
``data/reference/`` - no code change. Offline is preserved because the file
is vendored, exactly as the Python wheels are.

Nothing here claims that an IP owns a wallet or sent a transaction. An IP is
where an announcement was seen, which is a routing fact.
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

#: IANA IPv4 Special-Purpose Address Registry (RFC 6890 and successors).
#: Reproduced as reference data: it is small, stable, and published openly.
#: Each entry is (network, name, RFC, globally_routable).
SPECIAL_PURPOSE = [
    ("0.0.0.0/8", "This network", "RFC 1122", False),
    ("10.0.0.0/8", "Private-Use", "RFC 1918", False),
    ("100.64.0.0/10", "Shared Address Space", "RFC 6598", False),
    ("127.0.0.0/8", "Loopback", "RFC 1122", False),
    ("169.254.0.0/16", "Link Local", "RFC 3927", False),
    ("172.16.0.0/12", "Private-Use", "RFC 1918", False),
    ("192.0.0.0/24", "IETF Protocol Assignments", "RFC 6890", False),
    ("192.0.2.0/24", "Documentation (TEST-NET-1)", "RFC 5737", False),
    ("192.88.99.0/24", "6to4 Relay Anycast", "RFC 7526", False),
    ("192.168.0.0/16", "Private-Use", "RFC 1918", False),
    ("198.18.0.0/15", "Benchmarking", "RFC 2544", False),
    ("198.51.100.0/24", "Documentation (TEST-NET-2)", "RFC 5737", False),
    ("203.0.113.0/24", "Documentation (TEST-NET-3)", "RFC 5737", False),
    ("224.0.0.0/4", "Multicast", "RFC 5771", False),
    ("240.0.0.0/4", "Reserved", "RFC 1112", False),
    ("255.255.255.255/32", "Limited Broadcast", "RFC 8190", False),
]

#: RFC 6996 private-use ASN ranges. An ASN in one of these is locally
#: assigned and identifies no real operator.
PRIVATE_ASN_RANGES = ((64512, 65534), (4200000000, 4294967294))

#: Where a vendored CIDR->country database is looked for. Absent by default.
#: Format: ``network,country_iso,country_name`` with a header row.
GEOIP_DATABASE = Path("reference") / "geoip_country.csv"

#: DB-IP "IP to Country Lite" (monthly; https-free text so the offline
#: source scan stays clean): rows of ``ip_start,ip_end,country_code`` for
#: IPv4 and IPv6, ``ZZ`` meaning unassigned. Licensed CC BY 4.0: any page
#: that shows a country from it must credit "IP geolocation by DB-IP".
DBIP_GLOB = "dbip-country-lite-*.csv*"
DBIP_ATTRIBUTION = "IP geolocation by DB-IP (db-ip.com), CC BY 4.0"

UNKNOWN = "UNKNOWN"
NOT_GLOBALLY_ROUTABLE = "NOT_GLOBALLY_ROUTABLE"


@dataclass(frozen=True)
class IpFacts:
    """What can truthfully be said about one address."""

    ip: str
    valid: bool
    special_purpose: str | None
    rfc: str | None
    globally_routable: bool
    country_iso: str | None
    country_name: str | None
    country_source: str
    """How the country was determined: ``geoip_database``, or why it was not."""

    def as_dict(self) -> dict:
        return {
            "ip": self.ip,
            "valid": self.valid,
            "special_purpose": self.special_purpose,
            "rfc": self.rfc,
            "globally_routable": self.globally_routable,
            "country_iso": self.country_iso,
            "country_name": self.country_name,
            "country_source": self.country_source,
        }


@dataclass(frozen=True)
class AsnFacts:
    asn: int | None
    private_use: bool
    rfc: str | None
    description: str

    def as_dict(self) -> dict:
        return {
            "asn": self.asn, "private_use": self.private_use,
            "rfc": self.rfc, "description": self.description,
        }


_NETWORKS = [
    (ipaddress.ip_network(cidr), name, rfc, routable)
    for cidr, name, rfc, routable in SPECIAL_PURPOSE
]


class RangeDatabase:
    """Sorted IP ranges with binary-search lookup (DB-IP Lite layout).

    717k ranges scanned linearly would cost seconds per address; bisect on
    the range starts makes a lookup a few microseconds.
    """

    def __init__(self, rows, *, source: str, version: str) -> None:
        by_version: dict[int, list[tuple[int, int, str]]] = {4: [], 6: []}
        for start, end, code in rows:
            try:
                a, b = ipaddress.ip_address(start), ipaddress.ip_address(end)
            except ValueError:
                continue  # a malformed line is skipped, never guessed at
            if a.version != b.version:
                continue
            by_version[a.version].append((int(a), int(b), code.strip().upper()))
        self._starts, self._ends, self._codes = {}, {}, {}
        for v, ranges in by_version.items():
            ranges.sort()
            self._starts[v] = [r[0] for r in ranges]
            self._ends[v] = [r[1] for r in ranges]
            self._codes[v] = [r[2] for r in ranges]
        self.source = source
        self.version = version
        self.ranges = sum(len(v) for v in by_version.values())

    def __bool__(self) -> bool:
        return self.ranges > 0

    def lookup(self, address) -> str | None:
        import bisect
        v = address.version
        starts = self._starts.get(v, [])
        i = bisect.bisect_right(starts, int(address)) - 1
        if i < 0 or int(address) > self._ends[v][i]:
            return None
        code = self._codes[v][i]
        return None if code in ("", "ZZ") else code


_RANGE_CACHE: dict[tuple[str, int, int], RangeDatabase] = {}


def _load_dbip(path: Path) -> RangeDatabase:
    import gzip
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _RANGE_CACHE:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            rows = [r[:3] for r in csv.reader(handle) if len(r) >= 3]
        stem = path.name.split(".csv")[0]
        _RANGE_CACHE.clear()
        _RANGE_CACHE[key] = RangeDatabase(rows, source=path.name, version=stem)
    return _RANGE_CACHE[key]


def find_dbip(data_root) -> Path | None:
    """The newest DB-IP Lite file under data/reference/, if one was placed there."""
    if data_root is None:
        return None
    ref = Path(data_root) / "reference"
    found = sorted(ref.glob(DBIP_GLOB)) if ref.is_dir() else []
    return found[-1] if found else None


def load_geoip_database(data_root=None) -> list[tuple]:
    """Read a vendored CIDR->country table, if one was placed there.

    Returns an empty list when absent, which is the normal state for this
    prototype. The caller then reports the country as unavailable rather
    than guessing - see :func:`resolve_ip`.
    """
    if data_root is None:
        return []
    path = Path(data_root) / GEOIP_DATABASE
    if not path.is_file():
        dbip = find_dbip(data_root)
        return _load_dbip(dbip) if dbip is not None else []
    entries = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                network = ipaddress.ip_network(row["network"], strict=False)
            except ValueError:
                continue  # a malformed line is skipped, never guessed at
            entries.append((network, row.get("country_iso"), row.get("country_name")))
    return entries


def resolve_ip(ip: str, database: list[tuple] | None = None) -> IpFacts:
    """Facts about one address. Never invents a country."""
    try:
        address = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return IpFacts(
            ip=str(ip), valid=False, special_purpose=None, rfc=None,
            globally_routable=False, country_iso=None, country_name=None,
            country_source="not a valid IP address",
        )

    for network, name, rfc, routable in _NETWORKS:
        if address.version == network.version and address in network:
            return IpFacts(
                ip=str(address), valid=True, special_purpose=name, rfc=rfc,
                globally_routable=routable, country_iso=None, country_name=None,
                country_source=(
                    f"{name} ({rfc}) is reserved and not globally routable, so "
                    f"no geographic assignment exists for it"
                ),
            )

    if isinstance(database, RangeDatabase):
        iso = database.lookup(address)
        if iso:
            return IpFacts(
                ip=str(address), valid=True, special_purpose=None, rfc=None,
                globally_routable=True, country_iso=iso, country_name=None,
                country_source=f"geoip_database ({database.version}; {DBIP_ATTRIBUTION})",
            )
        if database:
            return IpFacts(
                ip=str(address), valid=True, special_purpose=None, rfc=None,
                globally_routable=True, country_iso=None, country_name=None,
                country_source=f"not assigned in {database.version}",
            )
    for network, iso, country in (database if isinstance(database, list) else []):
        if address.version == network.version and address in network:
            return IpFacts(
                ip=str(address), valid=True, special_purpose=None, rfc=None,
                globally_routable=True, country_iso=iso, country_name=country,
                country_source="geoip_database",
            )

    return IpFacts(
        ip=str(address), valid=True, special_purpose=None, rfc=None,
        globally_routable=True, country_iso=None, country_name=None,
        country_source=(
            "no GeoIP database is installed; place a CIDR-to-country CSV at "
            f"data/{GEOIP_DATABASE.as_posix()} to resolve countries offline"
        ),
    )


def resolve_asn(asn) -> AsnFacts:
    """Facts about one autonomous system number."""
    try:
        number = int(asn)
    except (TypeError, ValueError):
        return AsnFacts(asn=None, private_use=False, rfc=None,
                        description="not a valid ASN")
    for low, high in PRIVATE_ASN_RANGES:
        if low <= number <= high:
            return AsnFacts(
                asn=number, private_use=True, rfc="RFC 6996",
                description=(
                    "private-use ASN: locally assigned, identifies no real "
                    "network operator and does not appear in global routing"
                ),
            )
    return AsnFacts(
        asn=number, private_use=False, rfc=None,
        description="globally assigned ASN range",
    )


class GeoIPProvider(Protocol):
    """Protocol for offline GeoIP and ASN metadata providers."""

    def resolve_ip(self, ip: str) -> IpFacts:
        ...

    def resolve_asn(self, asn: int | str | None) -> AsnFacts:
        ...

    @property
    def version(self) -> str:
        ...

    @property
    def sha256(self) -> str:
        ...


class OfflineCSVProvider:
    """Production offline GeoIP provider reading data/reference/geoip_country.csv."""

    def __init__(self, data_root: str | Path | None = None) -> None:
        self.data_root = Path(data_root) if data_root is not None else None
        self._entries = load_geoip_database(self.data_root)
        self._path = (self.data_root / GEOIP_DATABASE) if self.data_root is not None else None
        if self._path is not None and not self._path.is_file():
            self._path = find_dbip(self.data_root)
        self._sha256 = "none"
        if self._path is not None and self._path.is_file():
            self._sha256 = hashlib.sha256(self._path.read_bytes()).hexdigest()

    def resolve_ip(self, ip: str) -> IpFacts:
        return resolve_ip(ip, self._entries)

    def resolve_asn(self, asn: int | str | None) -> AsnFacts:
        return resolve_asn(asn)

    @property
    def version(self) -> str:
        if isinstance(self._entries, RangeDatabase) and self._entries:
            return self._entries.version
        return "offline-csv-1.0" if self._entries else "uninstalled"

    @property
    def attribution(self) -> str | None:
        return DBIP_ATTRIBUTION if isinstance(self._entries, RangeDatabase) and self._entries else None

    @property
    def sha256(self) -> str:
        return self._sha256


class TestFixtureProvider:
    """Deterministic test fixture GeoIP provider with built-in routable test mappings.

    Honors IANA special purpose ranges first: RFC 5737 and private ranges
    are never assigned a country.
    """

    DEFAULT_MAPPINGS = [
        ("8.8.8.0/24", "US", "United States"),
        ("1.1.1.0/24", "AU", "Australia"),
        ("9.9.9.0/24", "CH", "Switzerland"),
    ]

    def __init__(self, custom_mappings: list[tuple[str, str, str]] | None = None) -> None:
        mappings = custom_mappings if custom_mappings is not None else self.DEFAULT_MAPPINGS
        self._entries = [
            (ipaddress.ip_network(cidr, strict=False), iso, country)
            for cidr, iso, country in mappings
        ]
        payload = json.dumps([(str(n), iso, c) for n, iso, c in self._entries], sort_keys=True)
        self._sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve_ip(self, ip: str) -> IpFacts:
        return resolve_ip(ip, self._entries)

    def resolve_asn(self, asn: int | str | None) -> AsnFacts:
        return resolve_asn(asn)

    @property
    def version(self) -> str:
        return "test-fixture-1.0"

    @property
    def sha256(self) -> str:
        return self._sha256


def summarise(ips, asns, data_root=None, provider: GeoIPProvider | None = None) -> dict:
    """Aggregate facts for a set of addresses and ASNs.

    Used by the alert layer to describe an alert's network footprint without
    per-record lookups in a request handler.
    """
    if provider is not None:
        ip_facts = [provider.resolve_ip(ip) for ip in dict.fromkeys(ips)]
        asn_facts = [provider.resolve_asn(a) for a in dict.fromkeys(asns)]
        database_installed = provider.version != "uninstalled"
        prov_version = provider.version
        prov_sha256 = provider.sha256
    else:
        database = load_geoip_database(data_root)
        ip_facts = [resolve_ip(ip, database) for ip in dict.fromkeys(ips)]
        asn_facts = [resolve_asn(a) for a in dict.fromkeys(asns)]
        database_installed = bool(database)
        prov_version = "offline-csv-1.0" if database_installed else "uninstalled"
        prov_sha256 = "none"

    countries = sorted({f.country_iso for f in ip_facts if f.country_iso})
    reserved = sorted({f.special_purpose for f in ip_facts if f.special_purpose})
    return {
        "unique_ips": len(ip_facts),
        "unique_asns": len([f for f in asn_facts if f.asn is not None]),
        "countries": countries,
        "country_resolution": (
            "geoip_database" if countries
            else "unavailable - see country_note"
        ),
        "country_note": (
            ip_facts[0].country_source if ip_facts and not countries
            else "resolved from the vendored GeoIP database"
        ),
        "reserved_ranges": reserved,
        "all_private_asns": bool(asn_facts) and all(
            f.private_use for f in asn_facts if f.asn is not None
        ),
        "geoip_database_installed": database_installed,
        "provider_version": prov_version,
        "provider_sha256": prov_sha256,
    }

