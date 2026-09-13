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
import ipaddress
from dataclasses import dataclass
from pathlib import Path

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
        return []
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

    for network, iso, country in (database or []):
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


def summarise(ips, asns, data_root=None) -> dict:
    """Aggregate facts for a set of addresses and ASNs.

    Used by the alert layer to describe an alert's network footprint without
    per-record lookups in a request handler.
    """
    database = load_geoip_database(data_root)
    ip_facts = [resolve_ip(ip, database) for ip in dict.fromkeys(ips)]
    asn_facts = [resolve_asn(a) for a in dict.fromkeys(asns)]
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
        "geoip_database_installed": bool(database),
    }
