"""Generate the synthetic demonstration capture (data/samples/demo_capture_synthetic.csv).

A small, deterministic capture in the canonical upload schema for showing
the investigator workflow end to end. It is SYNTHETIC: every address is
derived from a seed (no private key exists for any of them), every amount
and time is made up, and the scenario is written by hand. It is not
evidence about anyone and is not used for training or evaluation.

What it contains
----------------
- Ordinary payments between ordinary wallets (1-2 inputs, payment + change).
- A laundering pattern, end to end, so a money-flow path exists to trace:
  a source wallet peels value off in six hops into deposit addresses;
  those and other deposits are consolidated by one many-input transaction;
  the collector fans the value out; five of those outputs enter a
  CoinJoin-style transaction with equal-valued outputs.
- Network observations for every transaction from three named observers
  (``observer_id``), 2-4 announcing peers each, with millisecond arrival
  times. Peer IPs are drawn from public ranges in the vendored DB-IP Lite
  file so country resolution has something to resolve; ASNs are
  private-use numbers (64512-65534) because the capture is synthetic.
  A peer is a relay vantage point, never the sender.
- Two deliberate validation findings: one non-numeric fee, one exact
  duplicate record.

Run: python3 scripts/make_demo_capture.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import ipaddress
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "demo_capture_synthetic.csv"
DBIP = next((ROOT / "data" / "reference").glob("dbip-country-lite-*.csv.gz"))
SEED = 20260925
START = 1_789_977_600  # 2026-09-21 08:00:00 UTC
COLUMNS = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid", "input_addresses",
           "output_addresses", "input_amounts", "output_amounts", "fee", "script_type",
           "geo_country", "asn", "observer_id"]
OBSERVERS = {"obs-fra-1": "198.51.100.11", "obs-sgp-1": "198.51.100.12", "obs-nyc-1": "198.51.100.13"}
COUNTRIES = ["DE", "NL", "FR", "GB", "US", "CA", "SG", "JP", "IN"]

rng = random.Random(SEED)
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58check(payload: bytes) -> str:
    data = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n = int.from_bytes(data, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    return "1" * (len(data) - len(data.lstrip(b"\0"))) + out


def address(label: str) -> str:
    """A well-formed P2PKH address from a label. No key exists for it."""
    return _b58check(b"\x00" + hashlib.sha256(f"obsidianchain-demo:{label}".encode()).digest()[:20])


def txid(label: str) -> str:
    return hashlib.sha256(f"obsidianchain-demo-tx:{label}".encode()).hexdigest()


def _peer_pool() -> list[str]:
    ranges: dict[str, list[tuple[int, int]]] = {c: [] for c in COUNTRIES}
    with gzip.open(DBIP, "rt") as fh:
        for start, end, cc in csv.reader(fh):
            if cc in ranges and ":" not in start:
                a, b = int(ipaddress.ip_address(start)), int(ipaddress.ip_address(end))
                if b - a > 256:
                    ranges[cc].append((a, b))
    pool = []
    for cc in COUNTRIES:
        for a, b in rng.sample(ranges[cc], 4):
            ip = ipaddress.ip_address(rng.randint(a + 1, b - 1))
            if ip.is_global:
                pool.append(str(ip))
    return pool


PEERS = _peer_pool()
PEER_ASN = {ip: 64512 + i * 7 for i, ip in enumerate(PEERS)}
FAST_RELAY = PEERS[3]  # the peer that first announces every laundering hop
rows: list[dict] = []
clock = [START]


def emit(label: str, inputs: list[tuple[str, float]], outputs: list[tuple[str, float]], *,
         fee: float, suspicious: bool = False, fee_text: str | None = None) -> str:
    clock[0] += rng.randint(40, 420)
    tid = txid(label)
    base_ms = clock[0] * 1000 + rng.randint(0, 999)
    peers = rng.sample(PEERS, rng.randint(2, 4))
    if suspicious:
        peers = [FAST_RELAY] + [p for p in peers if p != FAST_RELAY][:2]
    observers = list(OBSERVERS)
    for i, peer in enumerate(peers):
        delay = 0 if i == 0 else (rng.randint(40, 400) if suspicious else rng.randint(300, 4200))
        observer = observers[i % len(observers)]
        ms = base_ms + delay
        rows.append({
            "timestamp": f"{ms / 1000:.3f}", "src_ip": peer, "dst_ip": OBSERVERS[observer],
            "src_port": 8333, "dst_port": 8333, "txid": tid,
            "input_addresses": ";".join(a for a, _ in inputs),
            "output_addresses": ";".join(a for a, _ in outputs),
            "input_amounts": ";".join(f"{v:.8f}" for _, v in inputs),
            "output_amounts": ";".join(f"{v:.8f}" for _, v in outputs),
            "fee": fee_text if fee_text is not None else f"{fee:.8f}",
            "script_type": "p2pkh", "geo_country": "", "asn": PEER_ASN[peer], "observer_id": observer,
        })
    return tid


def main() -> None:
    wallets = [address(f"wallet-{i}") for i in range(48)]
    merchants = [address(f"merchant-{i}") for i in range(40)]
    balance = {w: round(rng.uniform(0.4, 6.0), 8) for w in wallets}

    exchange = address("exchange-hot-wallet")
    spent: set[str] = set()

    def ordinary(n: int) -> None:
        """Everyday activity: a wallet pays once (payment + change) and its
        change is left unspent, or an exchange batches withdrawals."""
        for _ in range(n):
            if rng.random() < 0.25:
                outs = [(rng.choice(wallets + merchants), round(rng.uniform(0.05, 1.5), 8)) for _ in range(rng.randint(5, 9))]
                outs = list(dict(outs).items())
                total = round(sum(v for _, v in outs) + rng.uniform(5, 20), 8)
                fee = round(rng.uniform(0.0001, 0.0004), 8)
                change = round(total - sum(v for _, v in outs) - fee, 8)
                emit(f"payout-{len(rows)}", [(exchange, total)], outs + [(exchange, change)], fee=fee)
                continue
            fresh = [w for w in wallets if w not in spent]
            if not fresh:
                return
            payer = rng.choice(fresh)
            spent.add(payer)
            amount = round(balance[payer] * rng.uniform(0.1, 0.6), 8)
            fee = round(rng.uniform(0.00002, 0.0002), 8)
            change = round(balance[payer] - amount - fee, 8)
            payee = rng.choice(merchants)
            emit(f"ordinary-{len(rows)}", [(payer, balance[payer])],
                 [(payee, amount), (address(f"change-{payer}"), change)], fee=fee)

    ordinary(24)

    # 1. peel chain: 42 BTC peeled over six hops into deposit addresses
    source = address("source-wallet")
    deposits = [address(f"deposit-{i}") for i in range(12)]
    held, holder = 42.0, source
    for hop in range(6):
        peel = round(rng.uniform(2.6, 3.4), 8)
        fee = 0.00012
        change_addr = address(f"peel-change-{hop}")
        change = round(held - peel - fee, 8)
        emit(f"peel-{hop}", [(holder, held)], [(deposits[hop], peel), (change_addr, change)],
             fee=fee, suspicious=True)
        held, holder = change, change_addr
        ordinary(2)

    # the other deposits are funded by unrelated-looking wallets
    for i in range(6, 12):
        funder = address(f"feeder-{i}")
        amount = round(rng.uniform(1.1, 2.4), 8)
        emit(f"feed-{i}", [(funder, amount + 0.3)], [(deposits[i], amount), (address(f"feeder-change-{i}"), 0.29988)],
             fee=0.00012, suspicious=True)
    ordinary(4)

    # 2. consolidation: every deposit spent together into one collector
    collector = address("collector")
    dep_amounts = []
    for r in rows:
        outs, vals = r["output_addresses"].split(";"), r["output_amounts"].split(";")
        for a, v in zip(outs, vals):
            if a in deposits and (a, float(v)) not in dep_amounts:
                dep_amounts.append((a, float(v)))
    total = round(sum(v for _, v in dep_amounts), 8)
    emit("consolidate", dep_amounts, [(collector, round(total - 0.0009, 8))], fee=0.0009, suspicious=True)
    ordinary(3)

    # 3. fan-out from the collector
    held = round(total - 0.0009, 8)
    fan = [address(f"fanout-{i}") for i in range(8)]
    each = round((held - 0.0006) / 8, 8)
    emit("fanout", [(collector, held)], [(a, each) for a in fan], fee=0.0006, suspicious=True)
    ordinary(3)

    # 4. CoinJoin-style mix: five fan-out outputs with two ordinary wallets, equal outputs
    mix_in = [(a, each) for a in fan[:5]] + [(wallets[0], balance[wallets[0]]), (wallets[1], balance[wallets[1]])]
    denom = 1.0
    mix_out = [(address(f"mix-out-{i}"), denom) for i in range(7)]
    change_total = round(sum(v for _, v in mix_in) - denom * 7 - 0.0014, 8)
    if change_total > 0:
        mix_out.append((address("mix-change"), change_total))
    emit("coinjoin", mix_in, mix_out, fee=0.0014, suspicious=True)
    ordinary(6)

    # 5. two deliberate validation findings
    emit("bad-fee", [(merchants[0], 0.5)], [(merchants[1], 0.4999)], fee=0.0001, fee_text="n/a")
    rows.append(dict(rows[5]))  # an exact duplicate record

    rows.sort(key=lambda r: float(r["timestamp"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    txs = len({r["txid"] for r in rows})
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows, {txs} transactions, {len(PEERS)} peers")


if __name__ == "__main__":
    main()
