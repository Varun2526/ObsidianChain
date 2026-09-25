# Plan: bring the network layer up to the blockchain layer (2026-09-25)

Source: `docs/audit/2026-09-25-system-truth-audit.md`. Scope is the
problem statement's requirement: consume network metadata offline and
correlate it with blockchain data. Out of scope for the prototype: live
Bitcoin nodes, P2P sensors, streaming, multi-chain, graph databases, 10M+
scale.

## The constraint that shapes the ML step

No labelled dataset in this repository carries real network fields:
Elliptic++ has none; its overlay draws each transaction's origin uniformly
at random (`network/synthetic.py:343-344`), so it has no signal; the v2
`signal` world plants network behaviour through the label
(`world/noisy.py:97,177-178`), so any gain there is circular by construction.

Therefore: network features can be **computed, shown and tested**, but a
comparison of blockchain-only against blockchain+network can only be a
synthetic-control experiment. It demonstrates the mechanism; it cannot
justify changing production scoring. **The production model stays
blockchain-only** until labelled captures with real network fields exist,
and the documents say so.

## Phases

| # | Phase | Deliverable | Done when |
|---|---|---|---|
| 0 | Defects that corrupt existing outputs | (a) model explanation feature names; (b) repeated observations no longer inflate address amounts | tests fail before, pass after |
| 1 | Network fields survive ingestion | `observer_id` (alias `observer`) preserved as a documented extension column; network normalisation: IP validity and class (public / private / reserved), timestamps to epoch ms, integer ports and ASN | ingest test with two observers of one txid keeps both; normalisation unit tests |
| 2 | Correlation and propagation from uploaded observations | per transaction: observations, observers, peers, first seen, last seen, spread, first-seen peer and observer, per-peer first arrival | run writes `network_propagation.json`; values checked against a hand-computed capture |
| 3 | Concentration and context | per transaction and per alert cluster: peer diversity, ASN diversity, dominant-peer ratio, country diversity, share of private/reserved IPs | numbers match hand computation; shown as NETWORK evidence, **not fused** |
| 4 | GeoIP / ASN | provider wired to a configurable offline database; until one is supplied, capture-supplied `geo_country`/`asn` are shown as "capture-supplied, unverified" | a supplied CSV resolves; absence is reported, never guessed |
| 5 | Network graph and UI | tx -> peer IP -> ASN nodes with first-seen timestamps; propagation table in run results | visible in a clean end-to-end run |
| 6 | Runs off the request thread | analysis executes in a background worker; progress persisted | a 40k-row run no longer blocks other endpoints |
| 7 | Network-feature experiment (pre-registered) | blockchain-only vs blockchain+network on the synthetic null and signal worlds, protocol-B style folds | result recorded whichever way it goes; production scoring unchanged |
| 8 | Documentation and claims | audit claim lists updated; PPT wording from the audit's list B | every network claim cites an implementation |

**Status (2026-09-25):** phases 0-7 done. Phase 4 uses DB-IP "IP to
Country Lite" 2026-09 (CC BY 4.0) from `data/reference/`. Phase 8: the
audit's "Resolved since this audit" section records the changes and the
moved claims; the PPT was left unedited on instruction.

Order follows dependency: 0 and 1 first (everything downstream reads the
ingested frame), then 2-3 (computation), 4-5 (presentation), 6 before any
heavier processing, 7 last.

## What will be claimed afterwards

"ObsidianChain correlates uploaded Bitcoin transaction metadata with network
observations, computes propagation timing, peer and ASN context per
transaction and per alert, and shows it as investigation evidence. Whether
network-derived features improve detection is evaluated experimentally; on
the data available they are not used in the risk score."
