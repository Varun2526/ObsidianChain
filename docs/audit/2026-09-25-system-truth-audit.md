# System truth audit, 2026-09-25

Scope: what the repository actually implements for network analysis, the
model, the pipeline, scale and false positives, and which presentation
claims that evidence supports. Read-only audit at commit `ccd5c67` plus the
running deployment. Every statement cites code (`file:line`, paths relative
to `src/obsidianchain/` unless shown) or an artifact field. Where something
was computed during the audit it says so.

---

## 1. Network layer: where it is implemented

| Area | Where | What it does |
|---|---|---|
| Capture schema | `io/ingest.py:44-52`, `ingest():234-307` | 14 canonical columns; only `txid` required; network fields `timestamp, src_ip, dst_ip, src_port, dst_port, geo_country, asn` optional (warning at `:266-270` when absent) |
| Capture contract | `contracts/capture.py:147-155` | IP validity, ASN range |
| Stage 3 GeoIP/ASN | `pipeline/orchestrator.py:164-173`, `geoip.py:125-146,247` | `OfflineCSVProvider()` built with no data root: always `provider_version: "uninstalled"`; `data/reference/` does not exist |
| Stages 5-6 network analysis and correlation | `orchestrator.py:185-198` -> `correlation/engine.py:99-174` | exact-txid join; per tx status CORRELATED / NO_NETWORK_OBSERVATION / UNMATCHED |
| Stage 9 network features (group E) | `pipeline/features_ps.py:117-128, 261-294` | per-tx observation count, observer diversity, peer count, ASN count, dominant-peer share, arrival spread, **computed only if `include_network=True` (`:413, :613`), default False (`:375, :672`)** |
| Stage 13 network evidence | `pipeline/alerts.py:359-400` | `p2p_network_telemetry` line: distinct IPs/ASNs, fixed score 0.30 (>1 IP) or 0.10 |
| Stage 16 graph | `alerts/graph.py:204-233, 307-343` | `ANNOUNCED_BY` tx->IP edges |
| Synthetic overlay | `network/synthetic.py:227-427`, CLI `network-generate` `cli.py:425-497` | generates observers, peers, announcements for Elliptic++ |
| Arrivals / audit / separation | `network/arrivals.py:149,241,306`, `network/audit.py`, `network/separation.py:281,418` | arrival vectors, identifiability, cannot-link evidence |
| Constrained clustering (network veto) | `cluster/constrained.py`, `cluster/pipeline.py:110-155` (`run_fused`) | used by CLI `run` (`cli.py:521`), demo, eval. **Not** used by the 17-stage pipeline, which uses plain union-find (`pipeline/blockchain.py:171-193`) |
| Phase 6/7 network features M3 | `features/netfeat.py:26-93`, `features/dataset.py:57-62` | 7 `net_*` features from the synthetic overlay |
| Alert network table | `alerts/correlate.py:51-131` | builds `alert_network` |

Artifacts: `data/processed/network/observations.parquet` (1,589,863 rows,
202,804 txs, 8 observers, 56 peer IPs, 56 ASNs), `network/arrival_vectors.parquet`,
`network/manifest.json` (generator 2.0.0), quarantined `network_truth/`,
`alert_network.parquet` (110,636 rows; `synthetic_network=True`),
per-run `alerts.json` / `investigation_graph.json`.

API: `/api/alerts/{id}` (network_context, correlation), `/api/transactions/{txid}`,
`/api/alerts/{id}/graph`, `/api/alerts/{id}/separation-evidence`, run
results and run graph. UI: `NetworkContextPanel`, `CorrelationPanel`,
`SeparationEvidencePanel`, `NetworkSubPage`, graph `ANNOUNCED_BY`,
`RunResultsPanel` NETWORK evidence.

**Real uploaded network fields**: exact-txid correlation status, group-E
counts (not used by the model), the `p2p_network_telemetry` evidence line,
graph IP nodes. **Broken on real data**: `observer_id` is not a canonical
column and is dropped at `ingest.py:307`, so `distinct_observers_count` is
always 0; GeoIP is always uninstalled, so evidence `countries: []`, while the
graph shows the capture's own unverified `geo_country` (`graph.py:216`).

**Synthetic overlay only**: arrival vectors, separation evidence, the
network veto, M3 features, `alert_network`, and everything in the case
Network tab and correlation panels for reference alerts.

## 2. Network -> ML (critical)

**Network metadata does not affect the production risk score.**

- Champion `ps_native_v5` features = `data/models/ps_native/v5/manifest.json`
  `features` (31) = `CORE_PS_FEATURE_COLUMNS` (`features_ps.py:151-154`),
  which excludes `GROUP_E_NETWORK`. Overlap: empty. Fallback (24 features):
  no network features.
- Stage 9 calls `extract_ps_features(frame)` without `include_network`
  (`orchestrator.py:234`). No caller in `src/` sets it.
- Fusion weights (`pipeline/alerts.py:146-151`) cover MODEL, PROPAGATION,
  PATTERN, ANOMALY only. The network line is built but never appended to
  the fused scores (`:359-400`; appends at `:290, :357, :443, :454`), is not
  counted as corroboration (`:456-459`), and ranking and severity cannot see
  it (`:177-187, :487-488`).
- Counterfactual run during the audit (sample capture with and without
  network fields): model raw scores identical for all 31 addresses (max
  difference 0.0).
- The only model that ever consumed network features is the Phase 6
  LightGBM behind the Phase 7 reference alerts, through M3, built from the
  synthetic overlay (`alerts.parquet.meta.json` `synthetic_network: true`).
- Side effect (bug): repeated network observations of one txid keep extra
  rows (`ingest.py:427-429`) and `derive_canonical_address_features`
  (`pipeline/features.py:85-128`) counts amounts per row, inflating
  `btc_sent`, an anomaly input. Fused scores moved for 10 clusters in the
  counterfactual (e.g. `1MultiIn` 0.682 CRITICAL -> 0.646 HIGH).
- Arrival `timestamp` does feed the temporal features (each txid replayed at
  first-seen time, `features_ps.py:383, 414`); IP, port, ASN and geo do not.

## 3. P2P propagation

| Capability | Status |
|---|---|
| Propagation timing (first/last seen, spread) | Implemented on synthetic data (`correlate.py:99-100`, `arrivals.py` `spread_ms`); on real data only as an unused per-tx feature (`features_ps.py:290`) |
| Announcing-peer counts per tx | Implemented (`correlate.py:103-110`, `pipeline/alerts.py:384`) |
| First-seen (which observer first) | Partial: `arrival_vectors.parquet` `first_observer`, synthetic only, not exposed by any API; no "first peer" metric |
| Peer concentration | Per-tx `dominant_peer_share` only (unused feature); no dataset or alert-level concentration |
| ASN concentration | Not implemented (counts only) |
| Geographic propagation or concentration | Not implemented (GeoIP uninstalled; synthetic IPs are RFC 5737) |
| Observer diversity | Synthetic yes; real captures broken (`observer_id` dropped) |

## 4. Elliptic++ has no network observations

`data/raw/` headers (edge lists, `txs_features` 184 columns, `wallets_features`
57 columns, class files) contain no IP, peer, ASN, geo or observation field;
the only time field is `Time step`.

The overlay (`network/synthetic.py`, frozen config `:451-462`): 8 observers, 64
origins, 15% broadcasters, 12 peers per observer, lognormal delay with sigma
1.1506 (Decker & Wattenhofer), a median of 1,200 ms that the code marks as an
assumption, synthetic RFC 5737 IPs and private ASNs, seed 0. The delay
shrinks when observer and origin share an ASN or region, "the entire
synthetic signal" (`:315`). **The origin of each transaction is drawn
uniformly at random, independent of address, entity or label** (`:343-344`),
so the frozen overlay carries no illicit signal. (World v2 `signal` is
deliberately label-conditioned, `world/noisy.py:97,177-178`; its null world is
the control.)

Useful for: exercising the mechanism end to end, the truth boundary and
leakage guards, abstention, robustness to clock bias and missing observers,
UI wording, and whether an injected signal is recovered.

Must not claim: any network accuracy or precision on Bitcoin; any
relationship between an IP, ASN or country and illicit activity; that M3
feature contributions show real network value; that world v2 shows network
features detect laundering (the signal was injected through the label); any
geographic or ASN finding; that an announcing peer is a sender; that the
delay parameters are measured.

## 5. One uploaded transaction through the 17 stages

`tx_benign_02` (src_ip 8.8.8.8, port 8333, ASN 15169, geo "US") from
`data/samples/canonical_acceptance_capture.json`; its input address forms
alert `alert_cluster_1BenignChange1`.

| # | Stage | Network metadata |
|---|---|---|
| 1 | Ingest | carried |
| 2 | Validate/dedupe | used for row retention only |
| 3 | GeoIP/ASN | **lost** (provider uninstalled) |
| 4 | Blockchain analysis | not used |
| 5 | Network analysis | carried (12 observations) |
| 6 | Correlation | carried as a status (10/11 correlated) |
| 7-8 | Graph, clustering | not used (no network veto here) |
| 9 | Features | **lost** (`include_network` False); timestamp used; observation rows inflate `btc_sent` |
| 10 | ML | not used (score unchanged by the counterfactual) |
| 11 | Anomaly | indirect, through the duplication bug |
| 12 | Patterns | not used |
| 13 | Fusion | **displayed only** (score 0.3, not fused) |
| 14 | Ranking | not used |
| 15 | Explanation | displayed ("2 announcements across 2 peer IPs") |
| 16 | Investigation graph | displayed (`ip:8.8.8.8`, ASN, capture-supplied "US") |
| 17 | Report/manifest | provenance only |
| UI | Run results | evidence text only; IP/ASN not rendered in the panel; run graph shows IP nodes |
| Case report | — | **uploaded-run alerts cannot be referenced into a case** (`alerts/contract.py:48` id format), so they never reach reports |

Also found: the model explanation text in uploaded-run alerts drops feature
names ("Key contributing features:  (increases risk), ..."), because
`pipeline/alerts.py:417` reads `e.get('feature')` while
`pipeline/features.py:273` stores `feature_name`. Visible on the deployed
instance.

## 6. Scalability (measured on this Mac)

| Component | Measured now | Breaks at | Category |
|---|---|---|---|
| 17-stage run inside the HTTP request, blocking the event loop (`console/routes_investigations.py:347` -> `console/runs.py:224`) | 40k rows: 23.3 s, 1.3 GB; freezes every endpoint | **already today** with a medium upload; 64 MB ~3+ min (extrapolated) exceeds proxy timeouts | architectural: needs a job queue and DB-backed progress |
| In-memory chain index per process (`api/investigation.py:78-91`) | +535 MB, 0.49 s load; edges stored twice | 10M addresses ~6.5 GB/process, 100M ~65 GB (linear extrapolation, lower bound) | architectural: external index (DuckDB/Parquet, RocksDB, Postgres) |
| `async def` alert endpoints with blocking pandas, no caching (`api/app.py:411,600`; `api/artifacts.py:208,232`) | get_alert 80-150 ms warm, alert graph 138-606 ms | concurrent users queue | prototype: plain `def` + cached reads |
| Trace BFS via pandas rows (`investigation.py:409-450`) | 5-306 ms up to 1,500 nodes | cost tracks nodes visited, not data size | prototype |
| `RUN_PROGRESS` in-process dict (`console/runs.py:172`) | — | wrong with >1 worker, lost on restart | architectural |
| Chain-index build (`chain_index.py:103-105`) | 2.5 s, +474 MB | 10M ~6 GB; 100M does not fit; no incremental build | prototype (batch) / architectural (streaming) |
| Feature engine (`features_ps.py:425`) | 8,400 tx/s | throughput fine; no checkpointed state for streaming | architectural for streaming |
| Propagation path explanations (`ml/propagation.py:174-175`) | 10.5 s, +513 MB with 1,000 paths | path BFS in Python dicts | prototype |
| SQLite casework | 0.34 ms per connection | multi-host | architectural only for horizontal scale |
| Model scoring | 0.55 ms p50, 463k rows/s | not a bottleneck | — |
| Cytoscape | ≤ 410 ms layout, 1,500-node cap | bounded by design | — |

## 7. Model truth

- Registry roles (`data/models/ps_native/registry.json`): champion
  `ps_native_v5`, candidate `ps_native_v5` (same model), fallback
  `ps_native_v5_fallback_no_g`.
- Pipeline: `orchestrator.py:255-260` -> `_load_serving_model` (`:569-581`)
  tries champion then fallback via `PsNativeRiskModel.from_registry`
  (`ml/ps_model.py:170-188`), `registry.resolve` (schema check + artifact
  SHA-256 verify, `ml/registry.py:216-231`), then feature-order and contract
  checks (`orchestrator.py:271-282`). A real run's stage 10: `SCORED`,
  `model_version ps_native_v5`, `model_sha256 974d37f2...`.
- Tampering test (audit, scratch copy): corrupting `v5/model.joblib` made the
  run score with the fallback and record the verification failure.
- Both are LightGBM (`LGBMClassifier`). **Random Forest is not in the
  production path**: in `src/` only a comment (`ml/protocol.py:20`); in
  research scripts; the registered legacy `ps_native_v1` (RandomForest)
  holds no role. `docs/ML_PIPELINE.md:40` and `docs/PRODUCTION_FREEZE.md:48-51`
  are stale and still say Random Forest.
- The Phase 7 reference alerts come from a different model: Phase 6
  LightGBM, 43 features including synthetic M3, split train t1-34 /
  validation t35-41 / test t42-49 (`alerts.parquet.meta.json` `artifact.model`).

## 8. t43 / t45 / t47

Evidence already in the repository, plus label counts computed during the
audit from `data/raw` and from the cycle-2 event cache (built by the
production feature engine). No model was scored for this section.

**Proven**

1. **The collapse is real and not a ceiling artifact.** nAP is chance
   corrected (`ml/protocol.py:298-324`), so t45's 0.017 is essentially random
   ranking. P@100 at t45 is 0.02 against a ceiling of 0.27 (27 positives);
   at t43 0.18 against 0.97.
   (`data/models/ps_native/holdout/ps_native_v5.json` `by_first_seen_step`.)

2. **Label distribution change.** Prevalence at t43 1.6%, t45 0.30%, t47
   2.9%. Every development fold had prevalence of at least 4.8%
   (`v5/evaluation.json` folds). Across the eight holdout steps, prevalence
   and nAP rank-correlate (Spearman 0.90, p = 0.002); across development
   folds they do not significantly (0.45, p = 0.14).

3. **The number of independent illicit events collapses.** Illicit-labelled
   transactions per timestep (`txs_classes.csv` x `txs_features.csv`):
   t42 239, t43 24, t44 24, t45 5, t46 2, t47 22, t48 36, t49 56. Among
   holdout positives, the distinct illicit transactions they appear in at
   their first step: t43 24, t45 5, t46 2, t47 22. The development median
   was 95 per step (minimum 23). Wallet positives are clustered inside few
   transactions: at t46, 98.6% of the 505 positives come from one
   transaction. **Per-step holdout numbers therefore rest on very few
   independent events**: t46's 0.98 is effectively one success, and t45's
   0.017 is five transactions.

4. **The positives that fail look different from the training positives.**
   Descriptive statistics of positives at their snapshot event:

   | | dev t17-40 | t44 (ok) | t46 (ok) | t43 | t45 | t47 |
   |---|---|---|---|---|---|---|
   | median fan-in of the tx | 103 | 148 | 497 | 4 | 3 | 8 |
   | upstream-funded share | 0.91 | 0.85 | 0.99 | 0.65 | 0.56 | 0.78 |
   | no prior transaction | 0.23 | 0.30 | 0.77 | 0.52 | 0.59 | 0.72 |

   Development positives are mostly inputs of large consolidation
   transactions; the steps where the model works share that shape; the
   collapse steps are small transactions with less upstream funding.
   t42 (fan-in 1, nAP 0.70) is an exception, so fan-in is not the whole
   story.

5. **Input drift monitoring does not detect it.** `exp23_relative_drift.json`
   rates every holdout step, including t43/t45/t47, WITHIN_BASELINE.

6. **Labels are internally consistent.** Every positive wallet touches an
   illicit-labelled transaction (85-100% at its first step), so this is not
   a label-join error.

**Conclusion.** The evidence supports a label-distribution and typology
change (fewer, smaller, differently shaped illicit events than anything in
development) that the model does not generalise to, combined with extreme
sparsity that makes single steps high-variance. It is not a feature
pipeline failure (the same production engine built every period) and not
a leakage artifact.

**Not proven.** Why illicit activity changed at t43 (the literature on the
Elliptic data links it to a dark-market closure, but nothing in this
repository establishes that); whether the model scored the collapse-step
positives low as a group (that needs a diagnostic scoring pass, which ADR
0004 permits but which has not been run); whether any feature change would
fix it.

## 9. False positives

| Source | Mechanism | Where | Measured? |
|---|---|---|---|
| Alert rank budget (uploaded runs) | top 2/8/20% become CRITICAL/HIGH/MEDIUM; only absolute gate is fused ≥ 0.20; **a clean dataset still produces CRITICAL alerts** | `pipeline/alerts.py:159-187` | no |
| Anomaly (MAD z-scores vs the run's own median, cluster max) | anomaly alone passes the 0.20 floor at z ≥ ~1.02; on a synthetic unlabelled population 65.9% of addresses passed (audit scratch check) | `ml/anomaly.py:116-240`, `alerts.py:289` | no |
| Peeling rule | depth ≥ 2 with 2-3 outputs; no amount test, no suppressor (Phase 7 uses depth 5) | `pipeline/patterns.py:97-166` | no |
| Mixing rule | ≥2 equal outputs; no suppressor for exchange batches or payroll | `pipeline/patterns.py:200-230` | no |
| Noisy-OR fusion | assumes independent lines; hand-set weights; correlated structural lines compound | `alerts.py:146-174` | no |
| Common-input clustering | no CoinJoin exclusion, no network veto on this path; merges spread risk | `pipeline/blockchain.py:171-193` | no |
| Cluster model line = member max | one address makes the cluster high | `alerts.py:406-418` | no |
| Phase 7 severity = worst member | one CRITICAL member makes any cluster CRITICAL | `alerts/build.py:147-191` | no |
| Watchlist propagation | seed in cluster gives 1.0; guilt by association | `ml/propagation.py`, `alerts.py:447-453` | partly (exp14 used Elliptic labels, not OFAC) |
| Network evidence | not fused; cannot raise a score | `alerts.py:359-400` | n/a |

**What is measured**: address-level ranking on Elliptic++ labels only, with
unknown (class 3) addresses **excluded, never counted as negative**
(`research/reproduction/evaluate_holdout.py:57`, exp22 script `:72`,
`features/dataset.py:48,157-163`). v5 holdout: FP@100 = 0, FP@500 = 25,
FP@1000 = 307; development folds: FP@100 mean 2.9 (max 24). Phase 7 band
precision (0.90/0.75/0.50) is measured on the validation split the
calibrator was fitted on; exp09 showed such static bands lose 0.31-0.32
precision out of time.

**What is not measured**: any false-positive rate of the uploaded-run fused
alert path (no research script touches `fused_risk_score`, anomaly or
pattern detectors); precision of rank-budget severity; cluster-level FP;
FP among unknown addresses; any negative-control dataset showing zero
CRITICAL alerts.

## 10. Presentation claims (`docs/design/ObsidianChain_SIH2026_Idea.pptx`)

### A. Demonstrable today

| Claim | Evidence |
|---|---|
| Runs offline on Linux, no GPU, no cloud API | `tests/test_offline.py`, `tests/test_frontend_offline.py`; image built with `--network none` from vendored wheels (`Makefile` build) |
| Ingest CSV/JSON/XML with schema validation and SHA-256 per file | `io/ingest.py`, `contracts/capture.py`; dataset `sha256` stored per upload |
| LightGBM, 31 as-of-time features, 300 trees | `v5/manifest.json` `model_type`, `features`, `hyperparameters.n_estimators` |
| Per-row TreeSHAP explanations | `v5/manifest.json` `explanation_method`; `ml/ps_model.py:325` |
| About 0.55 ms per address (p50; p95 0.59) | `v5/evaluation.json` `performance.single_row_latency_ms` |
| 7 leakage defects found, each guarded by a test | `research/autoresearch_2026_09_23/19_leakage_audit.md` L1-L7 with named tests |
| Holdout t42-49 sealed in code, opened once under a decision record | `ml/protocol.py` `HoldoutSealError`; ADR 0003; `holdout/ps_native_v5.lock` |
| Earlier explanation method right on direction only ~34% of the time, replaced | `10_explanation_audit.md` (66.1% mislabelled) |
| ROC-AUC 0.946 on the holdout | `holdout/ps_native_v5.json` (show with nAP 0.548 and the per-step collapse) |
| 203,769 transactions, 822,942 addresses | `txs_features.csv` rows; chain index |
| LightGBM chosen by a 12-fold Holm-corrected comparison; RF and HistGB tied | `exp01_model_family.json` `family_comparison` (feature schema v2) |
| Registry attestation: hash-checked model, automatic fallback | `ml/registry.py:119-123, 216-231`; fallback demonstrated by tampering in the audit |
| Case lifecycle, RBAC, append-only audit, reviewer sign-off, export with integrity check | clean end-to-end run on the deployed instance (API and UI), `console/*`, `/integrity/verify` |
| Graph tracing, address and transaction intelligence | `api/investigation.py`, `/api/graph/trace`, Graph explorer |
| Transactions and network observations joined in one graph for uploaded captures | `correlation/engine.py`, `alerts/graph.py:204-233` (joined and displayed, not scored) |

### B. Partial, or demonstrated only on synthetic data (qualify on the slide)

| Claim as written | What is true |
|---|---|
| "Network origin can veto a merge; abstains when evidence is thin" | Implemented in `cluster/constrained.py` / `run_fused` and shown on synthetic worlds; **not used by the 17-stage pipeline** (plain union-find) |
| "Network-layer context" as a differentiator | Displayed evidence only; not in the model, the fused score or ranking; reference-data network is synthetic; GeoIP uninstalled; `observer_id` dropped |
| Reference alerts' risk scores | Scored by the Phase 6 model, which **includes synthetic network features (M3)**; say so wherever those scores are shown |
| "TXID / WTXID matching" | TXID only; WTXID not found anywhere |
| "Byte-identical runs" | Deterministic model reproduction (gate criterion 9, max difference 0) and read-only layers leave artifacts byte-identical; not a test that every pipeline run is byte-identical |
| "Signed disposition" | Named, timestamped, audited decisions; reports signed off by a reviewer; no cryptographic signature |
| "Precision collapses after a known market closure" | Collapse and label-distribution change proven (section 8); the market-closure cause is external literature, not shown here |
| "Every number carries VALID / WITHDRAWN status" | Registry marks withdrawn models; not every number in the product |
| "One workflow: ingestion to report" | Uploaded-run alerts cannot be referenced into a case, so they do not reach reports; the reference alerts do |
| Faster triage, less tool switching | Plausible, not measured against a baseline |
| Rank-budget severity | Implemented; its false-positive behaviour is unmeasured and produces CRITICAL alerts even on clean data |
| "React 19" | React 18.3.1 |
| "69 test files" | 70 Python test files and 6 frontend test files |

### C. Future scope: must not be presented as implemented

- Real P2P propagation analysis: first-seen peer, propagation timing, peer, ASN or geographic concentration on real traffic.
- Network features in the production risk model, or any network-informed score.
- GeoIP / country resolution (no database installed).
- Measured false-positive rate or precision of the alert pipeline; performance on live traffic.
- Multi-chain support (the "only the parser changes" claim is untested).
- Standing collection of propagation data; attribution plug-ins (KYC, seized devices); joint entity and origin inference.
- Streaming or 10M+ address scale (section 6: memory and blocking runs break first).
- WTXID matching; cryptographically signed dispositions.

## Defects found by this audit

1. Uploaded-run model explanations drop feature names (`pipeline/alerts.py:417` reads `feature`; `pipeline/features.py:273` stores `feature_name`). Visible in the demo.
2. Repeated network observations inflate `btc_sent` and change anomaly and fused scores (`ingest.py:427-429`, `pipeline/features.py:85-128`).
3. `observer_id` dropped at ingest (`io/ingest.py:307`), so observer diversity on real captures is always 0.
4. GeoIP provider is never configured in production (`orchestrator.py:166-167`); the graph shows unverified capture-supplied country.
5. Stale documentation still names Random Forest as production (`docs/ML_PIPELINE.md:40`, `docs/PRODUCTION_FREEZE.md:48-51`).
6. Analysis runs execute inside the HTTP request and block the server (`console/routes_investigations.py:347`).
