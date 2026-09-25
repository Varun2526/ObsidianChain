/**
 * The investigation read path: address profiles, money-flow traces,
 * transactions, search and model intelligence.
 *
 * Every shape here mirrors a backend module (api/investigation.py,
 * api/models.py, api/alerts.py get_alert_graph). Three layers are kept
 * apart on purpose, and the UI renders them apart:
 *
 *   observed   - chain structure from the Elliptic++ chain index
 *   attributed - an external watchlist (OFAC SDN, analyst CSVs)
 *   model      - a learned association from the Phase 7 reference run
 */

import { apiGet } from "./console";
import type { Severity } from "./types";

// ---- annotations ----------------------------------------------------------

export interface ModelAnnotation {
  risk_score: number | null;
  severity: Severity;
  alert_id: string;
  alerts: number;
  scope: string;
}

export interface ClusterAnnotation {
  cluster_id: number;
  size: number | null;
}

export interface WatchlistHit {
  source: string;
  label: string;
}

export interface ReadProvenance {
  chain_index_fingerprint: string;
  chain_scope: string;
  model_scope: string;
  watchlist_scope: string;
  labels_served: false;
  alert_run_fingerprint?: string;
}

export interface TxRow {
  txid: number;
  timestep: number | null;
  fee_btc: number | null;
  in_btc: number | null;
  out_btc: number | null;
  n_inputs: number;
  n_outputs: number;
}

// ---- address --------------------------------------------------------------

export interface Counterparty {
  address: string;
  shared_transactions: number;
  model: ModelAnnotation | null;
  cluster: ClusterAnnotation | null;
  watchlist: WatchlistHit[];
}

export interface ShapExplanation {
  alert_id: string;
  address: string;
  feature: string;
  contribution: number;
  feature_value: number | null;
  feature_group: string | null;
  signal_category: string | null;
}

export interface AddressProfile {
  address: string;
  observed: {
    transactions: number;
    as_input: number;
    as_output: number;
    first_timestep: number | null;
    last_timestep: number | null;
    active_timesteps: number;
    timeline: { timestep: number; as_input: number; as_output: number }[];
    transactions_list: (TxRow & { role: "input" | "output" | "both" })[];
    transactions_shown: number;
  };
  counterparties: {
    paid_to: Counterparty[];
    paid_to_total: number;
    paid_by: Counterparty[];
    paid_by_total: number;
    hub_transactions_skipped: number;
    hub_threshold: number;
  };
  cluster: ClusterAnnotation | null;
  watchlist: WatchlistHit[];
  model: {
    alerts: { alert_id: string; risk_score: number | null; severity: Severity; observed_at_timestep: number | null }[];
    explanations: ShapExplanation[];
    scope: string;
    meaning: string;
  };
  network: {
    observations: { txid: number; peers: number; asns: number; first_seen_ms: number | null }[];
    meaning: string;
  };
  provenance: ReadProvenance;
}

// ---- graph ----------------------------------------------------------------

export type GraphNodeKind = "address" | "transaction" | "cluster" | "ip" | "asn";
export type GraphEdgeKind = "SPENDS" | "PAYS" | "MEMBER_OF" | "ANNOUNCED_BY" | string;

export interface GraphNode {
  id: string;
  kind: GraphNodeKind;
  label: string;
  data: Record<string, unknown> & {
    address?: string;
    txid?: number;
    timestep?: number | null;
    hop?: number;
    seed?: boolean;
    alert_member?: boolean;
    model?: ModelAnnotation | null;
    cluster?: ClusterAnnotation | null;
    watchlist?: WatchlistHit[];
  };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: GraphEdgeKind;
  data: Record<string, unknown> & { timestep?: number | null };
}

export interface TraceResponse {
  seeds: { addresses: string[]; txids: number[] };
  direction: TraceDirection;
  hops: number;
  max_nodes: number;
  time_window?: { min_timestep: number | null; max_timestep: number | null };
  graph: { node_count: number; edge_count: number; nodes: GraphNode[]; edges: GraphEdge[] };
  truncated: boolean;
  hub_transactions_skipped: { txid: number; participants: number }[];
  hub_threshold: number;
  meaning: string;
  provenance: ReadProvenance;
}

export interface AlertGraphResponse extends Omit<TraceResponse, "seeds" | "time_window"> {
  alert_id: string;
  seeds: { used: number; members_total: number; omitted: number; rule: string };
  peers_omitted: boolean;
  layers: Record<string, string>;
}

export type TraceDirection = "upstream" | "downstream" | "both";

export interface TraceRequest {
  addresses?: string[];
  txids?: number[];
  direction?: TraceDirection;
  hops?: number;
  maxNodes?: number;
  minTimestep?: number | null;
  maxTimestep?: number | null;
}

export function traceQuery(req: TraceRequest): string {
  const p = new URLSearchParams();
  for (const a of req.addresses ?? []) p.append("address", a);
  for (const t of req.txids ?? []) p.append("txid", String(t));
  p.set("direction", req.direction ?? "both");
  p.set("hops", String(req.hops ?? 1));
  p.set("max_nodes", String(req.maxNodes ?? 300));
  if (req.minTimestep != null) p.set("min_timestep", String(req.minTimestep));
  if (req.maxTimestep != null) p.set("max_timestep", String(req.maxTimestep));
  return p.toString();
}

// ---- search ---------------------------------------------------------------

export interface SearchResult {
  kind: "alert" | "transaction" | "address";
  id: string;
  label: string;
  detail?: TxRow;
}

// ---- models ---------------------------------------------------------------

export type ResultType = "DEVELOPMENT" | "CONFIRMATION" | "HOLDOUT" | "PRODUCTION";

export interface Summary {
  n: number; mean: number; median: number; sd: number; min: number; max: number; ci95: [number, number];
}

export interface ModelListRow {
  version: string;
  role: string | null;
  feature_schema_version: string | null;
  registered_at: string | null;
  attested_source_commit: string | null;
  holdout: { nap?: number; "P@100"?: number; ece?: number; result?: string; sha256?: string } | null;
  notes: string | null;
  withdrawn: boolean;
}

export interface ModelList {
  roles: Record<string, string | null>;
  models: ModelListRow[];
  history: Record<string, unknown>[];
  meaning: string;
}

export interface FoldRow {
  fold: string;
  part: "tune" | "confirm";
  result_type: ResultType;
  n: number;
  positives: number;
  prevalence: number;
  nap: number | null;
  roc_auc: number | null;
  "P@100": number | null;
  "R@500": number | null;
  tx_weighted_nap: number | null;
  ece: number | null;
  brier: number | null;
  [k: string]: unknown;
}

export interface ModelDetail {
  version: string;
  roles: string[];
  feature_schema_version: string | null;
  attested_source_commit: string | null;
  registered_at: string | null;
  notes: string | null;
  manifest: Record<string, unknown>;
  result_types: Record<ResultType, string>;
  production: { available: false; meaning: string };
  evaluation?: {
    protocol: string | null;
    summary: Record<"tune" | "confirm" | "all", {
      result_type: string;
      address: Record<string, Summary>;
      calibration: Record<string, unknown> | null;
    }>;
    temporal_trend_nap: Record<string, unknown> | null;
    slices_nap: Record<string, unknown> | null;
    folds: FoldRow[];
    performance: Record<string, unknown> | null;
  };
  gate?: {
    decision: string;
    evaluated_at: string | null;
    spec: string | null;
    criteria: Record<string, { status: string; evidence: unknown }>;
  };
  holdout?: {
    result_type: "HOLDOUT";
    period: string | null;
    exception: string | null;
    n_addresses: number;
    positives: number;
    prevalence: number;
    address: Record<string, number>;
    calibration: Record<string, number>;
    reliability: { bin: number; n: number; mean_predicted: number; observed_rate: number }[] | null;
    slices_nap: Record<string, number | null>;
    by_first_seen_step: Record<string, { n: number; positives: number; nap: number | null; "P@100"?: number | null; r_precision?: number | null }> | null;
    sha256: string | null;
  };
  drift_baseline?: {
    calibrated_on: unknown;
    quantile: unknown;
    score_psi_p95: unknown;
    per_step: { step: number; score_psi: number; n_major: number }[];
    finding: string;
  };
  stacker?: Record<string, unknown>;
  calibration?: Record<string, unknown>;
}

// ---- run graph ------------------------------------------------------------

export interface RunGraphResponse {
  run_id: string;
  run_fingerprint: string | null;
  graph: { node_count: number; edge_count: number; nodes: GraphNode[]; edges: GraphEdge[] };
  truncated: boolean;
  nodes_total: number;
  result_type: string;
  meaning: string;
}

// ---- run network propagation (network/propagation.py) --------------------

export interface PeerArrival {
  peer_ip: string;
  first_seen_ms: number | null;
  observations: number;
  observers: string[];
  asns: number[];
  ip_class: string;
  country_iso?: string | null;
}

export interface TxPropagation {
  txid: string;
  observations: number;
  timed_observations: number;
  observer_source: string;
  observers: string[] | null;
  observer_count: number | null;
  first_seen_ms: number | null;
  last_seen_ms: number | null;
  spread_ms: number | null;
  first_seen_peers: string[];
  first_seen_observers: string[];
  peer_count: number;
  peers: PeerArrival[];
  asns: number[];
  asn_count: number;
  countries: string[];
  country_source: string | null;
  dominant_peer_ip: string | null;
  dominant_peer_share: number | null;
  non_routable_peer_share: number | null;
  resolved_countries?: string[];
  country_resolution?: string | null;
}

export interface RunNetworkResponse {
  run_id: string;
  schema: string;
  meaning: string;
  summary: {
    transactions_with_observations: number;
    observations: number;
    with_timing: number;
    with_spread: number;
    median_spread_ms: number | null;
    distinct_peers: number;
    distinct_asns: number;
    observer_source: Record<string, number>;
    distinct_resolved_countries?: number;
    country_resolution?: string | null;
  };
  transactions: TxPropagation[];
  transactions_total: number;
}

// ---- calls ----------------------------------------------------------------

const enc = encodeURIComponent;

export const getAddress = (address: string, signal?: AbortSignal) =>
  apiGet<AddressProfile>(`/addresses/${enc(address)}`, signal);

export const traceGraph = (req: TraceRequest, signal?: AbortSignal) =>
  apiGet<TraceResponse>(`/graph/trace?${traceQuery(req)}`, signal);

export const getAlertGraph = (alertId: string, opts: { hops?: number; direction?: TraceDirection; maxNodes?: number } = {},
  signal?: AbortSignal) => {
  const p = new URLSearchParams({ hops: String(opts.hops ?? 2), direction: opts.direction ?? "both",
    max_nodes: String(opts.maxNodes ?? 400) });
  return apiGet<AlertGraphResponse>(`/alerts/${enc(alertId)}/graph?${p}`, signal);
};

export const search = (q: string, signal?: AbortSignal) =>
  apiGet<{ query: string; results: SearchResult[] }>(`/search?q=${enc(q)}`, signal);

export const listModels = (signal?: AbortSignal) => apiGet<ModelList>("/models", signal);

export const getModel = (version: string, signal?: AbortSignal) =>
  apiGet<ModelDetail>(`/models/${enc(version)}`, signal);

export const getRunGraph = (investigationId: string, runId: string, signal?: AbortSignal) =>
  apiGet<RunGraphResponse>(`/investigations/${enc(investigationId)}/runs/${enc(runId)}/graph`, signal);

export const getRunNetwork = (investigationId: string, runId: string, signal?: AbortSignal) =>
  apiGet<RunNetworkResponse>(`/investigations/${enc(investigationId)}/runs/${enc(runId)}/network`, signal);
