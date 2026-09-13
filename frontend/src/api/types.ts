/**
 * Types mirroring the Phase 7 API exactly.
 *
 * Every field here was read off a real response (see `frontend/fixtures/`).
 * Nothing is invented: if the backend does not return a field, it does not
 * appear here, and the UI renders "not available" rather than filling a gap.
 *
 * `null` is meaningful throughout. The backend distinguishes "not computable
 * for this address" from "computed as zero", and collapsing the two in the
 * client would undo that at the last step.
 */

export type Severity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

/** The four categories the backend uses to say what KIND of claim a row is. */
export type Category =
  | "MODEL_SIGNAL"
  | "BLOCKCHAIN_CONTEXT"
  | "NETWORK_CONTEXT"
  | "INSUFFICIENT_EVIDENCE";

export type FeatureGroup = "M0" | "M1" | "M2" | "M3";

export type RelationshipKind = "CO_SPEND_COMPONENT" | "FUNDED_VIA_TRANSACTION";

export interface AlertSummary {
  alert_id: string;
  rank: number;
  cluster_id: number;
  risk_score: number | null;
  severity: Severity;
  ranking_aggregation: string;
  members_scored: number;
  members_total: number;
  first_timestep: number;
  last_timestep: number;
  top_signals: string[];
}

export interface ModelProvenance {
  family: string;
  calibration: string;
  n_features: number;
  explainability: string;
}

export interface SeverityBand {
  band: string;
  target: number;
  threshold: number | null;
  support: number;
  validation_precision: number | null;
  populated: boolean;
}

export interface Provenance {
  provenance_type: string | null;
  dataset_id: string | null;
  dataset_sha256: string | null;
  synthetic_network: boolean | null;
  run_fingerprint: string | null;
  public_run_fingerprint: string;
  phase6_dataset_fingerprint: string | null;
  artifact_schema: string | null;
  model: ModelProvenance | null;
  feature_semantics: string | null;
  split: Record<string, string> | null;
  scored_split: string | null;
  severity_bands: SeverityBand[] | null;
  ranking_aggregation: string | null;
  git_revision: string | null;
  notes: string[] | null;
}

export interface AlertListResponse {
  run_fingerprint: string;
  alert_count_total: number;
  alert_count_matched: number;
  offset: number;
  limit: number;
  alerts: AlertSummary[];
  meaning: string;
  score_scope: string;
  provenance: Provenance;
}

export interface Contribution {
  feature: string;
  feature_group: FeatureGroup;
  /** SHAP value in LOG-ODDS, not a probability delta. */
  contribution: number;
  feature_value: number | null;
  signal_category: Category;
  value_category: Category;
}

export interface MemberExplanation {
  address: string;
  base_value: number | null;
  contributions: Contribution[];
}

export interface WhyFlagged {
  units: string;
  meaning: string;
  insufficient_evidence_meaning: string;
  categories: Category[];
  per_member: MemberExplanation[];
}

export interface EvidenceMember {
  address: string;
  values: Record<string, number | boolean | null>;
  /** Features the backend could not compute for this address. */
  unavailable: string[];
}

export interface EvidenceGroup {
  category: Category;
  features: string[];
  members: EvidenceMember[];
}

export interface Evidence {
  aggregate: {
    category: Category;
    transactions: number | null;
    btc_sent_total: number | null;
    btc_received_total: number | null;
    unique_counterparties: number | null;
    peel_chain_members: number;
  };
  groups: Record<FeatureGroup, EvidenceGroup>;
  insufficient_evidence_meaning: string;
}

export interface RelationshipEdge {
  address_a: string;
  address_b: string;
  relationship: RelationshipKind;
  timestep: number;
  category: Category;
}

export interface Relationships {
  supported: Record<string, string>;
  count: number;
  edges: RelationshipEdge[];
  note: string;
}

export interface TimelinePoint {
  timestep: number;
  transactions: number;
  active_addresses: number;
  btc_sent: number | null;
  btc_received: number | null;
}

export interface Timeline {
  category: Category;
  unit: string;
  points: TimelinePoint[];
}

export interface NetworkContext {
  category: Category;
  members_with_observations: number;
  members_reaching_production_minimum: number;
  available: boolean;
  status: Category;
  meaning: string;
  synthetic_warning: string;
  insufficient_evidence_meaning: string | null;
}

export interface MemberRow {
  address: string;
  risk_score: number | null;
  severity: Severity;
  observed_at_timestep: number;
  first_timestep: number;
}

export interface Members {
  shown: number;
  total_scored: number;
  withheld: number;
  rows: MemberRow[];
}

export interface GeoFacts {
  unique_ips: number;
  unique_asns: number;
  countries: string[];
  country_resolution: string;
  country_note: string;
  reserved_ranges: string[];
  all_private_asns: boolean;
  geoip_database_installed: boolean;
}

export interface AnnouncingPeer {
  ip: string;
  port: number | null;
  asn: number | null;
  first_seen_ms: number | null;
  last_seen_ms: number | null;
  observers: number;
}

export interface CorrelatedTransaction {
  txid: string;
  /** How many DISTINCT peers announced it. Never read one peer as "the sender". */
  announcing_peers_total: number;
  peers_shown: number;
  first_seen_ms: number | null;
  addresses: { address: string; role: "input" | "output" }[];
  peers: AnnouncingPeer[];
}

export interface Correlation {
  category: Category;
  status: Category;
  available: boolean;
  transactions: CorrelatedTransaction[];
  summary: {
    transactions: number;
    announcing_peers: number;
    asns: number;
    observers: number;
    geo: GeoFacts;
  } | null;
  meaning: string;
  synthetic_warning?: string;
  insufficient_evidence_meaning: string;
}

export interface IngestValidation {
  source_format: string;
  rows_read: number;
  rows_valid: number;
  rows_rejected: number;
  columns_present: string[];
  columns_missing: string[];
  required_missing: string[];
  errors: string[];
  warnings: string[];
  ok: boolean;
}

export interface IngestResult {
  filename: string;
  bytes: number;
  validation: IngestValidation;
  correlation: {
    records: number;
    transactions: number;
    addresses: number;
    source_ips: number;
    asns: number;
    correlatable_records: number;
    time_span: { first: unknown; last: unknown; unit: string } | null;
  };
  preview: Record<string, unknown>[];
  canonical_columns: string[];
  next_step: { scored: boolean; why: string; command: string };
}

export interface AlertDetail {
  alert_id: string;
  run_fingerprint: string;
  summary: AlertSummary;
  meaning: string;
  score_scope: string;
  risk: {
    score: number | null;
    severity: Severity;
    ranking_aggregation: string;
    aggregations: Record<string, number | null>;
    top_member_risk: number | null;
  };
  why_flagged: WhyFlagged;
  evidence: Evidence;
  relationships: Relationships;
  timeline: Timeline;
  network_context: NetworkContext;
  correlation: Correlation;
  members: Members;
  provenance: Provenance;
}

export interface AlertFilters {
  severity?: Severity[];
  minRisk?: number;
  maxRisk?: number;
  firstTimestep?: number;
  lastTimestep?: number;
  limit?: number;
  offset?: number;
}

/** The error taxonomy the backend actually returns. */
export type ApiErrorKind =
  | "alert_id_invalid"
  | "alert_id_stale"
  | "alert_not_found"
  | "alert_filter_invalid"
  | "provenance_refused"
  | "artifact_missing"
  | "network"
  | "unknown";
