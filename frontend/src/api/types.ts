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
  exact_duplicates_rejected?: number;
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
  | "separation_basis_mismatch"
  // ---- application layer (obsidianchain.console) ----
  | "authentication_required"
  | "invalid_credentials"
  | "session_expired"
  | "access_denied"
  | "investigation_not_found"
  | "not_found"
  | "validation_failed"
  | "conflict"
  | "run_mismatch"
  | "upload_rejected"
  | "append_only"
  | "console_error"
  | "network"
  | "unknown";

/* ==========================================================================
   APPLICATION LAYER - mutable investigator state
   ==========================================================================
   Everything above this line describes IMMUTABLE ANALYTICAL TRUTH read from
   pipeline artifacts. Everything below describes state recorded by a named
   person in SQLite. The two are never merged in a response and are never
   merged here.
   ========================================================================== */

export type Role = "ADMIN" | "INVESTIGATOR" | "REVIEWER";

export type InvestigationStatus =
  | "DRAFT"
  | "VALIDATING"
  | "ANALYZING"
  | "ACTIVE"
  | "SUBMITTED"
  | "IN_REVIEW"
  | "APPROVED"
  | "RETURNED"
  | "CLOSED"
  | "ARCHIVED"
  | "REVIEW";

export type DispositionState =
  | "NEW" | "TRIAGED" | "IN_REVIEW" | "CONFIRMED" | "DISMISSED" | "ESCALATED";

/** How a case's stored analytical run compares to the artifact on disk. */
export type RunStatus = "UNBOUND" | "CURRENT" | "STALE" | "UNVERIFIABLE";

export type AnalysisRunStatus =
  | "NOT_RUN" | "QUEUED" | "RUNNING" | "COMPLETE" | "FAILED";

export interface AccountRef {
  id: string;
  username: string;
  display_name: string;
}

export interface Identity {
  user: AccountRef & {
    role: Role;
    active: boolean;
    created_at: string;
    last_login_at?: string | null;
  };
  /** For RENDERING only. The backend re-derives and re-checks every request. */
  capabilities: string[];
}

export interface AnalysisRun {
  id: string;
  dataset_id: string;
  status: AnalysisRunStatus;
  run_fingerprint: string | null;
  produced_alerts: boolean;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  /** Present only while NOT_RUN: why, and the offline command that changes it. */
  meaning?: string;
  command?: string;
}

export interface UploadedDataset {
  id: string;
  investigation_id: string;
  filename: string;
  sha256: string;
  size_bytes: number;
  format: string;
  uploaded_by: string;
  uploaded_at: string;
  status: "RECEIVED" | "VALIDATED" | "REJECTED";
  validation: IngestResult | Record<string, unknown>;
  analysis_run?: AnalysisRun | null;
}

export interface CaseSummary {
  alerts_referenced: number;
  high_risk_alerts?: number;
  dispositions_by_state: Record<DispositionState, number>;
  outstanding: number;
  notes: number;
  dataset_count?: number;
  dataset_status?: string | null;
  analysis_status?: string | null;
  last_activity_at?: string | null;
}

export interface AnalyticalRunBinding {
  bound_run_fingerprint: string | null;
  bound_run_at?: string | null;
  current_artifact_run: string | null;
  status: RunStatus;
  meaning?: string;
}

export interface Investigation {
  id: string;
  case_number: number;
  case_label: string;
  name: string;
  description: string;
  owner_id: string;
  owner?: AccountRef;
  status: InvestigationStatus;
  bound_run_fingerprint: string | null;
  created_at: string;
  updated_at: string;
  closed_at: string | null;
  analytical_run?: AnalyticalRunBinding;
  datasets?: UploadedDataset[];
  summary?: CaseSummary;
  run_status?: RunStatus;
}

export interface Disposition {
  id: string;
  state: DispositionState;
  rationale: string;
  decided_by: string;
  decided_by_username: string | null;
  decided_by_display_name: string | null;
  decided_at: string;
  superseded_by: string | null;
  active: boolean;
}

export interface InvestigatorNote {
  id: string;
  investigation_id: string;
  alert_id: string | null;
  body: string;
  author_id: string;
  author_username: string | null;
  author_display_name: string | null;
  created_at: string;
  updated_at: string | null;
}

export interface CaseAlertRow {
  alert_id: string;
  run_fingerprint: string;
  added_by: string;
  added_by_username: string | null;
  added_at: string;
  assigned_to: string | null;
  assigned_to_username: string | null;
  assigned_to_display_name: string | null;
  assigned_at: string | null;
  disposition: Disposition | null;
  /** null means the artifact could not be read, which is NOT the same as false. */
  stale: boolean | null;
}

/**
 * The analytical half of a case-scoped alert.
 *
 * Never silently absent. When the reference has gone stale, or the artifact
 * cannot be read, `available` is false and `reason` says which - the failure
 * the old report page hid behind a swallowed 409.
 */
export type CaseAlertAnalytical =
  | { available: true; alert: AlertDetail }
  | {
      available: false;
      reason: "STALE_REFERENCE" | "ARTIFACT_UNAVAILABLE" | "ALERT_NOT_IN_RUN";
      referenced_run: string;
      current_artifact_run: string | null;
      detail: string;
    };

export interface CaseAlertInvestigator {
  reference: {
    alert_id: string;
    run_fingerprint: string;
    added_by: string;
    added_at: string;
  };
  assignment: {
    assigned_to: string | null;
    assigned_at: string | null;
    assignee: AccountRef | null;
  };
  disposition: Disposition | null;
  disposition_history: Disposition[];
  notes: InvestigatorNote[];
  states: DispositionState[];
  meaning: string;
}

export interface CaseAlertDetail {
  alert_id: string;
  investigation_id: string;
  analytical: CaseAlertAnalytical;
  investigator: CaseAlertInvestigator;
}

export interface StoredReport {
  id: string;
  investigation_id: string;
  version: number;
  title: string;
  executive_summary: string;
  content: string;
  run_fingerprint: string | null;
  generated_by: string;
  generated_at: string;
  content_sha256: string;
  status: "DRAFT" | "FINAL";
  finalised_by: string | null;
  finalised_at: string | null;
}

export interface ReportPayload {
  case: Investigation;
  analytical_run: AnalyticalRunBinding;
  report: StoredReport | null;
  versions: {
    id: string;
    version: number;
    title: string;
    status: "DRAFT" | "FINAL";
    generated_at: string;
    content_sha256: string;
    generated_by_username: string | null;
  }[];
  alert_references: CaseAlertRow[];
  stale_references: (CaseAlertRow & { warning: string })[];
  unverifiable_references: CaseAlertRow[];
  notes: InvestigatorNote[];
  summary: CaseSummary;
  disposition_meaning: string;
  separation_note: string;
  may_finalise: boolean;
}

export interface AuditEvent {
  id: number;
  actor_id: string | null;
  actor_username: string | null;
  actor_display_name: string | null;
  action: string;
  object_type: string;
  object_id: string | null;
  investigation_id: string | null;
  at: string;
  detail: Record<string, unknown>;
}

/**
 * Network-separation records for one cluster's proposed merges.
 *
 * The vocabulary is the pipeline's own and is not simplified here:
 * SEPARATED is the only verdict carrying a constraint, and that constraint
 * is CANNOT-LINK. NOT_SEPARATED and NO_EVIDENCE are not evidence of common
 * ownership, and the backend's frozen wordings travel with the payload so
 * the UI cannot paraphrase them into one.
 */
export interface SeparationRow {
  evidence_id: string;
  edge_index: number;
  node_a: number;
  node_b: number;
  verdict: "SEPARATED" | "NOT_SEPARATED" | "NO_EVIDENCE" | string;
  reason_code: string;
  reason: string;
  min_pooled: number | null;
  size_a: number | null;
  size_b: number | null;
  chi2: number | null;
  p_value: number | null;
  effect: number | null;
}

export interface SeparationEvidence {
  alert_id: string;
  cluster_id: number;
  alert_run_fingerprint: string;
  evidence_run_fingerprint: string;
  statement: string;
  join_basis: string;
  proposed_merges_total: number;
  verdicts: Record<string, number>;
  reason_codes: Record<string, number>;
  separated_count: number;
  cannot_link_meaning: string;
  not_separated_meaning: string;
  verdict_scope: string;
  verdict_definition: string;
  frozen_run_limitation: string;
  reason_code_catalogue: Record<string, string>;
  unreachable_reason_codes: Record<string, string>;
  rows_shown: number;
  rows_withheld: number;
  rows: SeparationRow[];
}

/* ==========================================================================
   STRUCTURAL PATTERNS - peeling and mixing, behind one alert
   ========================================================================== */

/**
 * Which evidence layer the investigator is looking at.
 *
 * A VIEW mode, not three models. One analytical run produced everything
 * below; the toggle selects which already-fetched evidence is shown and
 * recomputes nothing.
 */
export type AnalysisLayer = "CHAIN" | "NETWORK" | "FUSED";

export type MixingClass =
  | "MIXING_PATTERN" | "MIXING_LIKELIHOOD" | "NO_MIXING_SIGNAL"
  | "INSUFFICIENT_DATA";

export interface MixingTransaction {
  txid: string;
  mixing_class: MixingClass;
  mixing_score: number | null;
  output_uniformity: number | null;
  input_heterogeneity: number | null;
  participant_symmetry: number | null;
  /** Why a structurally-similar transaction was NOT called a pattern. */
  suppressor: string | null;
  n_inputs: number | null;
  n_outputs: number | null;
}

export type PeelingBlock =
  | {
      available: true;
      members_scored: number;
      members_in_chain: number;
      peel_chain_members_recorded: number;
      max_chain_depth: number | null;
      fields: string[];
      members: Record<string, string | number | null>[];
      meaning: string;
    }
  | { available: false; status: string; detail: string; meaning: string };

export type MixingBlock =
  | {
      available: true;
      scan_id: string;
      detector: string | null;
      join_basis: string;
      transactions_measured: number;
      transactions_correlated: number;
      classes: Record<string, number>;
      pattern_count: number;
      suppressed: Record<string, number>;
      suppressor_meanings: Record<string, string>;
      transactions: MixingTransaction[];
      meaning: string;
      insufficient_data_meaning: string;
    }
  | { available: false; status: string; detail: string; meaning: string };

export interface AlertPatterns {
  alert_id: string;
  cluster_id: number;
  run_fingerprint: string;
  peeling: PeelingBlock;
  mixing: MixingBlock;
  category: Category;
}

export interface SavedFilter {
  id: string;
  user_id: string;
  investigation_id: string | null;
  name: string;
  filter_json: string;
  created_at: string;
}

export interface RelatedAlert {
  alert_id: string;
  cluster_id: number;
  rank: number;
  severity: Severity;
  risk_score: number | null;
  top_signals: string[];
  relationship_type: "SHARED_TRANSACTION" | "OBSERVED_SHARED_PEER" | string;
  connecting_identifier: string;
  detail: string;
  limitation: string;
}

export interface RelatedAlertsResponse {
  alert_id: string;
  run_fingerprint: string;
  count: number;
  related_alerts: RelatedAlert[];
  meaning: string;
}

export interface TransactionDrilldown {
  txid: number;
  input_count: number;
  output_count: number;
  inputs: { address: string; alert_id: string }[];
  outputs: { address: string; alert_id: string }[];
  associated_clusters: string[];
  announcing_peers: {
    ip: string;
    port: number | null;
    asn: number | null;
    observers: number;
    announcing_peers: number;
  }[];
  mixing: {
    available: boolean;
    mixing_class?: string;
    entropy?: number | null;
    n_inputs?: number;
    n_outputs?: number;
    meaning?: string;
  };
  limitation: string;
}

export interface RunStageInfo {
  stage_number: number;
  stage_name: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
  progress_pct: number;
  summary?: string;
}

export interface RunProgressResponse {
  run_id?: string;
  status: "NOT_RUN" | "RUNNING" | "COMPLETE" | "FAILED";
  current_stage: number;
  total_stages: number;
  stage_name: string;
  stage_status: string;
  progress_pct: number;
  stages?: RunStageInfo[];
  summary?: {
    alerts_generated?: number;
    entities_clustered?: number;
    correlations_identified?: number;
    run_fingerprint?: string;
    completed_at?: string;
  };
  run_fingerprint?: string | null;
  alerts_count?: number;
  error?: string | null;
}

export interface UserAccount {
  id: string;
  username: string;
  display_name: string;
  role: Role;
  active: boolean;
  created_at: string;
}

export interface ReviewDecisionPayload {
  decision: "APPROVE_FINDINGS" | "RETURN_FOR_CLARIFICATION" | "REQUEST_FURTHER_INVESTIGATION";
  rationale: string;
}


/** GET /investigations/{id}/runs/{run_id}/results - an uploaded-dataset run. */
export type RunSeverity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFORMATIONAL";
export type EvidenceClass = "MODEL" | "RULE" | "NETWORK" | "WATCHLIST" | "CONTEXT";

export interface RunEvidence {
  category: string;
  evidence_class: EvidenceClass;
  signal_name: string;
  status: "PRESENT" | "NO_EVIDENCE" | "UNAVAILABLE";
  score: number;
  explanation: string;
}

export interface RunAlert {
  alert_id: string;
  cluster_id: string;
  primary_address: string;
  member_count: number;
  fused_risk_score: number;
  severity: RunSeverity;
  rank: number;
  summary: { confidence?: number; corroborating_evidence_lines?: number } & Record<string, unknown>;
  explanation_statement?: string;
  evidence: RunEvidence[];
}

export interface RunMonitoringAlert {
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "INFO";
  code: string;
  detail: string;
}

export interface RunResults {
  run_id: string;
  run_fingerprint: string | null;
  created_at: string | null;
  input_sha256: string | null;
  ml_status: string;
  model: {
    version: string | null;
    feature_schema_version: string | null;
    holdout_result: { nap: number; "P@100": number; ece: number; sha256: string } | null;
    holdout_result_type: string;
  };
  run_result_type: string;
  monitoring_alerts: RunMonitoringAlert[];
  drift_relative_to_development: string | null;
  total_alerts: number;
  alerts: RunAlert[];
}
