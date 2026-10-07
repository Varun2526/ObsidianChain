/**
 * Plain-language names for the technical identifiers the API returns.
 *
 * The identifiers themselves (model versions, schema ids, statuses, feature
 * columns) stay unchanged everywhere they are stored and audited. These
 * functions only decide what a person reads; components keep the raw id
 * available as a tooltip so an auditor can still match it to the registry.
 */

/** ps_native_v5 -> "Risk Model 5"; ps_native_v5_fallback_no_g -> "Risk Model 5 (backup)". */
export function modelName(version: string | null | undefined): string {
  if (!version) return "No model";
  const m = /^ps_native_v(\d+)(_fallback_no_g)?$/.exec(version);
  if (!m) return version;
  return `Risk Model ${m[1]}${m[2] ? " (backup)" : ""}`;
}

/** The one-line description shown under a model name. */
export function modelBlurb(version: string | null | undefined): string {
  if (!version) return "";
  if (/_fallback_no_g$/.test(version)) {
    return "Backup model: serves only if the main model fails its integrity check. Leaves out the upstream-funding features.";
  }
  return "Gradient-boosted risk model over 31 on-chain features of an address.";
}

const ROLE: Record<string, string> = {
  champion: "In service",
  fallback: "Backup",
  candidate: "Under evaluation",
  CHAMPION: "In service",
  FALLBACK: "Backup",
  CANDIDATE: "Under evaluation",
};
export const modelRole = (role: string | null | undefined): string => (role ? ROLE[role] ?? title(role) : "Not serving");

/** ps_native_features/5 -> "Feature set 5". */
export function featureSetName(schema: string | null | undefined): string {
  if (!schema) return "n/a";
  const m = /^ps_native_features\/(\d+)$/.exec(schema);
  return m ? `Feature set ${m[1]}` : schema;
}

const STATUS: Record<string, string> = {
  FROZEN_PENDING_HOLDOUT: "Frozen and verified",
  PRODUCTION_HOLDOUT_PENDING: "Not yet tested on held-out data",
  WITHDRAWN: "Withdrawn",
  ACTIVE: "Active",
  PASS: "Passed",
  FAIL: "Failed",
  WARN: "Warning",
};
export const statusLabel = (status: string | null | undefined): string =>
  status ? STATUS[status] ?? title(status) : "n/a";

/** Evaluation-protocol ids to what they mean. */
export function protocolLabel(protocol: string | null | undefined): string {
  if (!protocol) return "n/a";
  if (/protocol_?b/i.test(protocol)) return "Time-split test on addresses never seen in training";
  return title(protocol);
}

const FEATURE: Record<string, string> = {
  input_count: "Number of inputs",
  output_count: "Number of outputs",
  total_input_amount: "Total input value (BTC)",
  fee: "Transaction fee (BTC)",
  fee_ratio: "Fee as a share of input value",
  input_amount_mean: "Average input value (BTC)",
  output_amount_mean: "Average output value (BTC)",
  input_spread: "Spread of input values",
  output_spread: "Spread of output values",
  n_txs_asof_t: "Transactions so far",
  n_sent_asof_t: "Payments sent so far",
  n_recv_asof_t: "Payments received so far",
  btc_recv_total_asof_t: "Total BTC received so far",
  net_flow_asof_t: "Net BTC flow so far",
  active_duration_seconds: "Time active",
  gap_since_last_tx: "Time since previous transaction",
  unique_counterparties_asof_t: "Distinct counterparties",
  cluster_size_asof_t: "Size of its co-spend cluster",
  is_peeling_candidate: "Looks like a peeling step",
  is_mixing_candidate: "Looks like a mixing transaction",
  addr_is_sender: "Address is the sender",
  addr_is_self_change: "Address receives its own change",
  counterparty_max_n_txs_asof_t: "Busiest counterparty's activity",
  counterparty_mean_n_txs_asof_t: "Counterparties' average activity",
  upstream_funded_share: "Share of inputs with a traceable source",
  upstream_mean_output_count: "Outputs of the funding transactions",
  upstream_mean_input_count: "Inputs of the funding transactions",
  upstream_peel_share: "Funding that came through peeling",
  upstream_mix_share: "Funding that came through mixing",
  upstream_min_hold_seconds: "Shortest time funds were held",
  upstream_chain_depth: "Length of the funding chain",
  network_observation_count: "Network observations",
  observer_diversity: "Distinct observers",
  peer_count: "Distinct relay peers",
  asn_count: "Distinct networks (ASNs)",
  dominant_peer_share: "Share from the top relay peer",
  arrival_spread_seconds: "Propagation spread",
  btc_sent: "BTC sent",
  btc_received: "BTC received",
  mean_fee_ratio: "Average fee share",
  tx_count: "Transaction count",
};
export const featureLabel = (name: string): string => {
  if (FEATURE[name]) return FEATURE[name];
  // "_asof_t" / "_asof" only say "computed as of that time"; drop them.
  const base = name.replace(/_asof(_t)?$/, "");
  return FEATURE[base] ?? title(base);
};

/** Replace raw feature column names inside a sentence with their plain names. */
export function humanizeFeatures(text: string): string {
  return text.replace(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g, (w) => (FEATURE[w] ? FEATURE[w] : w));
}

function title(raw: string): string {
  const words = raw.replace(/[_/]+/g, " ").trim().toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}
