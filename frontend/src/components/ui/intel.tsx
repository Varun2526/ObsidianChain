/**
 * Shared pieces for the intelligence pages. Each one keeps a distinction
 * visible: observed vs attributed vs model output, and research vs
 * production numbers.
 */
import { useState } from "react";
import { Link } from "react-router-dom";

import type { ClusterAnnotation, ModelAnnotation, ResultType, WatchlistHit } from "../../api/intel";
import type { Severity } from "../../api/types";
import { Icon } from "./Icon";

export type EvidenceKind = "chain" | "model" | "network" | "watchlist" | "rule" | "heuristic";

const EVIDENCE_LABEL: Record<EvidenceKind, string> = {
  chain: "On-chain",
  model: "Model",
  network: "Network",
  watchlist: "Attribution",
  rule: "Rule",
  heuristic: "Heuristic",
};

const EVIDENCE_TITLE: Record<EvidenceKind, string> = {
  chain: "Observed on the ledger. True independent of any model.",
  model: "A learned association from a model. A lead to investigate, not proof of illicit activity.",
  network: "Peer-to-peer relay observation. A vantage point, never the sender. Synthetic in this deployment.",
  watchlist: "Asserted by the named external source. Not a finding of this system.",
  rule: "A deterministic rule fired.",
  heuristic: "An inference such as common-input ownership. Can merge unrelated owners.",
};

export function EvidenceTag({ kind, children }: { kind: EvidenceKind; children?: React.ReactNode }) {
  return (
    <span className={`ev ev-${kind}`} title={EVIDENCE_TITLE[kind]}>
      {children ?? EVIDENCE_LABEL[kind]}
    </span>
  );
}

const RESULT_TITLE: Record<ResultType, string> = {
  DEVELOPMENT: "Measured on the folds used to select the model. Optimistic by construction.",
  CONFIRMATION: "Measured on later folds that played no part in selection.",
  HOLDOUT: "Measured once on sealed timesteps t42-49 after the model was frozen.",
  PRODUCTION: "Measured on labelled production traffic.",
};

export function ResultTypeTag({ type }: { type: ResultType | "SYNTHETIC" }) {
  return (
    <span className={`rtype rtype-${type}`} title={type === "SYNTHETIC" ? "Generated world, not Bitcoin." : RESULT_TITLE[type]}>
      {type}
    </span>
  );
}

export function SevTag({ severity }: { severity: Severity | string }) {
  return <span className={`sev sev-${severity}`}>{severity}</span>;
}

export function short(value: string, head = 8, tail = 6): string {
  return value.length > head + tail + 3 ? `${value.slice(0, head)}…${value.slice(-tail)}` : value;
}

export function AddressLink({ address, full = false }: { address: string; full?: boolean }) {
  return (
    <Link to={`/entity/${encodeURIComponent(address)}`} className="mono row-link" title={address}>
      {full ? address : short(address)}
    </Link>
  );
}

export function TxLink({ txid }: { txid: number | string }) {
  return <Link to={`/tx/${txid}`} className="mono row-link">{txid}</Link>;
}

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="btn btn-sm btn-ghost"
      aria-label={`${label}: ${value}`}
      onClick={() => {
        void navigator.clipboard?.writeText(value).then(() => {
          setDone(true);
          window.setTimeout(() => setDone(false), 1200);
        }).catch(() => {});
      }}
    >
      <Icon name="copy" size={14} />
      <span>{done ? "Copied" : label}</span>
    </button>
  );
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v == null || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(digits)}%`;
}

export function fixed(v: unknown, digits = 3): string {
  if (v == null || (typeof v === "number" && Number.isNaN(v))) return "n/a";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (v === "") return "none";
  return typeof v === "number" ? v.toFixed(digits) : String(v);
}

export function btc(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "n/a";
  return `${v.toLocaleString(undefined, { maximumFractionDigits: 8 })} BTC`;
}

export function int(v: number | null | undefined): string {
  return v == null ? "n/a" : v.toLocaleString();
}

/** What the model, the clustering heuristic and a watchlist each say about one address. */
export function AnnotationChips({ model, cluster, watchlist, compact = false }: {
  model?: ModelAnnotation | null; cluster?: ClusterAnnotation | null; watchlist?: WatchlistHit[]; compact?: boolean;
}) {
  const items: React.ReactNode[] = [];
  for (const w of watchlist ?? []) {
    items.push(<EvidenceTag key={`w${w.source}`} kind="watchlist">{w.source.replace("WATCHLIST:", "")}{w.label && !compact ? ` · ${w.label}` : ""}</EvidenceTag>);
  }
  if (model) {
    items.push(
      <span key="m" className="chips">
        <EvidenceTag kind="model">Model {pct(model.risk_score)}</EvidenceTag>
        <SevTag severity={model.severity} />
      </span>,
    );
  }
  if (cluster && !compact) {
    items.push(<EvidenceTag key="c" kind="heuristic">Cluster {cluster.cluster_id} · {int(cluster.size)}</EvidenceTag>);
  }
  if (items.length === 0) return compact ? null : <span className="faint small">No model score, attribution or cluster</span>;
  return <span className="chips">{items}</span>;
}

export function PageHeader({ eyebrow, title, sub, actions }: {
  eyebrow?: React.ReactNode; title: React.ReactNode; sub?: React.ReactNode; actions?: React.ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Metric({ k, v, d, tone }: { k: string; v: React.ReactNode; d?: React.ReactNode; tone?: string }) {
  return (
    <div className="metric">
      <span className="metric-k">{k}</span>
      <span className="metric-v" style={tone ? { color: tone } : undefined}>{v}</span>
      {d && <span className="metric-d">{d}</span>}
    </div>
  );
}
