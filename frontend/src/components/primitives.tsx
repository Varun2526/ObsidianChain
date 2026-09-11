/**
 * Small shared pieces. Each one exists to keep a backend distinction visible
 * in the UI rather than smoothing it away.
 */
import type { Category, Severity } from "../api/types";

export function SeverityBadge({ severity }: { severity: Severity }) {
  return <span className={`sev sev-${severity}`}>{severity}</span>;
}

export function CategoryChip({ category }: { category: Category }) {
  return (
    <span className={`cat cat-${category}`} title={describeCategory(category)}>
      {category.replace(/_/g, " ")}
    </span>
  );
}

export function describeCategory(category: Category): string {
  switch (category) {
    case "MODEL_SIGNAL":
      return "A SHAP contribution: how much this feature moved the model's score. A statement about the model, not about Bitcoin.";
    case "BLOCKCHAIN_CONTEXT":
      return "An observed on-chain quantity. True of the ledger, independent of any model.";
    case "NETWORK_CONTEXT":
      return "A summary of synthetic announcement observations. Investigative context only; never an ownership or identity claim.";
    case "INSUFFICIENT_EVIDENCE":
      return "Not computable for this address. Reported as insufficient evidence rather than as zero.";
  }
}

const SEVERITY_COLOUR: Record<Severity, string> = {
  CRITICAL: "var(--critical)",
  HIGH: "var(--high)",
  MEDIUM: "var(--medium)",
  LOW: "var(--low)",
};

export function RiskBar({ value, severity }: { value: number | null; severity: Severity }) {
  if (value === null) return <span className="faint">—</span>;
  return (
    <span className="riskbar" title={`${(value * 100).toFixed(2)}%`}>
      <span
        style={{ width: `${Math.max(1, Math.min(100, value * 100))}%`,
                 background: SEVERITY_COLOUR[severity] }}
      />
    </span>
  );
}

/**
 * A number, or an explicit absence. Never a zero standing in for a null:
 * the backend is careful to distinguish "not measured" from "measured as
 * none", and rendering both as 0 would throw that away at the last step.
 */
export function Value({ value, digits = 4, suffix = "" }:
  { value: number | boolean | null | undefined; digits?: number; suffix?: string }) {
  if (value === null || value === undefined) {
    return <span className="faint" title="Not computable for this address">n/a</span>;
  }
  if (typeof value === "boolean") return <>{value ? "yes" : "no"}</>;
  const text = Number.isInteger(value) ? String(value) : value.toFixed(digits);
  return <>{text}{suffix}</>;
}

export function Address({ value, truncate = true }: { value: string; truncate?: boolean }) {
  const shown = truncate && value.length > 20
    ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
  return <span className="mono" title={value}>{shown}</span>;
}

export function Skeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="panel-body" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${95 - i * 9}%` }} />
      ))}
    </div>
  );
}

export function EmptyState({ title, children }:
  { title: string; children?: React.ReactNode }) {
  return (
    <div className="state">
      <h3>{title}</h3>
      {children ? <p>{children}</p> : null}
    </div>
  );
}
