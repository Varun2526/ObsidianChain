/**
 * Results of an uploaded-dataset run.
 *
 * Three kinds of number appear here and are labelled so they are never read
 * as one another:
 *   - the MODEL's holdout result (measured once, on Elliptic++ t42-49);
 *   - this RUN's alerts, whose precision is unknown until labels arrive;
 *   - monitoring notices about this run's inputs.
 * Evidence is grouped by class (MODEL / RULE / NETWORK / WATCHLIST) so a
 * learned association is never presented as a rule or an observation.
 */
import { Fragment, useEffect, useState } from "react";

import * as api from "../../api/console";
import type { EvidenceClass, RunResults, RunSeverity } from "../../api/types";
import { ErrorState } from "../ui/ErrorState";
import { Address, Skeleton } from "../ui/primitives";
import { RunGraphPanel, RunNetworkPanel } from "./RunNetworkPanel";

const CLASS_LABEL: Record<EvidenceClass, string> = {
  MODEL: "Model (learned association)",
  RULE: "Rule",
  NETWORK: "Network observation",
  WATCHLIST: "Watchlist link",
  CONTEXT: "Context",
};

function Sev({ severity }: { severity: RunSeverity }) {
  return <span className={`sev sev-${severity}`}>{severity}</span>;
}

export function RunResultsPanel({ investigationId, runId, showNetwork = true }:
  { investigationId: string; runId: string; showNetwork?: boolean }) {
  const [data, setData] = useState<RunResults | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    api.getRunResults(investigationId, runId, 50, ctrl.signal)
      .then(setData)
      .catch((cause) => {
        if (!ctrl.signal.aborted) setError(cause instanceof Error ? cause : new Error(String(cause)));
      });
    return () => ctrl.abort();
  }, [investigationId, runId]);

  if (error) return <ErrorState error={error} />;
  if (!data) return <div className="panel" style={{ marginTop: 20 }}><Skeleton rows={4} /></div>;

  const holdout = data.model.holdout_result;
  return (
    <section className="panel" style={{ marginTop: 20 }} aria-label="Run results">
      <div className="panel-head"><h2>Ranked alerts for this run</h2></div>
      <div className="panel-body">
        <div className="card-grid-4" style={{ marginBottom: 14 }}>
          <div className="stat-mini">
            <span className="stat-mini-v mono" style={{ fontSize: "0.9rem" }}>{data.model.version ?? "none"}</span>
            <span className="stat-mini-k">Model served ({data.ml_status})</span>
          </div>
          <div className="stat-mini">
            <span className="stat-mini-v">{data.total_alerts.toLocaleString()}</span>
            <span className="stat-mini-k">Entities ranked</span>
          </div>
          <div className="stat-mini">
            <span className="stat-mini-v">{holdout ? holdout.nap.toFixed(3) : "—"}</span>
            <span className="stat-mini-k" title={data.model.holdout_result_type}>Model holdout nAP</span>
          </div>
          <div className="stat-mini">
            <span className="stat-mini-v">{holdout ? holdout["P@100"].toFixed(2) : "—"}</span>
            <span className="stat-mini-k" title={data.model.holdout_result_type}>Model holdout P@100</span>
          </div>
        </div>
        <p className="small muted" style={{ marginTop: 0 }}>
          The holdout figures describe the model on Elliptic++ t42-49, not this run.{" "}
          <strong>{data.run_result_type}.</strong> On the holdout, precision varied by window from 2% to 100%.
        </p>

        {data.monitoring_alerts.length > 0 && (
          <ul className="small" style={{ margin: "0 0 14px", paddingLeft: 18 }} aria-label="Monitoring notices">
            {data.monitoring_alerts.map((a) => (
              <li key={a.code}><strong>{a.severity} {a.code.replace(/_/g, " ").toLowerCase()}:</strong> {a.detail}</li>
            ))}
          </ul>
        )}

        <table className="table" style={{ width: "100%" }}>
          <thead>
            <tr><th>#</th><th>Severity</th><th>Lead address</th><th>Members</th><th>Fused score</th><th>Agreeing lines</th><th></th></tr>
          </thead>
          <tbody>
            {data.alerts.map((a) => (
              <Fragment key={a.alert_id}>
                <tr>
                  <td>{a.rank}</td>
                  <td><Sev severity={a.severity} /></td>
                  <td><Address value={a.primary_address} /></td>
                  <td>{a.member_count}</td>
                  <td className="mono">{a.fused_risk_score.toFixed(3)}</td>
                  <td>{a.summary.corroborating_evidence_lines ?? 0}</td>
                  <td>
                    <button className="btn btn-sm" onClick={() => setOpen(open === a.alert_id ? null : a.alert_id)}>
                      {open === a.alert_id ? "Hide" : "Why"}
                    </button>
                  </td>
                </tr>
                {open === a.alert_id && (
                  <tr>
                    <td colSpan={7}>
                      <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
                        {a.evidence.filter((e) => e.status !== "NO_EVIDENCE").map((e, i) => (
                          <li key={i}><strong>{CLASS_LABEL[e.evidence_class] ?? e.evidence_class}:</strong> {e.explanation}</li>
                        ))}
                      </ul>
                      {a.explanation_statement && <p className="small faint" style={{ marginBottom: 0 }}>{a.explanation_statement}</p>}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
        <p className="small faint">Run {data.run_id} · input sha256 {String(data.input_sha256).slice(0, 16)}…</p>
      </div>
      {showNetwork && (
        <div style={{ padding: "0 16px 16px" }}>
          <RunNetworkPanel investigationId={investigationId} runId={runId} />
          <RunGraphPanel investigationId={investigationId} runId={runId} />
        </div>
      )}
    </section>
  );
}
