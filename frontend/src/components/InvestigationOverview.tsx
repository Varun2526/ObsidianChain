/**
 * Investigation overview — the page that has to answer, without ambiguity:
 *
 *   What case am I in?              case label, name, description
 *   Who owns it?                    owner
 *   What dataset is it based on?    every stored dataset, by hash
 *   Was it analysed?                that dataset's AnalysisRun status
 *   Which run produced results?     the case's bound run vs. the artifact
 *   How many alerts are involved?   REFERENCED alerts only
 *   What has been reviewed?         dispositions by state
 *   What remains?                   outstanding count
 *
 * The number that is deliberately absent is the global artifact's alert
 * count. It is a property of the pipeline's frozen dataset, not of this
 * investigation, and showing it here is what made an upload look scored.
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import * as api from "../api/console";
import type { CaseAlertRow, DispositionState } from "../api/types";
import { useAuth } from "../store/auth";
import { useCase } from "../store/investigation";
import { DispositionBadge, RunStatusChip, StaleRunBanner } from "./CaseChrome";

const STATES: DispositionState[] = [
  "NEW", "TRIAGED", "IN_REVIEW", "CONFIRMED", "DISMISSED", "ESCALATED",
];

const NEXT_STATUS: Record<string, string[]> = {
  DRAFT: ["VALIDATING", "ACTIVE", "CLOSED"],
  VALIDATING: ["ACTIVE", "DRAFT", "CLOSED"],
  ACTIVE: ["REVIEW", "CLOSED"],
  REVIEW: ["ACTIVE", "CLOSED"],
  CLOSED: [],
};

export function InvestigationOverview() {
  const { invId = "" } = useParams();
  const { investigation: inv, reload } = useCase();
  const { can } = useAuth();

  const [alerts, setAlerts] = useState<CaseAlertRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    api.listCaseAlerts(invId, controller.signal)
      .then((r) => setAlerts(r.alerts))
      .catch(() => {})
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [invId]);

  // Indexed with a fallback: the status comes from the server and a value
  // this build does not know must not crash the page.
  const nextStatuses = NEXT_STATUS[inv.status] ?? [];
  const run = inv.analytical_run;
  const summary = inv.summary;
  const stale = alerts.filter((a) => a.stale === true).length;

  const changeStatus = async (status: string) => {
    setBusy(true);
    try {
      await api.setInvestigationStatus(invId, status);
      await reload();
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-header">
        <div>
          <h1>{inv.name}</h1>
          <p className="muted">
            {inv.case_label} · owned by{" "}
            {inv.owner?.display_name ?? inv.owner?.username ?? inv.owner_id}
            {inv.description ? ` · ${inv.description}` : ""}
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span className={`status-badge status-${inv.status.toLowerCase()}`}>
            {inv.status}
          </span>
          <Link to={`/inv/${invId}/report`} className="btn btn-sm">Report</Link>
        </div>
      </div>

      {run?.status === "STALE" && (
        <StaleRunBanner
          bound={run.bound_run_fingerprint}
          current={run.current_artifact_run}
          count={stale}
        />
      )}

      {/* Case-owned counts. */}
      <div className="card-grid-4" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <span className="stat-card-value">{summary?.alerts_referenced ?? 0}</span>
          <span className="stat-card-label">Alerts in this case</span>
        </div>
        <div className="stat-card stat-card--high">
          <span className="stat-card-value">{summary?.outstanding ?? 0}</span>
          <span className="stat-card-label">Awaiting a decision</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">
            {summary?.dispositions_by_state.CONFIRMED ?? 0}
          </span>
          <span className="stat-card-label">Confirmed</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{summary?.notes ?? 0}</span>
          <span className="stat-card-label">Notes</span>
        </div>
      </div>

      {/* Analytical provenance */}
      <section className="panel">
        <div className="panel-head">
          <h2>Analytical run</h2>
          <RunStatusChip status={run?.status ?? "UNBOUND"} />
        </div>
        <div className="panel-body">
          <dl className="kv">
            <dt>Bound run</dt>
            <dd className="mono">
              {run?.bound_run_fingerprint ?? (
                <span className="faint">
                  not bound — no alert has been referenced into this case yet
                </span>
              )}
            </dd>
            <dt>Artifact on disk</dt>
            <dd className="mono">
              {run?.current_artifact_run ?? (
                <span className="faint">unavailable</span>
              )}
            </dd>
          </dl>
          <p className="note">{run?.meaning}</p>
        </div>
      </section>

      {/* Datasets and their analysis state */}
      <section className="panel">
        <div className="panel-head">
          <h2>Datasets</h2>
          <span className="small muted">{inv.datasets?.length ?? 0} stored</span>
        </div>
        {!inv.datasets?.length ? (
          <div className="panel-body">
            <p className="muted">
              No dataset has been uploaded to this investigation.
            </p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>File</th><th>Format</th><th className="num">Bytes</th>
                  <th>SHA-256</th><th>Validation</th><th>Analysis</th>
                  <th>Alerts from it</th>
                </tr>
              </thead>
              <tbody>
                {inv.datasets.map((d) => (
                  <tr key={d.id}>
                    <td>{d.filename}</td>
                    <td className="small muted">{d.format.toUpperCase()}</td>
                    <td className="num">{d.size_bytes.toLocaleString()}</td>
                    <td className="mono small">{d.sha256.slice(0, 16)}…</td>
                    <td>
                      <span className={`status-badge status-${
                        d.status === "VALIDATED" ? "active" : "closed"}`}>
                        {d.status}
                      </span>
                    </td>
                    <td className="mono small">
                      {d.analysis_run?.status ?? "NOT_RUN"}
                    </td>
                    <td className="small faint">
                      {d.analysis_run?.produced_alerts
                        ? d.analysis_run.run_fingerprint
                        : "none"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {inv.datasets?.some((d) => d.analysis_run?.status === "NOT_RUN") && (
          <div className="panel-body">
            <p className="note" style={{ marginTop: 0 }}>
              {inv.datasets.find((d) => d.analysis_run?.status === "NOT_RUN")
                ?.analysis_run?.meaning}
            </p>
          </div>
        )}
      </section>

      {/* Referenced alerts — this case's own */}
      <section className="panel">
        <div className="panel-head">
          <h2>Alerts referenced by this case</h2>
          <Link to={`/inv/${invId}/alerts`} className="small muted">View all →</Link>
        </div>
        {loading ? (
          <div className="panel-body"><p className="muted">Loading…</p></div>
        ) : alerts.length === 0 ? (
          <div className="panel-body">
            <p className="muted">
              No alerts have been referenced into this investigation yet. Open
              an alert from the{" "}
              <Link to={`/inv/${invId}/alerts`}>alerts page</Link> and add it.
            </p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>Alert</th><th>Investigator decision</th>
                  <th>Assigned</th><th>Run</th><th>Added</th><th />
                </tr>
              </thead>
              <tbody>
                {alerts.slice(0, 10).map((a) => (
                  <tr key={a.alert_id}>
                    <td className="mono small">{a.alert_id}</td>
                    <td><DispositionBadge state={a.disposition?.state} /></td>
                    <td className="small muted">
                      {a.assigned_to_display_name ?? a.assigned_to_username ?? "—"}
                    </td>
                    <td>
                      {a.stale === true ? (
                        <span className="runchip runchip-stale">STALE</span>
                      ) : a.stale === null ? (
                        <span className="runchip runchip-unverifiable">unverifiable</span>
                      ) : (
                        <span className="runchip runchip-current">current</span>
                      )}
                    </td>
                    <td className="small muted">
                      {new Date(a.added_at).toLocaleDateString()}
                    </td>
                    <td>
                      <Link to={`/inv/${invId}/alerts/${a.alert_id}`}
                        className="btn btn-sm">Open</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Triage progress */}
      {summary && summary.alerts_referenced > 0 && (
        <section className="panel">
          <div className="panel-head"><h2>Review progress</h2></div>
          <div className="panel-body">
            <div className="severity-bars">
              {STATES.map((state) => {
                const count = summary.dispositions_by_state[state] ?? 0;
                const pct = summary.alerts_referenced
                  ? (count / summary.alerts_referenced) * 100 : 0;
                return (
                  <div key={state} className="severity-bar-row">
                    <span className="severity-bar-label">{state.replace("_", " ")}</span>
                    <div className="severity-bar-track">
                      <div className={`severity-bar-fill disp-fill-${state}`}
                        style={{ width: `${pct}%` }} />
                    </div>
                    <span className="severity-bar-count mono">{count}</span>
                  </div>
                );
              })}
            </div>
            <p className="note">
              These are INVESTIGATOR decisions. They are recorded by named
              people and are separate from the model's severity bands, which
              appear on each alert.
            </p>
          </div>
        </section>
      )}

      {/* Lifecycle */}
      {can("change_investigation_status") && nextStatuses.length > 0 && (
        <section className="panel">
          <div className="panel-head"><h2>Case status</h2></div>
          <div className="panel-body">
            <p className="muted small" style={{ marginTop: 0 }}>
              Status changes are explicit and audited. It never changes because
              a page was opened.
            </p>
            <div className="form-actions">
              {nextStatuses.map((s) => (
                <button key={s} className="btn btn-sm" disabled={busy}
                  onClick={() => changeStatus(s)}>
                  Move to {s}
                </button>
              ))}
            </div>
          </div>
        </section>
      )}
    </>
  );
}
