/**
 * Investigation Overview — ServiceNow-style Investigation Canvas
 *
 * Core Case Workspace:
 * - Sticky case chrome with persistent tabs (Overview, Alerts, Graph, Timeline, Network, Evidence, Notes, Report, Review, Audit)
 * - Investigation Health Box (Data Integrity, Evidence State, Analytical Binding, Review Readiness)
 * - Priority Alerts Rail (Ranked by Risk Score with key signals and quick investigate link)
 * - Datasets & Analytical Run Binding
 * - Case Lifecycle & Review Controls
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import * as api from "../../api/console";
import type { CaseAlertRow } from "../../api/types";
import { useAuth } from "../../store/auth";
import { useCase } from "../../store/investigation";
import { DispositionBadge, RunStatusChip, StaleRunBanner } from "../../components/layout/CaseChrome";


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

  const run = inv.analytical_run;
  const summary = inv.summary;
  const stale = alerts.filter((a) => a.stale === true).length;
  const dataset = inv.datasets?.[0];

  const changeStatus = async (status: string) => {
    setBusy(true);
    try {
      await api.setInvestigationStatus(invId, status);
      await reload();
    } catch (cause) {
      alert(`Status update failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleArchive = async () => {
    if (!confirm(`Archive investigation ${inv.case_label}?`)) return;
    setBusy(true);
    try {
      await api.archiveInvestigation(invId);
      await reload();
    } catch (cause) {
      alert(`Archive failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleRestore = async () => {
    setBusy(true);
    try {
      await api.restoreInvestigation(invId);
      await reload();
    } catch (cause) {
      alert(`Restore failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    const confirmation = prompt(
      `EXCEPTIONAL ADMINISTRATIVE ACTION: Permanently purge case ${inv.case_label}?\n` +
      `Cannot delete an active case or one with recorded dispositions or sealed integrity proofs.\n` +
      `Type 'DELETE ${inv.case_label}' to confirm:`
    );
    if (confirmation !== `DELETE ${inv.case_label}`) {
      if (confirmation !== null) alert("Confirmation mismatch. Deletion cancelled.");
      return;
    }
    setBusy(true);
    try {
      await api.deleteInvestigation(invId, confirmation);
      window.location.href = "/investigations";
    } catch (cause) {
      alert(`Case deletion failed: ${cause instanceof Error ? cause.message : String(cause)}`);
      setBusy(false);
    }
  };

  const canReview = can("review_investigation");
  const canArchive = can("archive_investigation");
  const canDelete = can("delete_investigation");
  const isOwnerOrAdmin = can("edit_investigation");

  return (
    <>
      {/* Top Action Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Lead Investigator: <strong>{inv.owner?.display_name ?? inv.owner?.username ?? inv.owner_id}</strong>
            {inv.description ? ` · ${inv.description}` : ""}
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          {/* Status Progression Controls */}
          {/* DRAFT -> VALIDATING -> ANALYZING -> ACTIVE follow real events
              (a validated upload, a completed analysis run) and are set by
              the backend. DRAFT -> ACTIVE is deliberately not a shortcut. */}
          {inv.status === "DRAFT" && isOwnerOrAdmin && (
            <Link to={`/investigations/new?case=${invId}`} className="btn btn-sm btn-primary">Upload a dataset</Link>
          )}

          {(inv.status === "VALIDATING" || inv.status === "ANALYZING") && (
            <span className="small muted">
              {inv.status === "VALIDATING" ? "Dataset validated; run the analysis from the dataset panel below." : "Analysis in progress."}
            </span>
          )}

          {inv.status === "ACTIVE" && isOwnerOrAdmin && (
            <>
              <button
                className="btn btn-sm btn-primary"
                onClick={() => changeStatus("SUBMITTED")}
                disabled={busy}
              >
                Submit for Review 
              </button>
              <button
                className="btn btn-sm"
                onClick={() => changeStatus("CLOSED")}
                disabled={busy}
              >
                Close Case
              </button>
            </>
          )}

          {inv.status === "RETURNED" && isOwnerOrAdmin && (
            <button
              className="btn btn-sm btn-primary"
              onClick={() => changeStatus("ACTIVE")}
              disabled={busy}
            >
              Resume Casework 
            </button>
          )}

          {inv.status === "SUBMITTED" && canReview && (
            <button
              className="btn btn-sm btn-primary"
              onClick={() => changeStatus("IN_REVIEW")}
              disabled={busy}
            >
              Begin Review 
            </button>
          )}

          {inv.status === "IN_REVIEW" && canReview && (
            <Link to={`/inv/${invId}/review`} className="btn btn-sm btn-primary">
              Review Work Product 
            </Link>
          )}

          {inv.status === "APPROVED" && (
            <button
              className="btn btn-sm btn-primary"
              onClick={() => changeStatus("CLOSED")}
              disabled={busy}
            >
              Close Approved Case 
            </button>
          )}

          {inv.status === "CLOSED" && (
            <>
              {isOwnerOrAdmin && (
                <button
                  className="btn btn-sm"
                  onClick={() => changeStatus("ACTIVE")}
                  disabled={busy}
                >
                  Reopen Case
                </button>
              )}
              {canArchive && (
                <button
                  className="btn btn-sm"
                  onClick={handleArchive}
                  disabled={busy}
                >
                  Archive Case
                </button>
              )}
              {canDelete && (
                <button
                  className="btn btn-sm"
                  style={{ color: "var(--critical)" }}
                  onClick={handleDelete}
                  disabled={busy}
                >
                  Delete Case
                </button>
              )}
            </>
          )}

          {inv.status === "ARCHIVED" && (
            <>
              {canArchive && (
                <button
                  className="btn btn-sm btn-primary"
                  onClick={handleRestore}
                  disabled={busy}
                >
                  Restore Case
                </button>
              )}
              {canDelete && (
                <button
                  className="btn btn-sm"
                  style={{ color: "var(--critical)" }}
                  onClick={handleDelete}
                  disabled={busy}
                >
                  Delete Case
                </button>
              )}
            </>
          )}

          <Link to={`/inv/${invId}/report`} className="btn btn-sm">
            Generate Report 
          </Link>
        </div>
      </div>

      {run?.status === "STALE" && (
        <StaleRunBanner
          bound={run.bound_run_fingerprint}
          current={run.current_artifact_run}
          count={stale}
        />
      )}

      {/* 1. TOP: Case Health & Status Canvas */}
      <div className="panel" style={{ 
        marginBottom: 20, 
        background: "linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-raised) 100%)",
        borderColor: "var(--border-strong)"
      }}>
        <div className="panel-head">
          <h2 style={{ fontSize: "0.95rem", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Case context
          </h2>
          <span className="small muted mono">Case Status: {inv.status}</span>
        </div>
        <div className="panel-body">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
            {/* 1. Case Health */}
            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Case status</span>
                <span className={`status-badge status-${inv.status.toLowerCase()}`}>{inv.status}</span>
              </div>
              <strong style={{ display: "block", fontSize: 13 }}>
                {inv.status === "SUBMITTED" || inv.status === "IN_REVIEW" ? "Under independent review" : inv.status === "APPROVED" ? "Findings approved" : inv.status === "CLOSED" ? "Closed" : "Casework"}
              </strong>
              <span className="small faint mono">
                Audit Trail Append-Only
              </span>
            </div>

            {/* 2. Analysis Status */}
            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Analysis Status</span>
                <span className="mono" style={{ color: "var(--blockchain)", fontSize: 13 }}>
                  {dataset?.analysis_run?.status ?? "NOT RUN"}
                </span>
              </div>
              <strong style={{ display: "block", fontSize: 13 }}>{dataset?.analysis_run ? "Uploaded-dataset pipeline" : "No run yet"}</strong>
              <span className="small faint">{dataset?.analysis_run ? "Scored by the registry champion" : "Upload and run a dataset to score it"}</span>
            </div>

            {/* 3. Dataset */}
            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Dataset</span>
                <span className="mono" style={{ color: dataset ? "var(--model)" : "var(--high)", fontSize: 13 }}>
                  {dataset ? dataset.status : "NONE"}
                </span>
              </div>
              <strong style={{ display: "block", fontSize: 13, textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }} title={dataset?.filename}>
                {dataset ? dataset.filename : "No dataset attached"}
              </strong>
              <span className="small faint mono">
                {dataset ? `SHA: ${dataset.sha256.slice(0, 12)}…` : "Unattached"}
              </span>
            </div>

            {/* 4. Run Fingerprint */}
            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Reference alert run</span>
                <RunStatusChip status={run?.status ?? "CURRENT"} />
              </div>
              <strong style={{ display: "block", fontSize: 13 }} className="mono">
                {run?.bound_run_fingerprint ? `${run.bound_run_fingerprint.slice(0, 14)}…` : "Not bound"}
              </strong>
              <span className="small faint">{run?.bound_run_fingerprint ? "Fixed for every alert in this case" : "Binds when the first alert is referenced"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. PRIORITY ALERTS */}
      <section className="panel" style={{ marginBottom: 20 }}>
        <div className="panel-head">
          <h2>Alerts in this case</h2>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <span className="small muted">Ranked by risk prioritization</span>
            <Link to={`/inv/${invId}/alerts`} className="btn btn-sm">
              All Alerts ({alerts.length}) 
            </Link>
          </div>
        </div>

        {loading ? (
          <div className="panel-body"><p className="muted">Loading case alerts…</p></div>
        ) : alerts.length === 0 ? (
          <div className="panel-body">
            <p className="muted" style={{ margin: 0 }}>
              No reference-run alerts are in this case yet. Add them from the{" "}
              <Link to="/alerts">alert queue</Link>.
              {dataset?.analysis_run?.status === "COMPLETE" && " The uploaded dataset's own ranked entities are listed under the dataset run below."}
            </p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>Alert ID</th>
                  <th>Disposition</th>
                  <th>Assigned To</th>
                  <th>Run Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {alerts.slice(0, 6).map((a) => (
                  <tr key={a.alert_id}>
                    <td className="mono small">
                      <Link to={`/inv/${invId}/alerts/${a.alert_id}`} style={{ fontWeight: 600 }}>
                        {a.alert_id}
                      </Link>
                    </td>
                    <td><DispositionBadge state={a.disposition?.state} /></td>
                    <td className="small muted">
                      {a.assigned_to_display_name ?? a.assigned_to_username ?? "—"}
                    </td>
                    <td>
                      {a.stale === true ? (
                        <span className="runchip runchip-stale">STALE</span>
                      ) : (
                        <span className="runchip runchip-current">CURRENT</span>
                      )}
                    </td>
                    <td>
                      <Link to={`/inv/${invId}/alerts/${a.alert_id}`} className="btn btn-sm">
                        Investigate 
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 4. INVESTIGATION ACTIVITY */}
      <section className="panel" style={{ marginBottom: 20 }}>
        <div className="panel-head">
          <h2>Investigation Activity</h2>
          <span className="small muted">Casework Material & Evidence Workspaces</span>
        </div>
        <div className="panel-body">
          {/* Metrics 4-Grid */}
          <div className="card-grid-4" style={{ marginBottom: 18 }}>
            <div className="stat-card">
              <span className="stat-card-value">{summary?.alerts_referenced ?? alerts.length}</span>
              <span className="stat-card-label">Referenced alerts</span>
            </div>
            <div className="stat-card stat-card--high">
              <span className="stat-card-value">{summary?.outstanding ?? 0}</span>
              <span className="stat-card-label">Awaiting a decision</span>
            </div>
            <div className="stat-card">
              <span className="stat-card-value">
                {summary?.dispositions_by_state.CONFIRMED ?? 0}
              </span>
              <span className="stat-card-label">Confirmed findings</span>
            </div>
            <div className="stat-card">
              <span className="stat-card-value">{summary?.notes ?? 0}</span>
              <span className="stat-card-label">Investigator notes</span>
            </div>
          </div>

          {/* Direct Workspace Links */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
            <Link to={`/inv/${invId}/graph`} className="panel" style={{ margin: 0, padding: 14, textDecoration: "none", color: "inherit", display: "block" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <strong>Investigation Graph</strong>
                <span className="small" style={{ color: "var(--cyan)" }}>Open </span>
              </div>
              <p className="small muted" style={{ margin: 0 }}>
                Interactive graph exploring entity clusters, co-spending links, and transaction flows.
              </p>
            </Link>

            <Link to={`/inv/${invId}/timeline`} className="panel" style={{ margin: 0, padding: 14, textDecoration: "none", color: "inherit", display: "block" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <strong>Activity Timeline</strong>
                <span className="small" style={{ color: "var(--cyan)" }}>Open </span>
              </div>
              <p className="small muted" style={{ margin: 0 }}>
                Chronological sequence of transactions, peer announcements, and pattern emergence.
              </p>
            </Link>

            <Link to={`/inv/${invId}/network`} className="panel" style={{ margin: 0, padding: 14, textDecoration: "none", color: "inherit", display: "block" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <strong>Network Intelligence</strong>
                <span className="small" style={{ color: "var(--cyan)" }}>Open </span>
              </div>
              <p className="small muted" style={{ margin: 0 }}>
                P2P propagation timing, peer observer diversity, and ASN/GeoIP telemetry.
              </p>
            </Link>

            <Link to={`/inv/${invId}/evidence`} className="panel" style={{ margin: 0, padding: 14, textDecoration: "none", color: "inherit", display: "block" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <strong>Evidence Matrix</strong>
                <span className="small" style={{ color: "var(--cyan)" }}>Open </span>
              </div>
              <p className="small muted" style={{ margin: 0 }}>
                Multi-layer evidence fusion across ledger, topology, network, and ML signals.
              </p>
            </Link>
          </div>
        </div>
      </section>

      {/* 5. NEXT ACTION */}
      <section className="panel" style={{ 
        marginBottom: 20,
        borderColor: "var(--cyan)",
        background: "rgba(6, 182, 212, 0.03)"
      }}>
        <div className="panel-head">
          <h2 style={{ color: "var(--cyan)" }}>Next Action</h2>
          <span className="small muted">Investigative Guidance</span>
        </div>
        <div className="panel-body">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16 }}>
            <div>
              <h3 style={{ margin: "0 0 6px", fontSize: "1rem" }}>
                {(summary?.outstanding ?? 0) > 0
                  ? `Review and triage ${summary?.outstanding} pending alert${summary?.outstanding === 1 ? "" : "s"}`
                  : inv.status === "ACTIVE" && (summary?.alerts_referenced ?? 0) === 0
                  ? "No alerts referenced yet"
                  : inv.status === "ACTIVE"
                  ? "Every referenced alert has a disposition - ready to submit for review"
                  : inv.status === "SUBMITTED" || inv.status === "IN_REVIEW"
                  ? "Investigation awaiting reviewer assessment and formal sign-off"
                  : inv.status === "RETURNED"
                  ? "Returned by the reviewer - read the review note, resume casework"
                  : "Investigation completed and findings recorded"}
              </h3>
              <p className="small muted" style={{ margin: 0, maxWidth: 650, lineHeight: 1.5 }}>
                {(summary?.outstanding ?? 0) > 0
                  ? "Open the highest-priority alert to inspect the SHAP feature contributions, peeling-chain patterns, and network observations, then record your disposition."
                  : inv.status === "ACTIVE" && (summary?.alerts_referenced ?? 0) === 0
                  ? "Review the dataset run's ranked entities, or reference alerts from the queue and record a disposition for each, before submitting."
                  : inv.status === "ACTIVE"
                  ? "Submit this investigation to the Review Queue for independent quality assurance and reviewer decision."
                  : inv.status === "SUBMITTED" || inv.status === "IN_REVIEW"
                  ? "A reviewer will evaluate the case findings, inspect evidence layers, and record a decision."
                  : inv.status === "RETURNED"
                  ? "The reviewer's rationale is in the case notes."
                  : "Findings are recorded. The report can be read and exported; the audit trail holds every step."}
              </p>
            </div>

            <div>
              {(summary?.outstanding ?? 0) > 0 ? (
                <Link to={`/inv/${invId}/alerts`} className="btn btn-primary">
                  Triage Priority Alerts 
                </Link>
              ) : inv.status === "ACTIVE" ? (
                <button
                  className="btn btn-primary"
                  onClick={() => changeStatus("SUBMITTED")}
                  disabled={busy}
                >
                  Submit for review
                </button>
              ) : inv.status === "SUBMITTED" || inv.status === "IN_REVIEW" ? (
                <Link to={`/inv/${invId}/review`} className="btn btn-primary">
                  Open Review Workspace 
                </Link>
              ) : (
                <Link to={`/inv/${invId}/report`} className="btn btn-primary">
                  View Final Report 
                </Link>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Datasets and Analytical Run Details */}
      <section className="panel">
        <div className="panel-head">
          <h2>Bound Datasets & Ingestion Records</h2>
          <span className="small muted">{inv.datasets?.length ?? 0} Stored</span>
        </div>
        {!inv.datasets?.length ? (
          <div className="panel-body">
            <p className="muted" style={{ margin: 0 }}>No dataset has been uploaded to this investigation.</p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Format</th>
                  <th className="num">Size</th>
                  <th>Cryptographic SHA-256</th>
                  <th>Validation</th>
                  <th>Analysis Run</th>
                </tr>
              </thead>
              <tbody>
                {inv.datasets.map((d) => (
                  <tr key={d.id}>
                    <td><strong>{d.filename}</strong></td>
                    <td className="small muted">{d.format.toUpperCase()}</td>
                    <td className="num">{d.size_bytes.toLocaleString()} B</td>
                    <td className="mono small">{d.sha256.slice(0, 20)}…</td>
                    <td>
                      <span className={`status-badge status-${d.status === "VALIDATED" ? "active" : "closed"}`}>
                        {d.status}
                      </span>
                    </td>
                    <td className="mono small">
                      {d.analysis_run?.status ?? "NOT RUN"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
