/**
 * Investigation Review Workspace — Independent Governance & Quality Assurance
 *
 * Provides formal review workflow for case work products:
 * - Three explicit actions:
 *   [ APPROVE FINDINGS ]
 *   [ RETURN FOR CLARIFICATION ]
 *   [ REQUEST FURTHER INVESTIGATION ]
 * - Rationale notes recorded into append-only audit log
 * - Status transition with validation
 */
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import * as api from "../../api/console";
import type { CaseAlertRow, InvestigatorNote, RunStatus } from "../../api/types";
import { useCase } from "../../store/investigation";
import { RunStatusChip } from "../../components/layout/CaseChrome";

export function InvestigationReview() {
  const { invId = "" } = useParams();
  const navigate = useNavigate();
  const { investigation: inv, reload } = useCase();

  const [alerts, setAlerts] = useState<CaseAlertRow[]>([]);
  const [notes, setNotes] = useState<InvestigatorNote[]>([]);
  const [rationale, setRationale] = useState("");
  const [busy, setBusy] = useState(false);
  const [submittedMessage, setSubmittedMessage] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.listCaseAlerts(invId, controller.signal)
      .then((r) => setAlerts(r.alerts))
      .catch(() => {});
    api.listNotes(invId)
      .then((r) => setNotes(r.notes))
      .catch(() => {});
    return () => controller.abort();
  }, [invId]);

  const summary = inv.summary;
  const confirmed = summary?.dispositions_by_state.CONFIRMED ?? 0;
  const escalated = summary?.dispositions_by_state.ESCALATED ?? 0;
  const outstanding = summary?.outstanding ?? 0;

  const handleDecision = async (decision: "APPROVE" | "CLARIFICATION" | "INVESTIGATION") => {
    if (!rationale.trim()) return;
    setBusy(true);
    setSubmittedMessage(null);

    let prefix = "";
    let nextStatus = "ACTIVE";

    if (decision === "APPROVE") {
      prefix = "[REVIEW DECISION: APPROVE FINDINGS]";
      nextStatus = "APPROVED";
    } else if (decision === "CLARIFICATION") {
      prefix = "[REVIEW DECISION: RETURN FOR CLARIFICATION]";
      nextStatus = "RETURNED";
    } else {
      prefix = "[REVIEW DECISION: REQUEST FURTHER INVESTIGATION]";
      nextStatus = "RETURNED";
    }

    try {
      // 1. Add audit rationale note
      await api.addNote(invId, `${prefix}\n\n${rationale}`);
      // 2. Transition investigation status. A submitted case is taken into
      //    review first: the lifecycle is SUBMITTED -> IN_REVIEW -> decision.
      if (inv.status === "SUBMITTED") await api.setInvestigationStatus(invId, "IN_REVIEW");
      await api.setInvestigationStatus(invId, nextStatus);
      await reload();

      setSubmittedMessage(`Decision recorded: ${prefix.replace(/[\[\]]/g, "")}`);
      setRationale("");

      setTimeout(() => {
        navigate(`/inv/${invId}`);
      }, 1500);
    } catch (cause) {
      alert(`Review submission failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-header" style={{ marginBottom: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <span className="case-header-label">{inv.case_label}</span>
            <h1 style={{ margin: 0, fontSize: "1.4rem" }}>Review Work Product: {inv.name}</h1>
            <span className={`status-badge status-${inv.status.toLowerCase()}`}>
              {inv.status}
            </span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Lead Investigator: <strong>{inv.owner?.display_name ?? inv.owner?.username ?? "Unassigned"}</strong>
            {" · "}
            Alert run: <RunStatusChip status={(inv.run_status ?? inv.analytical_run?.status ?? "UNBOUND") as RunStatus} />
          </p>
        </div>
      </div>

      {submittedMessage && (
        <div className="banner banner-synthetic" style={{ marginBottom: 20 }}>
          <h4>{submittedMessage}</h4>
          <p>Redirecting to case overview…</p>
        </div>
      )}

      {/* Case Findings & Triage Summary */}
      <div className="panel" style={{ marginBottom: 20 }}>
        <div className="panel-head">
          <h2>Investigative Work Product Summary</h2>
          <span className="small muted mono">Quality Assurance Inspection</span>
        </div>
        <div className="panel-body">
          <div className="card-grid-4">
            <div className="stat-card">
              <span className="stat-card-value">{alerts.length}</span>
              <span className="stat-card-label">Referenced Alerts</span>
            </div>
            <div className="stat-card">
              <span className="stat-card-value" style={{ color: "var(--model)" }}>{confirmed}</span>
              <span className="stat-card-label">Confirmed Findings</span>
            </div>
            <div className="stat-card">
              <span className="stat-card-value" style={{ color: "var(--critical)" }}>{escalated}</span>
              <span className="stat-card-label">Escalated Priority</span>
            </div>
            <div className="stat-card">
              <span className="stat-card-value" style={{ color: outstanding > 0 ? "var(--high)" : "inherit" }}>
                {outstanding}
              </span>
              <span className="stat-card-label">Pending Decisions</span>
            </div>
          </div>

          <h3 className="field-label" style={{ margin: "20px 0 8px" }}>What to check</h3>
          {alerts.length === 0 ? (
            <p className="muted small">No alerts are referenced in this case. Approving it approves no finding.</p>
          ) : (
            <table>
              <thead><tr><th>Alert</th><th>Investigator decision</th><th>Rationale</th><th>Decided by</th><th>Run</th></tr></thead>
              <tbody>{alerts.map((a) => (
                <tr key={a.alert_id}>
                  <td><Link className="mono small row-link" to={`/inv/${invId}/alerts/${encodeURIComponent(a.alert_id)}`}>{a.alert_id}</Link></td>
                  <td>{a.disposition ? <span className="disp">{a.disposition.state}</span> : <span className="small" style={{ color: "var(--oc-sev-high)" }}>none</span>}</td>
                  <td className="small">{a.disposition?.rationale?.trim()
                    ? a.disposition.rationale.length > 120 ? `${a.disposition.rationale.slice(0, 120)}…` : a.disposition.rationale
                    : <span style={{ color: "var(--oc-sev-high)" }}>missing</span>}</td>
                  <td className="small">{a.disposition?.decided_by_display_name ?? a.disposition?.decided_by_username ?? "—"}</td>
                  <td>{a.stale === true ? <span className="runchip runchip-stale">stale</span> : a.stale === null ? <span className="runchip runchip-unverifiable">unverifiable</span> : <span className="runchip runchip-current">current</span>}</td>
                </tr>
              ))}</tbody>
            </table>
          )}
          <p className="note" style={{ marginTop: 8 }}>
            Check that every confirmed or escalated alert has a rationale grounded in on-chain evidence, that no claim treats a
            relay peer as a sender, and that a model score is not cited as proof.
          </p>
          <ReportSignOff invId={invId} />
        </div>
      </div>

      {/* Reviewer Action Canvas (Correction 8) */}
      <section className="panel" style={{ 
        borderColor: "var(--border-strong)", 
        background: "linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-raised) 100%)",
        marginBottom: 20
      }}>
        <div className="panel-head">
          <h2>Formal Review Decision</h2>
          <span className="small muted">Independent Oversight</span>
        </div>
        <div className="panel-body">
          <div className="form-group">
            <label htmlFor="review-rationale">Reviewer Assessment & Rationale (Mandatory)</label>
            <textarea
              id="review-rationale"
              rows={5}
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
              placeholder="State the basis of your review decision. Note any evidentiary gaps, unverified peer links, or justification for closing the case findings…"
            />
          </div>

          <p className="small muted" style={{ marginTop: 4 }}>
            Recorded in the append-only audit trail with your identity and timestamp.
          </p>

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 20 }}>
            <button
              type="button"
              className="btn btn-primary"
              style={{ background: "#059669", borderColor: "#10b981", color: "#ffffff", padding: "10px 18px", fontWeight: 600 }}
              disabled={!rationale.trim() || busy}
              onClick={() => handleDecision("APPROVE")}
            >
              ✓ APPROVE FINDINGS
            </button>

            <button
              type="button"
              className="btn"
              style={{ background: "#d97706", borderColor: "#f59e0b", color: "#ffffff", padding: "10px 18px", fontWeight: 600 }}
              disabled={!rationale.trim() || busy}
              onClick={() => handleDecision("CLARIFICATION")}
            >
              Return for clarification
            </button>

            <button
              type="button"
              className="btn"
              style={{ background: "#374151", borderColor: "#4b5563", color: "#ffffff", padding: "10px 18px", fontWeight: 600 }}
              disabled={!rationale.trim() || busy}
              onClick={() => handleDecision("INVESTIGATION")}
            >
              ⟳ REQUEST FURTHER INVESTIGATION
            </button>
          </div>
        </div>
      </section>

      {/* Prior Casework Notes & Review History */}
      <section className="panel">
        <div className="panel-head">
          <h2>Prior Casework Notes & Decisions</h2>
          <span className="small muted">{notes.length} Notes</span>
        </div>
        <div className="panel-body">
          {notes.length === 0 ? (
            <p className="muted">No prior notes recorded on this case.</p>
          ) : (
            <ul className="note-list">
              {notes.map((n) => (
                <li key={n.id} style={{ marginBottom: 16, paddingBottom: 16, borderBottom: "1px solid var(--hairline)" }}>
                  <div className="note-meta" style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <strong>{n.author_display_name ?? n.author_username ?? "Analyst"}</strong>
                    <span className="faint small mono">{new Date(n.created_at).toLocaleString()}</span>
                  </div>
                  <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{n.body}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </>
  );
}

/** The report's real state, and the sign-off a reviewer (not the author) gives. */
function ReportSignOff({ invId }: { invId: string }) {
  const [rep, setRep] = useState<Awaited<ReturnType<typeof api.getReport>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const load = () => api.getReport(invId).then(setRep).catch((e) => setError((e as Error).message));
  useEffect(() => { void load(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [invId]);
  const report = rep?.report ?? null;
  return (
    <div className="panel-inset" style={{ marginTop: 16 }}>
      <div className="row">
        <strong className="small">Report</strong>
        {report ? <span className="chip">v{report.version} · {report.status}</span> : <span className="small" style={{ color: "var(--oc-sev-high)" }}>no saved version</span>}
        <span className="spacer" />
        <Link to={`/inv/${invId}/report`} className="btn btn-sm">Read the report</Link>
        {rep?.may_finalise && report && report.status === "DRAFT" && (
          <button type="button" className="btn btn-sm btn-primary" disabled={busy}
            onClick={async () => { setBusy(true); setError(null);
              try { await api.finaliseReport(invId, report.version); await load(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }}>
            Sign off v{report.version}
          </button>
        )}
      </div>
      {error && <p className="small" style={{ color: "var(--oc-danger)", marginTop: 6 }} role="alert">{error}</p>}
      {report?.status === "FINAL" && <p className="note" style={{ marginTop: 6 }}>Signed off. A FINAL version cannot be edited; a change is a new version.</p>}
    </div>
  );
}
