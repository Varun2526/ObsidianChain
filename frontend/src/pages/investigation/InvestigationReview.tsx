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
import type { CaseAlertRow, InvestigatorNote } from "../../api/types";
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
      // 2. Transition investigation status
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
              ● {inv.status}
            </span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Lead Investigator: <strong>{inv.owner?.display_name ?? inv.owner?.username ?? "Unassigned"}</strong>
            {" · "}
            Analytical Run: <RunStatusChip status={inv.run_status ?? "CURRENT"} />
          </p>
        </div>
      </div>

      {submittedMessage && (
        <div className="banner banner-synthetic" style={{ marginBottom: 20 }}>
          <h4>✓ {submittedMessage}</h4>
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

          <div style={{ marginTop: 20 }}>
            <h3 className="small muted" style={{ marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Case Report & Artifact State
            </h3>
            <p className="small muted" style={{ marginTop: 0 }}>
              The lead investigator has assembled evidence across ledger clusters, P2P network observations, and heuristic structural patterns.
              Before approval, verify that all high-risk alerts have documented rationales and that claims do not conflate network announcement vantage points with legal entity ownership.
            </p>
            <Link to={`/inv/${invId}/report`} className="btn btn-sm">
              Inspect Full Case Report →
            </Link>
          </div>
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
              ⚠ RETURN FOR CLARIFICATION
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
