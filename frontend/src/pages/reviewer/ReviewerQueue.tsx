/**
 * Reviewer Queue — Governance & Oversight Workspace
 *
 * Dedicated triage queue for Reviewers & Senior Analysts:
 * - Lists investigations submitted for review (status === "REVIEW")
 * - Summary statistics of review backlog
 * - Direct navigation to formal Review Canvas
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../../api/console";
import type { Investigation } from "../../api/types";
import { RunStatusChip } from "../../components/layout/CaseChrome";

export function ReviewerQueue() {
  const [cases, setCases] = useState<Investigation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    api.listInvestigations(controller.signal)
      .then((r) => setCases(r.investigations))
      .catch(() => {})
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  const inReview = cases.filter(
    (c) => c.status === "SUBMITTED" || c.status === "IN_REVIEW" || c.status === "REVIEW"
  );
  const activeCases = cases.filter(
    (c) => c.status === "ACTIVE" || c.status === "VALIDATING" || c.status === "ANALYZING"
  );
  const closedCases = cases.filter(
    (c) => c.status === "CLOSED" || c.status === "APPROVED" || c.status === "ARCHIVED"
  );

  return (
    <>
      <div className="page-header" style={{ marginBottom: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <h1 style={{ margin: 0 }}>Reviewer Oversight Workspace</h1>
            <span className="status-badge status-review">GOVERNANCE QUEUE</span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Formal quality assurance and independent review of investigative findings before sign-off
          </p>
        </div>
      </div>

      <div className="card-grid-4" style={{ marginBottom: 24 }}>
        <div className="stat-card stat-card--high">
          <span className="stat-card-value">{inReview.length}</span>
          <span className="stat-card-label">Awaiting Review</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{activeCases.length}</span>
          <span className="stat-card-label">Active Field Cases</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{closedCases.length}</span>
          <span className="stat-card-label">Approved & Closed</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{cases.length}</span>
          <span className="stat-card-label">Total Casework</span>
        </div>
      </div>

      {/* CASES AWAITING REVIEW */}
      <section className="panel" style={{ marginBottom: 24, borderColor: inReview.length > 0 ? "var(--high)" : "var(--hairline)" }}>
        <div className="panel-head">
          <h2>Cases Submitted for Review</h2>
          <span className="small muted">{inReview.length} Actionable</span>
        </div>
        {loading ? (
          <div className="panel-body"><p className="muted">Loading reviewer queue…</p></div>
        ) : inReview.length === 0 ? (
          <div className="panel-body">
            <p className="muted">
              No investigations are currently awaiting review. Field investigators submit cases once their alerts are triaged and hypotheses documented.
            </p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>Case ID</th>
                  <th>Title</th>
                  <th>Lead Investigator</th>
                  <th className="num">Alerts</th>
                  <th className="num">Pending Decisions</th>
                  <th>Analytical Run</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {inReview.map((c) => (
                  <tr key={c.id}>
                    <td className="mono font-semibold">
                      <Link to={`/inv/${c.id}/review`}>{c.case_label}</Link>
                    </td>
                    <td>{c.name}</td>
                    <td>{c.owner?.display_name ?? c.owner?.username ?? "Unassigned"}</td>
                    <td className="num">{c.summary?.alerts_referenced ?? 0}</td>
                    <td className="num">{c.summary?.outstanding ?? 0}</td>
                    <td><RunStatusChip status={c.run_status ?? "CURRENT"} /></td>
                    <td>
                      <Link to={`/inv/${c.id}/review`} className="btn btn-sm btn-primary">
                        Review Case Findings →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* OTHER ACTIVE INVESTIGATIONS */}
      <section className="panel">
        <div className="panel-head">
          <h2>Active Investigations (In Progress)</h2>
          <span className="small muted">{activeCases.length} Active</span>
        </div>
        <div className="panel-body flush">
          <table>
            <thead>
              <tr>
                <th>Case ID</th>
                <th>Title</th>
                <th>Status</th>
                <th>Lead Investigator</th>
                <th className="num">Alerts</th>
                <th>Analytical Run</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {activeCases.map((c) => (
                <tr key={c.id}>
                  <td className="mono">{c.case_label}</td>
                  <td>{c.name}</td>
                  <td><span className="status-badge status-active">{c.status}</span></td>
                  <td className="small muted">{c.owner?.display_name ?? c.owner?.username ?? "—"}</td>
                  <td className="num">{c.summary?.alerts_referenced ?? 0}</td>
                  <td><RunStatusChip status={c.run_status ?? "CURRENT"} /></td>
                  <td>
                    <Link to={`/inv/${c.id}`} className="btn btn-sm">
                      Inspect →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
