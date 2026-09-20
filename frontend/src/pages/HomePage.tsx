/**
 * Investigator landing page.
 *
 * The counts at the top are CASE-OWNED: how many investigations this user
 * has, how many alerts they have referenced, how many still need a decision.
 * The global artifact's alert count is not among them.
 *
 * That is the point. The previous version put "2,128 Total Alerts" beside a
 * case list, which is a property of the pipeline's frozen dataset and not of
 * anyone's investigation. The reference queue lower down still shows those
 * alerts, under a heading that says whose they are.
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../api/console";
import type { AuditEvent, Investigation } from "../api/types";
import { useAuth } from "../store/auth";
import { RunStatusChip } from "../components/layout/CaseChrome";
import { InvestigationGuideModal } from "../components/modals/InvestigationGuideModal";

export function HomePage() {
  const { identity, can } = useAuth();
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [activity, setActivity] = useState<AuditEvent[]>([]);
  const [scope, setScope] = useState<"all" | "owned">("owned");
  const [loading, setLoading] = useState(true);
  const [guideOpen, setGuideOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    api.listInvestigations(controller.signal)
      .then((r) => { setInvestigations(r.investigations); setScope(r.scope); })
      .catch(() => {})
      .finally(() => setLoading(false));
    api.recentActivity(10, controller.signal)
      .then((r) => setActivity(r.events))
      .catch(() => {});
    return () => controller.abort();
  }, []);

  const open = investigations.filter((i) => i.status !== "CLOSED");
  const referenced = investigations.reduce(
    (n, i) => n + (i.summary?.alerts_referenced ?? 0), 0);
  const outstanding = investigations.reduce(
    (n, i) => n + (i.summary?.outstanding ?? 0), 0);

  return (
    <>
      {/* Hero Orientation Banner */}
      <div className="page-header" style={{ alignItems: "flex-start", marginBottom: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <h1 style={{ margin: 0 }}>Welcome back, {identity?.user.display_name}</h1>
            <span className="status-badge status-draft" style={{ textTransform: "uppercase", fontSize: 11 }}>
              {identity?.user.role}
            </span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            {scope === "all"
              ? "Institutional investigation console · All workstation cases"
              : "Institutional investigation console · Cases owned by your station"}
            {" · "}
            <span className="faint">Press <kbd className="mono" style={{ padding: "1px 4px", border: "1px solid var(--hairline)", borderRadius: 3 }}>⌘K</kbd> to search everything</span>
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setGuideOpen(true)}
            style={{ display: "flex", alignItems: "center", gap: 6 }}
          >
            <span>◫</span>
            <span>SOP Guide</span>
          </button>
          {can("create_investigation") && (
            <Link to="/investigations/new" className="btn btn-primary">
              + New Investigation
            </Link>
          )}
        </div>
      </div>

      {loading ? (
        <section className="panel"><div className="panel-body"><p className="muted">Loading investigations…</p></div></section>
      ) : investigations.length === 0 ? (
        /* Empty Investigator Home */
        <div className="panel" style={{
          padding: "56px 32px",
          textAlign: "center",
          maxWidth: 640,
          margin: "32px auto",
          background: "var(--bg-panel)",
          border: "1px solid var(--hairline)",
          borderRadius: 8,
          boxShadow: "0 4px 24px rgba(0, 0, 0, 0.4)"
        }}>
          <div style={{
            width: 48,
            height: 48,
            borderRadius: "50%",
            background: "var(--bg-raised)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 16px",
            border: "1px solid var(--border-strong)",
            color: "var(--cyan)",
            fontSize: "1.4rem"
          }}>
            ◫
          </div>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", margin: "0 0 8px" }}>
            NO ACTIVE INVESTIGATIONS
          </h2>
          <p className="muted" style={{ fontSize: "0.95rem", lineHeight: 1.6, maxWidth: 500, margin: "0 auto 24px" }}>
            Create an investigation and upload a Bitcoin transaction/network dataset to begin analysis.
          </p>
          <div style={{ display: "flex", gap: 10, justifyContent: "center", alignItems: "center" }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setGuideOpen(true)}
              style={{ padding: "10px 20px", fontSize: "0.95rem" }}
            >
              ◫ View SOP Flow
            </button>
            {can("create_investigation") && (
              <Link to="/investigations/new" className="btn btn-primary" style={{ padding: "10px 24px", fontSize: "0.95rem", fontWeight: 700 }}>
                + NEW INVESTIGATION
              </Link>
            )}
          </div>
          <div style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid var(--hairline)" }}>
            <span className="faint small">
              Press <kbd className="mono" style={{ padding: "1px 5px", border: "1px solid var(--hairline)", borderRadius: 3 }}>⌘K</kbd> to search everything
            </span>
          </div>
        </div>
      ) : (
        <>
          {/* Hero: Continue Active Investigation if available */}
          {(() => {
            const activeCase = open[0];
            if (!activeCase) return null;
            return (
              <div className="panel" style={{ 
                background: "linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-raised) 100%)",
                borderColor: "var(--border-strong)",
                marginBottom: 20,
                boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)"
              }}>
                <div className="panel-body" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 24, padding: "20px 24px" }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                      <span className="small muted mono" style={{ letterSpacing: "0.05em" }}>ACTIVE CASE WORKSPACE</span>
                      <span className={`status-badge status-${activeCase.status.toLowerCase()}`}>
                        ● {activeCase.status}
                      </span>
                      <span className="mono small faint">{activeCase.case_label}</span>
                    </div>
                    <h2 style={{ margin: "0 0 8px", fontSize: "1.3rem", fontWeight: 600 }}>{activeCase.name}</h2>
                    <p className="muted small" style={{ margin: 0, maxWidth: 650 }}>
                      {activeCase.description || "Comprehensive multi-layer forensic investigation workspace with blockchain graph, network correlation, and risk triage."}
                    </p>
                    <div style={{ display: "flex", gap: 24, marginTop: 14 }}>
                      <div>
                        <span className="small muted">Referenced Alerts: </span>
                        <strong className="mono">{activeCase.summary?.alerts_referenced ?? 0}</strong>
                      </div>
                      <div>
                        <span className="small muted">Pending Decisions: </span>
                        <strong className="mono" style={{ color: (activeCase.summary?.outstanding ?? 0) > 0 ? "var(--high)" : "inherit" }}>
                          {activeCase.summary?.outstanding ?? 0}
                        </strong>
                      </div>
                      <div>
                        <span className="small muted">Analytical Run: </span>
                        <RunStatusChip status={activeCase.run_status ?? "UNBOUND"} />
                      </div>
                    </div>
                  </div>
                  <div>
                    <Link to={`/inv/${activeCase.id}`} className="btn btn-primary" style={{ padding: "10px 20px", whiteSpace: "nowrap" }}>
                      Resume Investigation →
                    </Link>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Case-owned counts only */}
          <div className="card-grid-4" style={{ marginBottom: 20 }}>
            <div className="stat-card">
              <span className="stat-card-value">{open.length}</span>
              <span className="stat-card-label">Open cases</span>
            </div>
            <div className="stat-card">
              <span className="stat-card-value">{referenced}</span>
              <span className="stat-card-label">Alerts in your cases</span>
            </div>
            <div className="stat-card stat-card--high">
              <span className="stat-card-value">{outstanding}</span>
              <span className="stat-card-label">Awaiting a decision</span>
            </div>
            <div className="stat-card">
              <span className="stat-card-value">{investigations.length}</span>
              <span className="stat-card-label">Cases total</span>
            </div>
          </div>

          <section className="panel">
            <div className="panel-head">
              <h2>Your investigations</h2>
              <Link to="/investigations" className="small muted">View all →</Link>
            </div>
            <div className="panel-body flush">
              <table>
                <thead>
                  <tr>
                    <th>Case</th><th>Name</th><th>Status</th>
                    <th className="num">Alerts</th><th className="num">Outstanding</th>
                    <th>Analytical run</th><th>Updated</th><th />
                  </tr>
                </thead>
                <tbody>
                  {open.slice(0, 8).map((inv) => (
                    <tr key={inv.id}>
                      <td className="mono">{inv.case_label}</td>
                      <td>{inv.name}</td>
                      <td>
                        <span className={`status-badge status-${inv.status.toLowerCase()}`}>
                          {inv.status}
                        </span>
                      </td>
                      <td className="num">{inv.summary?.alerts_referenced ?? 0}</td>
                      <td className="num">{inv.summary?.outstanding ?? 0}</td>
                      <td><RunStatusChip status={inv.run_status ?? "UNBOUND"} /></td>
                      <td className="small muted">{timeAgo(inv.updated_at)}</td>
                      <td><Link to={`/inv/${inv.id}`} className="btn btn-sm">Open</Link></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* RECENT ACTIVITY */}
          <section className="panel">
            <div className="panel-head">
              <h2>Recent casework activity</h2>
              <span className="small muted">Append-only audit events</span>
            </div>
            {activity.length === 0 ? (
              <div className="panel-body">
                <p className="muted">No recent casework activity recorded.</p>
              </div>
            ) : (
              <div className="panel-body flush">
                <table>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Investigator</th>
                      <th>Action</th>
                      <th>Object</th>
                      <th>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activity.map((evt) => (
                      <tr key={evt.id}>
                        <td className="small muted">{timeAgo(evt.at)}</td>
                        <td className="small">
                          <strong>{evt.actor_display_name || evt.actor_username || "System"}</strong>
                        </td>
                        <td>
                          <span className="status-badge status-draft">
                            {evt.action.replace(/_/g, " ")}
                          </span>
                        </td>
                        <td className="mono small">
                          {evt.investigation_id ? (
                            <Link to={`/inv/${evt.investigation_id}`}>
                              {evt.object_id || evt.investigation_id}
                            </Link>
                          ) : (
                            evt.object_id || "—"
                          )}
                        </td>
                        <td className="small muted">
                          {evt.detail && typeof evt.detail === "object"
                            ? Object.entries(evt.detail)
                                .map(([k, v]) => `${k}: ${v}`)
                                .slice(0, 2)
                                .join(", ") || "—"
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      <InvestigationGuideModal open={guideOpen} onClose={() => setGuideOpen(false)} />
    </>
  );
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}
