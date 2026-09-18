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

import { fetchAlerts } from "../api/client";
import * as api from "../api/console";
import type { AlertListResponse, Investigation } from "../api/types";
import { useAuth } from "../store/auth";
import { SeverityBadge } from "./primitives";
import { RunStatusChip } from "./CaseChrome";

export function HomePage() {
  const { identity, can } = useAuth();
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [reference, setReference] = useState<AlertListResponse | null>(null);
  const [scope, setScope] = useState<"all" | "owned">("owned");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    api.listInvestigations(controller.signal)
      .then((r) => { setInvestigations(r.investigations); setScope(r.scope); })
      .catch(() => {})
      .finally(() => setLoading(false));
    fetchAlerts({ limit: 5 }, controller.signal)
      .then(setReference)
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
      <div className="page-header">
        <div>
          <h1>Welcome back, {identity?.user.display_name}</h1>
          <p className="muted">
            {scope === "all"
              ? "All investigations on this workstation"
              : "Investigations you own"}
          </p>
        </div>
        {can("create_investigation") && (
          <Link to="/investigations/new" className="btn btn-primary">
            + New Investigation
          </Link>
        )}
      </div>

      {/* Case-owned counts only. */}
      <div className="card-grid-4">
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
        {loading ? (
          <div className="panel-body"><p className="muted">Loading…</p></div>
        ) : open.length === 0 ? (
          <div className="panel-body">
            <p className="muted">
              No open investigations.{" "}
              {can("create_investigation")
                ? <Link to="/investigations/new">Create one</Link>
                : "A reviewer sees cases once investigators create them."}
            </p>
          </div>
        ) : (
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
        )}
      </section>

      {reference && reference.alerts.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2>Reference analytical run</h2>
            <span className="small muted">
              run {reference.run_fingerprint} ·{" "}
              {reference.alert_count_total.toLocaleString()} alerts
            </span>
          </div>
          <div className="panel-body">
            <div className="banner banner-synthetic" style={{ marginTop: 0 }}>
              <h4>These alerts belong to the pipeline, not to a case</h4>
              <p>
                They were produced by an offline analysis run over the frozen
                reference dataset. They are not the result of any file uploaded
                through this console. Open an alert from a case to reference it
                into that investigation.
              </p>
            </div>
          </div>
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th className="num">#</th><th>Severity</th>
                  <th className="num">Risk</th><th>Cluster</th>
                  <th className="num">Members</th><th>Top signals</th>
                </tr>
              </thead>
              <tbody>
                {reference.alerts.map((a) => (
                  <tr key={a.alert_id}>
                    <td className="num">{a.rank}</td>
                    <td><SeverityBadge severity={a.severity} /></td>
                    <td className="num mono">{a.risk_score?.toFixed(4) ?? "—"}</td>
                    <td className="mono">{a.cluster_id}</td>
                    <td className="num">{a.members_scored}</td>
                    <td className="small muted">{a.top_signals.slice(0, 2).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
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
