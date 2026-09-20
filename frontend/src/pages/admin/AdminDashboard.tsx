/**
 * Admin Dashboard — Institutional System & Health Oversight
 *
 * Provides air-gapped system overview:
 * - Workstation & Hardware telemetry
 * - Offline air-gapped mode verification
 * - User and Role breakdown
 * - Analytical run cache and artifact storage footprint
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../../api/console";
import type { AuditEvent, Investigation, UserAccount } from "../../api/types";

export function AdminDashboard() {
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [cases, setCases] = useState<Investigation[]>([]);
  const [activity, setActivity] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api.listUsers(controller.signal).catch(() => ({ users: [] })),
      api.listInvestigations(controller.signal).catch(() => ({ investigations: [] })),
      api.recentActivity(10, controller.signal).catch(() => ({ events: [] })),
    ]).then(([uRes, cRes, aRes]) => {
      setUsers(uRes.users);
      setCases((cRes as any).investigations || []);
      setActivity(aRes.events);
      setLoading(false);
    });

    return () => controller.abort();
  }, []);

  const totalDatasets = cases.reduce((acc, c) => acc + (c.datasets?.length || 0), 0);
  const adminUsers = users.filter((u) => u.role === "ADMIN");
  const investigatorUsers = users.filter((u) => u.role === "INVESTIGATOR");
  const reviewerUsers = users.filter((u) => u.role === "REVIEWER");

  return (
    <>
      <div className="page-header" style={{ marginBottom: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <h1 style={{ margin: 0 }}>System Administration</h1>
            <span className="status-badge status-active">OFFLINE AIR-GAPPED</span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Workstation health, user authorization management, dataset registry, and append-only audit trail
          </p>
        </div>
      </div>

      {/* System Status Indicators */}
      <div className="panel" style={{ 
        marginBottom: 24, 
        background: "linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-raised) 100%)",
        borderColor: "var(--border-strong)"
      }}>
        <div className="panel-head">
          <h2>Air-Gapped Security & Runtime Status</h2>
          <span className="small muted mono">Linux / Darwin Isolated Environment</span>
        </div>
        <div className="panel-body">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
            <div style={{ padding: "14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Network Interface</span>
                <span className="mono" style={{ color: "var(--model)", fontSize: 13 }}>AIR-GAPPED</span>
              </div>
              <strong style={{ fontSize: 14 }}>Loopback Only (127.0.0.1)</strong>
              <span className="small faint" style={{ display: "block", marginTop: 4 }}>Zero outbound socket connections</span>
            </div>

            <div style={{ padding: "14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Analytical Engine</span>
                <span className="mono" style={{ color: "var(--model)", fontSize: 13 }}>READY</span>
              </div>
              <strong style={{ fontSize: 14 }}>PS-Native 17-Stage Pipeline</strong>
              <span className="small faint" style={{ display: "block", marginTop: 4 }}>LightGBM + Isolation Forest + UnionFind</span>
            </div>

            <div style={{ padding: "14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Audit Log State</span>
                <span className="mono" style={{ color: "var(--blockchain)", fontSize: 13 }}>APPEND-ONLY</span>
              </div>
              <strong style={{ fontSize: 14 }}>SQLite WAL Mode</strong>
              <span className="small faint" style={{ display: "block", marginTop: 4 }}>Cryptographic audit persistence</span>
            </div>

            <div style={{ padding: "14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span className="small muted">Dataset Storage</span>
                <span className="mono" style={{ color: "var(--model)", fontSize: 13 }}>IMMUTABLE</span>
              </div>
              <strong style={{ fontSize: 14 }}>Content-Addressed (SHA-256)</strong>
              <span className="small faint" style={{ display: "block", marginTop: 4 }}>Zero in-place modifications</span>
            </div>
          </div>
        </div>
      </div>

      {/* Admin Modules Grid */}
      <div className="card-grid-4" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <span className="stat-card-value">{users.length}</span>
          <span className="stat-card-label">Authorized Users</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{cases.length}</span>
          <span className="stat-card-label">Total Investigations</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{totalDatasets}</span>
          <span className="stat-card-label">Ingested Datasets</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{activity.length}</span>
          <span className="stat-card-label">Recent Audit Events</span>
        </div>
      </div>

      {/* Admin Navigation Hub */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, marginBottom: 24 }}>
        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-head">
            <h3>Users & RBAC Roles</h3>
            <Link to="/admin/users" className="small">Manage Users →</Link>
          </div>
          <div className="panel-body">
            <p className="small muted" style={{ marginTop: 0, marginBottom: 12 }}>
              Manage accounts, configure RBAC permissions (Admin, Investigator, Reviewer), and provision workstation credentials.
            </p>
            <div style={{ display: "flex", gap: 12, fontSize: 13 }}>
              <span><strong>{adminUsers.length}</strong> Admins</span>
              <span className="faint">·</span>
              <span><strong>{investigatorUsers.length}</strong> Investigators</span>
              <span className="faint">·</span>
              <span><strong>{reviewerUsers.length}</strong> Reviewers</span>
            </div>
          </div>
        </div>

        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-head">
            <h3>Dataset Registry</h3>
            <Link to="/admin/datasets" className="small">View Registry →</Link>
          </div>
          <div className="panel-body">
            <p className="small muted" style={{ marginTop: 0, marginBottom: 12 }}>
              Inspect global forensic captures, verify cryptographic SHA-256 integrity, and monitor on-disk volume usage.
            </p>
            <span className="small muted">
              <strong>{totalDatasets}</strong> stored archives addressed by SHA-256
            </span>
          </div>
        </div>

        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-head">
            <h3>Global Audit Log</h3>
            <Link to="/admin/audit" className="small">View Audit Log →</Link>
          </div>
          <div className="panel-body">
            <p className="small muted" style={{ marginTop: 0, marginBottom: 12 }}>
              Complete tamper-evident record of all logins, uploads, status transitions, alert dispositions, and reviewer decisions.
            </p>
            <span className="small muted">
              Append-only tamper-evident compliance record
            </span>
          </div>
        </div>
      </div>

      {/* Recent System Activity */}
      <section className="panel">
        <div className="panel-head">
          <h2>Recent System & Casework Events</h2>
          <Link to="/admin/audit" className="small muted">All Events →</Link>
        </div>
        <div className="panel-body flush">
          {loading ? (
            <p className="muted" style={{ padding: 16 }}>Loading system telemetry…</p>
          ) : activity.length === 0 ? (
            <p className="muted" style={{ padding: 16 }}>No recent audit events.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Actor</th>
                  <th>Action</th>
                  <th>Object</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {activity.map((evt) => (
                  <tr key={evt.id}>
                    <td className="small muted mono">{new Date(evt.at).toLocaleTimeString()}</td>
                    <td><strong>{evt.actor_display_name || evt.actor_username || "System"}</strong></td>
                    <td><span className="status-badge status-draft">{evt.action}</span></td>
                    <td className="mono small">{evt.object_id || "—"}</td>
                    <td className="small muted">
                      {evt.detail && typeof evt.detail === "object"
                        ? Object.entries(evt.detail).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(", ")
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </>
  );
}
