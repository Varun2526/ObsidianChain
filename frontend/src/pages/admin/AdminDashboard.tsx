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
import { fetchAlerts } from "../../api/client";
import { listModels } from "../../api/intel";
import { useApi } from "../../lib/useApi";
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
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Workstation health, user authorization management, dataset registry, and append-only audit trail
          </p>
        </div>
      </div>

      {/* Deployment facts: each one read from the API, none asserted */}
      <DeploymentFacts />

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

function DeploymentFacts() {
  const registry = useApi((sig) => listModels(sig), []);
  const run = useApi((sig) => fetchAlerts({ limit: 1 }, sig), []);
  return (
    <section className="panel">
      <div className="panel-head"><h2>Deployment</h2><span className="small faint">read from the registry and the alert artifact</span></div>
      <div className="panel-body">
        <dl className="kv">
          <dt>Champion model</dt><dd className="mono">{registry.data?.roles.champion ?? (registry.error ? "registry unavailable" : "…")}</dd>
          <dt>Fallback model</dt><dd className="mono">{registry.data?.roles.fallback ?? (registry.data ? "none" : "…")}</dd>
          <dt>Candidate (shadow)</dt><dd className="mono">{registry.data?.roles.candidate ?? (registry.data ? "none" : "…")}</dd>
          <dt>Reference alert run</dt><dd className="mono">{run.data ? `${run.data.run_fingerprint.slice(0, 16)} · ${run.data.alert_count_total.toLocaleString()} alerts` : run.error ? "not available" : "…"}</dd>
          <dt>Alert artifact provenance</dt><dd>{run.data?.provenance.provenance_type ?? "…"}{run.data?.provenance.synthetic_network ? " · network layer synthetic" : ""}</dd>
        </dl>
        <p className="note" style={{ marginTop: 8 }}>
          Network isolation is a property of how the container is run (<code>make serve</code> drops every interface but the published port); this page cannot observe it, so it does not claim it.
        </p>
      </div>
    </section>
  );
}
