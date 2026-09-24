/**
 * System administration: who can use the system, what state the casework
 * is in, what security-relevant events happened, and what the deployment
 * is serving. Every figure is read from an API; nothing is asserted.
 *
 * Counts from the audit log cover the latest 100 events (the endpoint's
 * limit), and the page says so rather than presenting them as totals.
 */
import { Link } from "react-router-dom";

import * as api from "../../api/console";
import { fetchAlerts } from "../../api/client";
import { listModels } from "../../api/intel";
import type { AuditEvent } from "../../api/types";
import { ErrorState } from "../../components/ui/ErrorState";
import { Metric, PageHeader } from "../../components/ui/intel";
import { Skeleton } from "../../components/ui/primitives";
import { useApi } from "../../lib/useApi";

const LIFECYCLE = ["DRAFT", "VALIDATING", "ANALYZING", "ACTIVE", "SUBMITTED", "IN_REVIEW", "RETURNED", "APPROVED", "CLOSED", "ARCHIVED"];
const SECURITY_ACTIONS = new Set(["LOGIN_FAILED", "USER_CREATED", "USER_DEACTIVATED", "USER_ACTIVATED", "USER_ROLE_CHANGED",
  "USER_DELETED", "PASSWORD_RESET", "PASSWORD_CHANGED", "INVESTIGATION_DELETED", "INVESTIGATION_ARCHIVED"]);

export function AdminDashboard() {
  const users = useApi((s) => api.listUsers(s), []);
  const cases = useApi((s) => api.listInvestigations(s), []);
  const activity = useApi((s) => api.recentActivity(100, s), []);

  const u = users.data?.users ?? [];
  const c = cases.data?.investigations ?? [];
  const events: AuditEvent[] = activity.data?.events ?? [];
  const byStatus = new Map(LIFECYCLE.map((s) => [s, c.filter((x) => x.status === s).length]));
  const maxStatus = Math.max(1, ...byStatus.values());
  const datasets = c.reduce((n, x) => n + (x.summary?.dataset_count ?? x.datasets?.length ?? 0), 0);
  const failedLogins = events.filter((e) => e.action === "LOGIN_FAILED").length;
  const security = events.filter((e) => SECURITY_ACTIONS.has(e.action));
  const loading = users.loading || cases.loading || activity.loading;

  return (
    <>
      <PageHeader
        eyebrow="Administration"
        title="System"
        sub="Accounts, casework state, security events and what the deployment is serving."
        actions={<>
          <Link className="btn btn-sm" to="/admin/users">Users and roles</Link>
          <Link className="btn btn-sm" to="/admin/audit">Audit log</Link>
        </>}
      />

      {[users.error, cases.error, activity.error].filter((e) => e != null).map((e, i) => <ErrorState key={i} error={e} />)}

      <div className="metric-strip">
        <Metric k="Active accounts" v={loading ? "…" : u.filter((x) => x.active).length}
                d={`${u.filter((x) => x.role === "INVESTIGATOR" && x.active).length} investigator · ${u.filter((x) => x.role === "REVIEWER" && x.active).length} reviewer · ${u.filter((x) => x.role === "ADMIN" && x.active).length} admin`} />
        <Metric k="Open cases" v={loading ? "…" : c.filter((x) => !["CLOSED", "ARCHIVED"].includes(x.status)).length} d={`${c.length} in total`} />
        <Metric k="Awaiting review" v={loading ? "…" : byStatus.get("SUBMITTED") ?? 0} d={`${byStatus.get("IN_REVIEW") ?? 0} in review`} />
        <Metric k="Datasets uploaded" v={loading ? "…" : datasets} d="across all cases" />
        <Metric k="Failed sign-ins" v={loading ? "…" : failedLogins} tone={failedLogins ? "var(--oc-sev-high)" : undefined} d="in the latest 100 audit events" />
      </div>

      <div className="grid-2">
        <section className="panel">
          <div className="panel-head"><h2>Cases by lifecycle status</h2></div>
          <div className="panel-body">
            {cases.loading ? <Skeleton rows={4} /> : (
              <div className="severity-bars">
                {LIFECYCLE.map((s) => (
                  <div className="severity-bar-row" key={s} style={{ gridTemplateColumns: "96px 1fr 40px" }}>
                    <span className="severity-bar-label">{s.replace("_", " ")}</span>
                    <span className="severity-bar-track"><span className="severity-bar-fill" style={{ display: "block", width: `${((byStatus.get(s) ?? 0) / maxStatus) * 100}%`, background: "var(--oc-ev-model)" }} /></span>
                    <span className="severity-bar-count">{byStatus.get(s) ?? 0}</span>
                  </div>
                ))}
              </div>
            )}
            <p className="note" style={{ marginTop: 10 }}>DRAFT to ACTIVE follow uploads and completed analysis runs; the rest are decisions by people.</p>
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Accounts</h2><span className="spacer" /><Link className="btn btn-sm btn-ghost" to="/admin/users">Manage</Link>
          </div>
          <div className="panel-body flush table-wrap">
            {users.loading ? <Skeleton rows={3} /> : (
              <table>
                <thead><tr><th>User</th><th>Role</th><th>Status</th><th>Created</th></tr></thead>
                <tbody>{u.map((x) => (
                  <tr key={x.id}>
                    <td><span className="strong small">{x.display_name}</span> <span className="mono small faint">{x.username}</span></td>
                    <td><span className={`status-badge status-${x.role.toLowerCase()}`}>{x.role}</span></td>
                    <td><span className={`status-badge ${x.active ? "status-active" : "status-closed"}`}>{x.active ? "active" : "deactivated"}</span></td>
                    <td className="small muted">{new Date(x.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}</tbody>
              </table>
            )}
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2>Security events</h2>
          <span className="small faint">failed sign-ins, account and role changes, deletions and archives, from the latest 100 audit events</span>
        </div>
        <div className="panel-body flush table-wrap">
          {activity.loading ? <Skeleton rows={3} /> : security.length === 0 ? (
            <p className="muted small" style={{ padding: 16 }}>None in the latest 100 events.</p>
          ) : (
            <table>
              <thead><tr><th>When</th><th>Action</th><th>By</th><th>Object</th><th>Detail</th></tr></thead>
              <tbody>{security.slice(0, 20).map((e) => (
                <tr key={e.id}>
                  <td className="small muted nowrap">{new Date(e.at).toLocaleString()}</td>
                  <td><span className={`audit ${e.action === "LOGIN_FAILED" ? "audit-warn" : "audit-write"}`}>{e.action.replace(/_/g, " ").toLowerCase()}</span></td>
                  <td className="small">{e.actor_display_name ?? e.actor_username ?? "unauthenticated"}</td>
                  <td className="mono small">{e.object_id ?? "—"}</td>
                  <td className="small muted">{Object.entries(e.detail ?? {}).slice(0, 3).map(([k, v]) => `${k}: ${String(v)}`).join(", ") || "—"}</td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
      </section>

      <DeploymentFacts />

      <section className="panel">
        <div className="panel-head">
          <h2>Recent activity</h2><span className="spacer" /><Link className="btn btn-sm btn-ghost" to="/admin/audit">Full audit log</Link>
        </div>
        <div className="panel-body flush table-wrap">
          {activity.loading ? <Skeleton rows={3} /> : (
            <table>
              <thead><tr><th>When</th><th>Who</th><th>Action</th><th>Case</th></tr></thead>
              <tbody>{events.slice(0, 12).map((e) => (
                <tr key={e.id}>
                  <td className="small muted nowrap">{new Date(e.at).toLocaleString()}</td>
                  <td className="small">{e.actor_display_name ?? e.actor_username ?? "system"}</td>
                  <td><span className="audit">{e.action.replace(/_/g, " ").toLowerCase()}</span></td>
                  <td className="mono small">{e.investigation_id ? <Link to={`/inv/${e.investigation_id}`}>{e.investigation_id}</Link> : "—"}</td>
                </tr>
              ))}</tbody>
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
      <div className="panel-head"><h2>Deployment</h2><span className="small faint">read from the model registry and the alert artifact</span></div>
      <div className="panel-body">
        <dl className="kv">
          <dt>Champion model</dt><dd className="mono">{registry.data?.roles.champion ?? (registry.error ? "registry unavailable" : "…")}</dd>
          <dt>Fallback model</dt><dd className="mono">{registry.data?.roles.fallback ?? (registry.data ? "none" : "…")}</dd>
          <dt>Candidate (shadow)</dt><dd className="mono">{registry.data?.roles.candidate ?? (registry.data ? "none" : "…")}</dd>
          <dt>Reference alert run</dt><dd className="mono">{run.data ? `${run.data.run_fingerprint.slice(0, 16)} · ${run.data.alert_count_total.toLocaleString()} alerts` : run.error ? "not available" : "…"}</dd>
          <dt>Alert artifact provenance</dt><dd>{run.data?.provenance.provenance_type ?? "…"}{run.data?.provenance.synthetic_network ? " · network layer synthetic" : ""}</dd>
        </dl>
        <p className="note" style={{ marginTop: 8 }}>
          Network isolation is a property of how the container is run; this page cannot observe it, so it does not claim it.
          Model health against delayed labels is checked with <code>obsidianchain model health</code>.
        </p>
      </div>
    </section>
  );
}
