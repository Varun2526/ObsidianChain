/**
 * Admin Audit Log Page — Append-Only Forensic Audit Trail
 *
 * Immutable chain of custody for all system actions:
 * - Filterable by actor, action type, and timeframe
 * - Detailed payload inspection for evidentiary integrity
 * - JSON export for external compliance archives
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../../api/console";
import type { AuditEvent } from "../../api/types";
import { Skeleton } from "../../components/ui/primitives";

export function AdminAuditLogPage() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [filterAction, setFilterAction] = useState("");
  const [filterActor, setFilterActor] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    // The endpoint serves at most 100 events (routes_investigations.recent_activity).
    // Asking for more was a 422 that this page used to swallow and render as
    // "no audit events" - an empty audit trail shown for a full one.
    api.recentActivity(100, controller.signal)
      .then((r) => setEvents(r.events))
      .catch((e: unknown) => { if ((e as Error)?.name !== "AbortError") setLoadError((e as Error)?.message ?? "Could not load the audit log"); })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, []);

  const actions = Array.from(new Set(events.map((e) => e.action)));
  const actors = Array.from(new Set(events.map((e) => e.actor_display_name || e.actor_username || "System")));

  const filtered = events.filter((e) => {
    const matchAction = filterAction ? e.action === filterAction : true;
    const actorName = e.actor_display_name || e.actor_username || "System";
    const matchActor = filterActor ? actorName === filterActor : true;
    return matchAction && matchActor;
  });

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(events, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `obsidianchain_audit_trail_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="page-header" style={{ marginBottom: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <h1 style={{ margin: 0 }}>Audit log</h1>
            <span className="status-badge status-draft">Append-only</span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            The latest 100 events: logins, uploads, analytical runs, dispositions and approvals. Rows are never updated or deleted.
          </p>
        </div>
        <div>
          <button className="btn btn-primary" onClick={exportJSON}>
            Export JSON
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="panel" style={{ marginBottom: 20 }}>
        <div className="panel-body" style={{ padding: "12px 18px", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <label className="small muted" style={{ display: "block", marginBottom: 4 }}>Filter by Action</label>
            <select
              value={filterAction}
              onChange={(e) => setFilterAction(e.target.value)}
              style={{ width: "100%" }}
            >
              <option value="">All Actions ({actions.length})</option>
              {actions.map((act) => (
                <option key={act} value={act}>{act.replace(/_/g, " ")}</option>
              ))}
            </select>
          </div>

          <div style={{ flex: 1, minWidth: 200 }}>
            <label className="small muted" style={{ display: "block", marginBottom: 4 }}>Filter by Actor</label>
            <select
              value={filterActor}
              onChange={(e) => setFilterActor(e.target.value)}
              style={{ width: "100%" }}
            >
              <option value="">All Actors ({actors.length})</option>
              {actors.map((act) => (
                <option key={act} value={act}>{act}</option>
              ))}
            </select>
          </div>

          {(filterAction || filterActor) && (
            <div style={{ paddingTop: 20 }}>
              <button className="btn btn-sm" onClick={() => { setFilterAction(""); setFilterActor(""); }}>
                Clear Filters
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Audit Events Table */}
      <section className="panel">
        <div className="panel-head">
          <h2>Recorded Events</h2>
          <span className="small muted">{filtered.length} Events Displayed</span>
        </div>
        <div className="panel-body flush">
          {loading ? (
            <div style={{ padding: 16 }}><Skeleton rows={6} /></div>
          ) : filtered.length === 0 ? (
            loadError
              ? <div className="banner banner-error" style={{ margin: 16 }}><h4>The audit log could not be loaded</h4><p>{loadError}</p></div>
              : <p className="muted" style={{ padding: 16 }}>No audit events matching filters.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Actor</th>
                  <th>Action Type</th>
                  <th>Associated Case / Object</th>
                  <th>Forensic Details</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((evt) => (
                  <tr key={evt.id}>
                    <td className="mono small muted" style={{ whiteSpace: "nowrap" }}>
                      {new Date(evt.at).toLocaleString()}
                    </td>
                    <td>
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
                            .join(" · ") || "—"
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
