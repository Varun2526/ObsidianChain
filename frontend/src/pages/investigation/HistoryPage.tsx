/**
 * The case's audit trail.
 *
 * Append-only at the storage layer: `audit_events` carries BEFORE UPDATE and
 * BEFORE DELETE triggers that abort. Nothing in this console can amend it,
 * which is what makes it worth reading - a forensic tool that cannot say who
 * looked at what, when, and what they concluded cannot account for its own
 * output.
 */
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import * as api from "../../api/console";
import type { AuditEvent } from "../../api/types";
import { ErrorState } from "../../components/ui/ErrorState";

const TONE: Record<string, string> = {
  LOGIN: "neutral",
  LOGOUT: "neutral",
  INVESTIGATION_CREATED: "create",
  INVESTIGATION_VIEWED: "read",
  ALERT_VIEWED: "read",
  INVESTIGATION_UPDATED: "write",
  INVESTIGATION_STATUS_CHANGED: "write",
  DATASET_UPLOADED: "write",
  DATASET_VALIDATED: "write",
  DATASET_REJECTED: "warn",
  ANALYTICAL_RUN_BOUND: "provenance",
  ALERT_REFERENCED: "write",
  ALERT_ASSIGNED: "write",
  DISPOSITION_SET: "decision",
  NOTE_CREATED: "write",
  REPORT_CREATED: "report",
  REPORT_UPDATED: "report",
  REPORT_FINALISED: "report",
  REPORT_EXPORTED: "report",
};

export function HistoryPage() {
  const { invId = "" } = useParams();
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    api.getHistory(invId, controller.signal)
      .then((r) => setEvents(r.events))
      .catch((cause) => {
        if ((cause as Error)?.name !== "AbortError") setError(cause);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [invId]);

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Investigation history</h1>
          <p className="muted">
            Append-only. Every entry names who acted and when.
          </p>
        </div>
        <span className="status-badge status-active">APPEND-ONLY</span>
      </div>

      <section className="panel">
        {error ? (
          <ErrorState error={error} />
        ) : loading ? (
          <div className="panel-body"><p className="muted">Loading…</p></div>
        ) : events.length === 0 ? (
          <div className="panel-body"><p className="muted">No events recorded.</p></div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th className="num">#</th><th>When</th><th>Who</th>
                  <th>Action</th><th>Object</th><th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e) => (
                  <tr key={e.id}>
                    <td className="num faint">{e.id}</td>
                    <td className="small muted">{new Date(e.at).toLocaleString()}</td>
                    <td className="small">
                      {e.actor_display_name ?? e.actor_username ?? (
                        <span className="faint">—</span>
                      )}
                    </td>
                    <td>
                      <span className={`audit audit-${TONE[e.action] ?? "neutral"}`}>
                        {e.action.replace(/_/g, " ")}
                      </span>
                    </td>
                    <td className="mono small faint">
                      {e.object_type}
                      {e.object_id ? ` ${e.object_id}` : ""}
                    </td>
                    <td className="small muted">{renderDetail(e.detail)}</td>
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

function renderDetail(detail: Record<string, unknown>): string {
  const entries = Object.entries(detail ?? {}).filter(([, v]) => v != null);
  if (!entries.length) return "";
  return entries.map(([k, v]) => `${k}=${String(v)}`).join("  ");
}
