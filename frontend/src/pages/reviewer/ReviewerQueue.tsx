/**
 * Review queue: the reviewer's work, from the case list and the audit log.
 *
 * Every number is a count over cases this reviewer may read, or an audit
 * event. Nothing is estimated. A case submitted with alerts that still have
 * no disposition is flagged, because approving it would sign off on
 * decisions nobody made.
 */
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import * as api from "../../api/console";
import type { AuditEvent, Investigation } from "../../api/types";
import { RunStatusChip } from "../../components/layout/CaseChrome";
import { ErrorState } from "../../components/ui/ErrorState";
import { Metric, PageHeader } from "../../components/ui/intel";
import { Skeleton } from "../../components/ui/primitives";
import { useApi } from "../../lib/useApi";
import { useAuth } from "../../store/auth";

const REVIEW_EVENTS = new Set(["INVESTIGATION_STATUS_CHANGED", "REPORT_FINALISED"]);

function ago(iso?: string | null): string {
  if (!iso) return "n/a";
  const mins = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const h = Math.floor(mins / 60);
  return h < 24 ? `${h}h ago` : `${Math.floor(h / 24)}d ago`;
}

export function ReviewerQueue() {
  const { identity } = useAuth();
  const navigate = useNavigate();
  const cases = useApi((s) => api.listInvestigations(s), []);
  const activity = useApi((s) => api.recentActivity(100, s), []);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<unknown>(null);

  const all = cases.data?.investigations ?? [];
  const submitted = all.filter((c) => c.status === "SUBMITTED");
  const inReview = all.filter((c) => c.status === "IN_REVIEW");
  const returned = all.filter((c) => c.status === "RETURNED");
  const approved = all.filter((c) => c.status === "APPROVED");
  const closed = all.filter((c) => c.status === "CLOSED" || c.status === "ARCHIVED");

  const decisions: AuditEvent[] = (activity.data?.events ?? []).filter((e) =>
    REVIEW_EVENTS.has(e.action) &&
    (e.action === "REPORT_FINALISED" || ["IN_REVIEW", "APPROVED", "RETURNED"].includes(String(e.detail?.to))));
  const mine = decisions.filter((e) => e.actor_id === identity?.user.id);
  const label = new Map(all.map((c) => [c.id, `${c.case_label} · ${c.name}`]));

  const take = async (c: Investigation) => {
    setBusy(c.id); setActionError(null);
    try {
      await api.setInvestigationStatus(c.id, "IN_REVIEW");
      navigate(`/inv/${c.id}/review`);
    } catch (cause) {
      setActionError(cause); setBusy(null);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Review"
        title="Review queue"
        sub="Independent review of submitted casework before sign-off. A reviewer decides on the case; dispositions stay the investigator's."
      />

      <div className="metric-strip">
        <Metric k="Awaiting review" v={cases.loading ? "…" : submitted.length} tone={submitted.length ? "var(--oc-sev-high)" : undefined} d="submitted, not yet taken" />
        <Metric k="In review" v={cases.loading ? "…" : inReview.length} d="taken by a reviewer" />
        <Metric k="Returned" v={cases.loading ? "…" : returned.length} d="with the investigator" />
        <Metric k="Approved" v={cases.loading ? "…" : approved.length} d="awaiting closure" />
        <Metric k="Your decisions" v={activity.loading ? "…" : mine.length} d="in the latest 100 audit events" />
      </div>

      {cases.error != null && <ErrorState error={cases.error} onRetry={cases.reload} />}
      {actionError != null && <ErrorState error={actionError} />}

      <QueueTable
        title="Awaiting review"
        empty="Nothing is waiting. Submitted cases appear here."
        rows={submitted}
        loading={cases.loading}
        action={(c) => (
          <button type="button" className="btn btn-sm btn-primary" disabled={busy === c.id} onClick={() => take(c)}>
            {busy === c.id ? "Taking…" : "Take into review"}
          </button>
        )}
      />
      <QueueTable
        title="In review"
        empty="No case is in review."
        rows={inReview}
        loading={cases.loading}
        action={(c) => <Link className="btn btn-sm btn-primary" to={`/inv/${c.id}/review`}>Continue review</Link>}
      />
      {(returned.length > 0 || approved.length > 0) && (
        <QueueTable
          title="Returned or approved"
          empty=""
          rows={[...returned, ...approved]}
          loading={false}
          showStatus
          action={(c) => <Link className="btn btn-sm" to={`/inv/${c.id}`}>Open</Link>}
        />
      )}

      <section className="panel">
        <div className="panel-head">
          <h2>Recent review decisions</h2>
          <span className="small faint">audit log: status changes into review, approvals, returns, report sign-offs</span>
        </div>
        <div className="panel-body flush table-wrap">
          {activity.loading ? <Skeleton rows={3} /> : decisions.length === 0 ? (
            <p className="muted small" style={{ padding: 16 }}>No review decisions recorded yet.</p>
          ) : (
            <table>
              <thead><tr><th>When</th><th>Reviewer</th><th>Decision</th><th>Case</th></tr></thead>
              <tbody>{decisions.slice(0, 20).map((e) => (
                <tr key={e.id}>
                  <td className="small muted nowrap">{new Date(e.at).toLocaleString()}</td>
                  <td className="small">{e.actor_display_name ?? e.actor_username ?? "system"}</td>
                  <td>
                    {e.action === "REPORT_FINALISED"
                      ? <span className="audit audit-report">report signed off</span>
                      : <span className="audit audit-decision">{String(e.detail?.from)} to {String(e.detail?.to)}</span>}
                  </td>
                  <td className="small">{e.investigation_id ? <Link to={`/inv/${e.investigation_id}`}>{label.get(e.investigation_id) ?? e.investigation_id}</Link> : "—"}</td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
      </section>

      <p className="note">{closed.length === 1 ? "1 closed or archived case is" : `${closed.length} closed or archived cases are`} not listed here; see Investigations.</p>
    </>
  );
}

function QueueTable({ title, empty, rows, loading, action, showStatus = false }: {
  title: string; empty: string; rows: Investigation[]; loading: boolean;
  action: (c: Investigation) => React.ReactNode; showStatus?: boolean;
}) {
  return (
    <section className="panel">
      <div className="panel-head"><h2>{title}</h2><span className="small faint">{rows.length}</span></div>
      <div className="panel-body flush table-wrap">
        {loading ? <Skeleton rows={2} /> : rows.length === 0 ? (
          <p className="muted small" style={{ padding: 16 }}>{empty}</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Case</th><th>Title</th>{showStatus && <th>Status</th>}<th>Investigator</th>
                <th className="num">Alerts</th><th className="num">Confirmed</th><th className="num">Escalated</th>
                <th className="num">Dismissed</th><th className="num">Undecided</th><th className="num">Notes</th>
                <th>Alert run</th><th>Last change</th><th />
              </tr>
            </thead>
            <tbody>{rows.map((c) => {
              const d = c.summary?.dispositions_by_state;
              const undecided = c.summary?.outstanding ?? 0;
              return (
                <tr key={c.id}>
                  <td className="mono small">{c.case_label}</td>
                  <td><Link className="row-link" to={`/inv/${c.id}`}>{c.name}</Link></td>
                  {showStatus && <td><span className={`status-badge status-${c.status.toLowerCase()}`}>{c.status}</span></td>}
                  <td className="small">{c.owner?.display_name ?? c.owner?.username ?? c.owner_id}</td>
                  <td className="num">{c.summary?.alerts_referenced ?? 0}</td>
                  <td className="num">{d?.CONFIRMED ?? 0}</td>
                  <td className="num">{d?.ESCALATED ?? 0}</td>
                  <td className="num">{d?.DISMISSED ?? 0}</td>
                  <td className="num" style={undecided ? { color: "var(--oc-sev-high)" } : undefined}
                      title={undecided ? "Submitted with alerts that have no investigator decision" : undefined}>
                    {undecided}{undecided ? " !" : ""}
                  </td>
                  <td className="num">{c.summary?.notes ?? 0}</td>
                  <td><RunStatusChip status={c.run_status ?? "UNBOUND"} /></td>
                  <td className="small muted nowrap">{ago(c.summary?.last_activity_at ?? c.updated_at)}</td>
                  <td>{action(c)}</td>
                </tr>
              );
            })}</tbody>
          </table>
        )}
      </div>
    </section>
  );
}
