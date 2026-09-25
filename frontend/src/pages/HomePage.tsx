/**
 * Overview: the investigator's own work first, then the two analytical
 * sources they work from, each labelled with whose it is.
 *
 * The counts at the top are CASE-OWNED: how many investigations this user
 * has, how many alerts they have referenced, how many still need a decision.
 * The reference alert run and the serving model sit below, under headings
 * that say they belong to the pipeline, not to anyone's investigation.
 */
import { useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../api/console";
import { getModel, listModels } from "../api/intel";
import { useAuth } from "../store/auth";
import { RunStatusChip } from "../components/layout/CaseChrome";
import { InvestigationGuideModal } from "../components/modals/InvestigationGuideModal";
import { Icon } from "../components/ui/Icon";
import { Metric, PageHeader, ResultTypeTag, fixed, int } from "../components/ui/intel";
import { Skeleton } from "../components/ui/primitives";
import { useApi } from "../lib/useApi";

export function HomePage() {
  const { identity, can } = useAuth();
  const [guideOpen, setGuideOpen] = useState(false);
  const cases = useApi((s) => api.listInvestigations(s), []);
  const activity = useApi((s) => api.recentActivity(10, s), []);
  const registry = useApi((s) => listModels(s), []);
  const champion = registry.data?.roles.champion ?? null;
  const model = useApi(champion ? (s) => getModel(champion, s) : null, [champion]);

  const investigations = cases.data?.investigations ?? [];
  const open = investigations.filter((i) => i.status !== "CLOSED");
  const referenced = investigations.reduce((n, i) => n + (i.summary?.alerts_referenced ?? 0), 0);
  const outstanding = investigations.reduce((n, i) => n + (i.summary?.outstanding ?? 0), 0);
  const canCreate = can("create_investigation") || identity?.user.role !== "REVIEWER";

  return (
    <>
      <PageHeader
        eyebrow={identity ? `${identity.user.display_name} · ${identity.user.role.toLowerCase()}` : undefined}
        title="Overview"
        sub={cases.data?.scope === "all" ? "Every investigation on this workstation." : "Investigations you own or are assigned."}
        actions={<>
          <button type="button" className="btn btn-sm" onClick={() => setGuideOpen(true)}>Workflow guide</button>
          {canCreate && <Link to="/investigations/new" className="btn btn-sm btn-primary"><Icon name="plus" size={14} />New investigation</Link>}
        </>}
      />

      <div className="metric-strip">
        <Metric k="Open investigations" v={cases.loading ? "…" : int(open.length)} d={`${int(investigations.length)} in total`} />
        <Metric k="Alerts in your cases" v={cases.loading ? "…" : int(referenced)} d="referenced into your investigations" />
        <Metric k="Awaiting a decision" v={cases.loading ? "…" : int(outstanding)} d="no disposition recorded yet"
                tone={outstanding > 0 ? "var(--oc-sev-high)" : undefined} />
        <Metric k="Recent activity" v={activity.loading ? "…" : int(activity.data?.events.length ?? 0)} d="audit events, latest 10" />
      </div>

      <div className="grid-main-side">
        <div className="stack">
          <section className="panel">
            <div className="panel-head">
              <h2>Your investigations</h2>
              <span className="spacer" />
              <Link to="/investigations" className="btn btn-sm btn-ghost">All investigations<Icon name="arrowRight" size={14} /></Link>
            </div>
            <div className="panel-body flush table-wrap">
              {cases.loading ? <Skeleton rows={4} /> : open.length === 0 ? (
                <div className="state">
                  <h3>No open investigations</h3>
                  <p>Create an investigation and upload a capture for the offline pipeline to validate, analyse and rank.</p>
                </div>
              ) : (
                <table>
                  <thead><tr><th>Case</th><th>Name</th><th>Status</th><th className="num">Alerts</th><th className="num">Undecided</th><th>Alert run</th><th>Updated</th></tr></thead>
                  <tbody>{open.slice(0, 8).map((inv) => (
                    <tr key={inv.id}>
                      <td className="mono small">{inv.case_label}</td>
                      <td><Link className="row-link" to={`/inv/${inv.id}`}>{inv.name}</Link></td>
                      <td><span className={`status-badge status-${inv.status.toLowerCase()}`}>{inv.status}</span></td>
                      <td className="num">{inv.summary?.alerts_referenced ?? 0}</td>
                      <td className="num">{inv.summary?.outstanding ?? 0}</td>
                      <td><RunStatusChip status={inv.run_status ?? "UNBOUND"} /></td>
                      <td className="small muted">{timeAgo(inv.updated_at)}</td>
                    </tr>
                  ))}</tbody>
                </table>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Recent casework activity</h2><span className="small faint">append-only audit log</span></div>
            <div className="panel-body flush">
              {activity.loading ? <Skeleton rows={3} /> : (activity.data?.events.length ?? 0) === 0 ? (
                <p className="muted small" style={{ padding: 16 }}>No casework activity recorded yet.</p>
              ) : (
                <table>
                  <thead><tr><th>When</th><th>Who</th><th>Action</th><th>Object</th></tr></thead>
                  <tbody>{activity.data!.events.map((e) => (
                    <tr key={e.id}>
                      <td className="small muted nowrap">{timeAgo(e.at)}</td>
                      <td className="small">{e.actor_display_name || e.actor_username || "system"}</td>
                      <td><span className="audit">{e.action.replace(/_/g, " ")}</span></td>
                      <td className="mono small">
                        {e.investigation_id ? <Link to={`/inv/${e.investigation_id}`}>{e.object_id || e.investigation_id}</Link> : (e.object_id || "—")}
                      </td>
                    </tr>
                  ))}</tbody>
                </table>
              )}
            </div>
          </section>
        </div>

        <div className="stack">
          <section className="panel">
            <div className="panel-head"><h2>Start from</h2></div>
            <div className="panel-body" style={{ display: "grid", gap: 8 }}>
              {canCreate && <Link className="btn" style={{ justifyContent: "flex-start" }} to="/investigations/new"><Icon name="upload" />New investigation from a capture</Link>}
              <Link className="btn" style={{ justifyContent: "flex-start" }} to="/investigations"><Icon name="folder" />Your investigations</Link>
              <Link className="btn btn-ghost" style={{ justifyContent: "flex-start" }} to="/alerts"><Icon name="alert" />Reference run (Elliptic++)</Link>
              <p className="note">Press <kbd>/</kbd> or <kbd>⌘K</kbd> anywhere to search addresses, transactions, alerts and cases.</p>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Serving model</h2>
              <span className="spacer" />
              <Link to="/models" className="btn btn-sm btn-ghost">Details<Icon name="arrowRight" size={14} /></Link>
            </div>
            <div className="panel-body">
              {registry.loading || model.loading ? <Skeleton rows={4} /> : !model.data ? (
                <p className="muted small">No champion is registered. Uploaded datasets cannot be scored until one is.</p>
              ) : (
                <>
                  <dl className="kv">
                    <dt>Champion</dt><dd className="mono">{model.data.version}</dd>
                    <dt>Type</dt><dd>{String(model.data.manifest.model_type ?? "n/a")}</dd>
                    <dt>Features</dt><dd className="mono small">{model.data.feature_schema_version}</dd>
                    <dt>Fallback</dt><dd className="mono small">{registry.data?.roles.fallback ?? "none"}</dd>
                  </dl>
                  <div style={{ display: "grid", gap: 6, marginTop: 12 }}>
                    <div className="row"><ResultTypeTag type="CONFIRMATION" /><span className="num small">nAP {fixed(model.data.evaluation?.summary.confirm?.address.nap?.mean)}</span></div>
                    <div className="row"><ResultTypeTag type="HOLDOUT" /><span className="num small">nAP {fixed(model.data.holdout?.address.nap)} · P@100 {fixed(model.data.holdout?.address["P@100"], 2)}</span></div>
                    <div className="row"><ResultTypeTag type="PRODUCTION" /><span className="small muted">unknown until labels arrive</span></div>
                  </div>
                  <p className="note" style={{ marginTop: 10 }}>Known failure: ranking collapses in some holdout windows; input drift monitoring does not detect it.</p>
                </>
              )}
            </div>
          </section>
        </div>
      </div>

      <InvestigationGuideModal open={guideOpen} onClose={() => setGuideOpen(false)} />
    </>
  );
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}
