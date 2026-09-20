/**
 * One alert, opened — in two panels that are never merged.
 *
 * ANALYTICAL ASSESSMENT is read from the immutable artifact: calibrated
 * risk, severity band, SHAP, M0-M3 evidence, network context, provenance.
 * Nothing in it can be written, and nothing in it is anyone's opinion.
 *
 * INVESTIGATOR ASSESSMENT is read from the application database: the
 * disposition and its full history, assignment, and notes. Every field is
 * attributable to a named person.
 *
 * They are visually and structurally separate because they are different
 * kinds of claim. A model that scored a cluster CRITICAL and an investigator
 * who marked it DISMISSED are both saying something true, about different
 * things, and a layout that blends them turns a ranking signal into a
 * finding.
 *
 * Outside a case (`/alerts/:alertId`) only the analytical panel exists -
 * there is no investigation to hold a decision.
 */
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { fetchAlert, fetchAlerts } from "../../api/client";
import * as api from "../../api/console";
import type {
  AlertDetail as AlertDetailData,
  CaseAlertDetail,
  DispositionState,
  RelatedAlertsResponse,
} from "../../api/types";
import { useAuth } from "../../store/auth";
import { DispositionBadge, DispositionHistory } from "../../components/layout/CaseChrome";
import { ErrorState } from "../../components/ui/ErrorState";
import { EvidencePanel } from "../../components/forensics/EvidencePanel";
import { InvestigationGraph } from "../../components/forensics/InvestigationGraph";
import { CorrelationPanel } from "../../components/forensics/CorrelationPanel";
import { NetworkContextPanel } from "../../components/forensics/NetworkContextPanel";
import { ProvenancePanel } from "../../components/forensics/ProvenancePanel";
import { SeparationEvidencePanel } from "../../components/forensics/SeparationEvidencePanel";
import { StructuralPatternsPanel } from "../../components/forensics/StructuralPatternsPanel";
import { AnalysisLayerToggle, useAnalysisLayer } from "../../components/forensics/AnalysisLayerToggle";
import { Timeline } from "../../components/forensics/Timeline";
import { WhyFlagged } from "../../components/forensics/WhyFlagged";
import { Address, RiskBar, SeverityBadge, Skeleton, Value } from "../../components/ui/primitives";
import { TransactionModal, ClusterCompareModal } from "../../components/modals/InvestigationModals";

export function AlertDetailPage() {
  const { invId, alertId = "" } = useParams();
  return invId
    ? <CaseAlertDetailPage invId={invId} alertId={alertId} />
    : <GlobalAlertDetailPage alertId={alertId} />;
}

/* ---------------------------------------------------------------- global */

function GlobalAlertDetailPage({ alertId }: { alertId: string }) {
  const navigate = useNavigate();
  const [data, setData] = useState<AlertDetailData | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [reloads, setReloads] = useState(0);
  const [alertIds, setAlertIds] = useState<string[]>([]);
  const alertIdx = alertIds.indexOf(alertId);

  useEffect(() => {
    fetchAlerts({ limit: 100 })
      .then((r) => setAlertIds(r.alerts.map((a) => a.alert_id)))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(null); setData(null);
    fetchAlert(alertId, controller.signal)
      .then((response) => { setData(response); setLoading(false); })
      .catch((cause) => {
        if ((cause as Error)?.name === "AbortError") return;
        setError(cause); setLoading(false);
      });
    return () => controller.abort();
  }, [alertId, reloads]);

  if (loading) return <section className="panel"><Skeleton rows={10} /></section>;
  if (error) {
    return (
      <>
        <p><Link to="/alerts">← Back to the alert queue</Link></p>
        <section className="panel">
          <ErrorState error={error} onRetry={() => setReloads((n) => n + 1)} />
        </section>
      </>
    );
  }
  if (!data) return null;

  const prevId = alertIdx > 0 ? alertIds[alertIdx - 1] : undefined;
  const nextId =
    alertIdx >= 0 && alertIdx < alertIds.length - 1 ? alertIds[alertIdx + 1] : undefined;

  return (
    <>
      <div className="alert-detail-header-bar">
        <p style={{ margin: 0 }}><Link to="/alerts">← Back to the alert queue</Link></p>
        <div style={{ display: "flex", gap: 8 }}>
          {prevId && (
            <button className="btn btn-sm" onClick={() => navigate(`/alerts/${prevId}`)}>
              ← Previous
            </button>
          )}
          {nextId && (
            <button className="btn btn-sm" onClick={() => navigate(`/alerts/${nextId}`)}>
              Next →
            </button>
          )}
        </div>
      </div>
      <AnalyticalAssessment data={data} />
    </>
  );
}

/* ------------------------------------------------------------------ case */

function CaseAlertDetailPage({ invId, alertId }: { invId: string; alertId: string }) {
  const [detail, setDetail] = useState<CaseAlertDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(null);
    api.getCaseAlert(invId, alertId, controller.signal)
      .then((r) => { setDetail(r); setLoading(false); })
      .catch((cause) => {
        if ((cause as Error)?.name === "AbortError") return;
        setError(cause); setLoading(false);
      });
    return () => controller.abort();
  }, [invId, alertId, reloads]);

  const base = `/inv/${invId}/alerts`;

  if (loading) return <section className="panel"><Skeleton rows={10} /></section>;

  // A 404 here means the alert is not referenced by this case yet - which is
  // an ordinary state with an obvious next step, not a failure.
  if (api.isNotFound(error)) {
    return <NotReferenced invId={invId} alertId={alertId}
      onReferenced={() => setReloads((n) => n + 1)} />;
  }
  if (error || !detail) {
    return (
      <>
        <p><Link to={base}>← Back to the alert queue</Link></p>
        <section className="panel">
          <ErrorState error={error} onRetry={() => setReloads((n) => n + 1)} />
        </section>
      </>
    );
  }

  return (
    <>
      <div className="alert-detail-header-bar">
        <p style={{ margin: 0 }}><Link to={base}>← Back to the alert queue</Link></p>
        <span className="mono small faint">{detail.alert_id}</span>
      </div>

      <h2 className="assessment-heading assessment-heading--analytical">
        Analytical assessment
        <span className="assessment-sub">
          Model output, read from the immutable artifact. Not editable.
        </span>
      </h2>
      {detail.analytical.available ? (
        <AnalyticalAssessment data={detail.analytical.alert} />
      ) : (
        <section className="panel">
          <div className="panel-head">
            <h2>Analytical results unavailable</h2>
            <span className="runchip runchip-stale">
              {detail.analytical.reason.replace(/_/g, " ")}
            </span>
          </div>
          <div className="panel-body">
            <div className="banner banner-error" style={{ marginTop: 0 }}>
              <h4>This alert cannot be resolved against the current run</h4>
              <p>{detail.analytical.detail}</p>
            </div>
            <dl className="kv">
              <dt>Referenced under run</dt>
              <dd className="mono">{detail.analytical.referenced_run}</dd>
              <dt>Artifact on disk</dt>
              <dd className="mono">
                {detail.analytical.current_artifact_run ?? "unavailable"}
              </dd>
            </dl>
            <p className="note">
              The reference and every decision recorded against it are kept
              below. Nothing has been removed.
            </p>
          </div>
        </section>
      )}

      <h2 className="assessment-heading assessment-heading--investigator">
        Investigator assessment
        <span className="assessment-sub">
          Recorded by named people in this investigation. Separate from the
          model's output above.
        </span>
      </h2>
      <InvestigatorAssessment
        invId={invId}
        alertId={alertId}
        detail={detail}
        onChanged={() => setReloads((n) => n + 1)}
      />
    </>
  );
}

function NotReferenced({
  invId, alertId, onReferenced,
}: { invId: string; alertId: string; onReferenced: () => void }) {
  const { can } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const add = async () => {
    setBusy(true); setError(null);
    try {
      await api.referenceAlert(invId, alertId);
      onReferenced();
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <p><Link to={`/inv/${invId}/alerts`}>← Back to the alert queue</Link></p>
      <section className="panel">
        <div className="panel-head"><h2>Not referenced by this investigation</h2></div>
        <div className="panel-body">
          <p className="muted" style={{ marginTop: 0 }}>
            Alert <span className="mono">{alertId}</span> belongs to the
            analytical run. It is not part of this case until it is referenced,
            which records which run it came from so a later regeneration cannot
            silently re-point it.
          </p>
          {error ? <ErrorState error={error} /> : null}
          {can("reference_alert") ? (
            <div className="form-actions">
              <button className="btn btn-primary" onClick={add} disabled={busy}>
                {busy ? "Adding…" : "Add to this investigation"}
              </button>
            </div>
          ) : (
            <p className="note">
              Your role may not reference alerts into a case.
            </p>
          )}
        </div>
      </section>
    </>
  );
}

/* ------------------------------------------------ investigator assessment */

function InvestigatorAssessment({
  invId, alertId, detail, onChanged,
}: {
  invId: string;
  alertId: string;
  detail: CaseAlertDetail;
  onChanged: () => void;
}) {
  const { can } = useAuth();
  const investigator = detail.investigator;

  const [state, setState] = useState<DispositionState>(
    investigator.disposition?.state ?? "NEW",
  );
  const [rationale, setRationale] = useState("");
  const [note, setNote] = useState("");
  const [assignees, setAssignees] = useState<
    { id: string; username: string; display_name: string }[]
  >([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (!can("assign_alert")) return;
    api.assignableUsers().then((r) => setAssignees(r.users)).catch(() => {});
  }, [can]);

  const run = useCallback(
    async (fn: () => Promise<unknown>) => {
      setBusy(true); setError(null);
      try {
        await fn();
        onChanged();
      } catch (cause) {
        setError(cause);
      } finally {
        setBusy(false);
      }
    },
    [onChanged],
  );

  return (
    <div className="grid-2">
      <div>
        <section className="panel panel--investigator">
          <div className="panel-head">
            <h2>Decision</h2>
            <DispositionBadge state={investigator.disposition?.state} />
          </div>
          <div className="panel-body">
            {error ? <ErrorState error={error} /> : null}

            {can("set_disposition") ? (
              <>
                <div className="form-group">
                  <label htmlFor="disp-state">Disposition</label>
                  <select id="disp-state" value={state}
                    onChange={(e) => setState(e.target.value as DispositionState)}>
                    {investigator.states.map((s) => (
                      <option key={s} value={s}>{s.replace("_", " ")}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="disp-why">Rationale</label>
                  <textarea id="disp-why" rows={3} value={rationale}
                    onChange={(e) => setRationale(e.target.value)}
                    placeholder="Why this decision…" />
                </div>
                <div className="form-actions">
                  <button className="btn btn-primary" disabled={busy}
                    onClick={() => run(async () => {
                      await api.setDisposition(invId, alertId, state, rationale);
                      setRationale("");
                    })}>
                    {busy ? "Recording…" : "Record decision"}
                  </button>
                </div>
              </>
            ) : (
              <p className="note" style={{ marginTop: 0 }}>
                Your role may read this case's decisions but not change them.
              </p>
            )}

            <h3 className="small muted" style={{ margin: "20px 0 8px" }}>
              Decision history
            </h3>
            <DispositionHistory history={investigator.disposition_history} />
            <p className="note">{investigator.meaning}</p>
          </div>
        </section>
      </div>

      <div>
        <section className="panel panel--investigator">
          <div className="panel-head"><h2>Assignment</h2></div>
          <div className="panel-body">
            <dl className="kv">
              <dt>Assigned to</dt>
              <dd>
                {investigator.assignment.assignee?.display_name ?? (
                  <span className="faint">unassigned</span>
                )}
              </dd>
              {investigator.assignment.assigned_at && (
                <>
                  <dt>Since</dt>
                  <dd>{new Date(investigator.assignment.assigned_at).toLocaleString()}</dd>
                </>
              )}
            </dl>
            {can("assign_alert") && (
              <div className="form-group" style={{ marginTop: 12 }}>
                <label htmlFor="assign-to">Reassign</label>
                <select id="assign-to"
                  value={investigator.assignment.assigned_to ?? ""}
                  disabled={busy}
                  onChange={(e) => run(() =>
                    api.assignAlert(invId, alertId, e.target.value || null))}>
                  <option value="">— unassigned —</option>
                  {assignees.map((u) => (
                    <option key={u.id} value={u.id}>
                      {u.display_name} ({u.username})
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </section>

        <section className="panel panel--investigator">
          <div className="panel-head">
            <h2>Notes on this alert</h2>
            <span className="small muted">{investigator.notes.length}</span>
          </div>
          <div className="panel-body">
            {can("write_note") && (
              <>
                <div className="form-group">
                  <label htmlFor="note-body">Add a note</label>
                  <textarea id="note-body" rows={3} value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="What you checked, and what you found…" />
                </div>
                <div className="form-actions">
                  <button className="btn" disabled={busy || !note.trim()}
                    onClick={() => run(async () => {
                      await api.addNote(invId, note, alertId);
                      setNote("");
                    })}>
                    Save note
                  </button>
                </div>
              </>
            )}
            {investigator.notes.length === 0 ? (
              <p className="muted small">No notes on this alert yet.</p>
            ) : (
              <ul className="note-list">
                {investigator.notes.map((n) => (
                  <li key={n.id}>
                    <div className="note-meta">
                      <strong>{n.author_display_name ?? n.author_username}</strong>
                      <span className="faint small">
                        {new Date(n.created_at).toLocaleString()}
                      </span>
                    </div>
                    <p>{n.body}</p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

/* --------------------------------------------------- analytical assessment */

function AnalyticalAssessment({ data }: { data: AlertDetailData }) {
  const { summary, risk } = data;
  const [layer, setLayer] = useAnalysisLayer();
  const [activeTxid, setActiveTxid] = useState<string | null>(null);
  const [compareTargetAlertId, setCompareTargetAlertId] = useState<string | null>(null);

  // Whether there is any network evidence to show at all. Read from the
  // data already in hand - the toggle must not fetch anything to decide
  // what it can offer.
  const networkAvailable =
    data.network_context.available || data.correlation.available;

  const showChain = layer === "CHAIN" || layer === "FUSED";
  const showNetwork = layer === "NETWORK" || layer === "FUSED";

  return (
    <>
      {/* 1. TOP: ALERT CONTEXT */}
      <section className="panel" style={{
        borderColor: summary.severity === "CRITICAL" ? "var(--critical)" : summary.severity === "HIGH" ? "var(--high)" : "var(--border-strong)",
        background: "linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-raised) 100%)",
        marginBottom: 20
      }}>
        <div className="panel-head" style={{ borderBottom: "1px solid var(--hairline)", padding: "14px 18px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span className="mono small" style={{ color: "var(--muted)", textTransform: "uppercase" }}>ALERT</span>
            <h2 style={{ margin: 0, fontSize: "1.2rem", fontWeight: 700 }}>
              {data.alert_id}
            </h2>
            <SeverityBadge severity={summary.severity} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div>
              <span className="small muted" style={{ display: "block", fontSize: 11, textTransform: "uppercase" }}>RISK SCORE</span>
              <span className="mono" style={{ fontSize: "1.3rem", fontWeight: 700, color: summary.severity === "CRITICAL" ? "var(--critical)" : summary.severity === "HIGH" ? "var(--high)" : "inherit" }}>
                {((risk.score ?? 0) * 100).toFixed(0)} / 100
              </span>
            </div>
            <div>
              <span className="small muted" style={{ display: "block", fontSize: 11, textTransform: "uppercase" }}>INVESTIGATIVE PRIORITY</span>
              <span className="mono" style={{ fontSize: "1.3rem", fontWeight: 700 }}>
                {summary.severity}
              </span>
            </div>
          </div>
        </div>

        <div className="panel-body">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <span className="small muted" style={{ display: "block", fontSize: 11, textTransform: "uppercase", marginBottom: 4 }}>ENTITY</span>
              <strong style={{ display: "block", fontSize: 14 }}>Cluster {summary.cluster_id}</strong>
              <span className="small faint">{summary.members_total} address members</span>
            </div>

            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <span className="small muted" style={{ display: "block", fontSize: 11, textTransform: "uppercase", marginBottom: 4 }}>STATUS</span>
              <strong style={{ display: "block", fontSize: 14 }}>
                {summary.severity === "CRITICAL" || summary.severity === "HIGH" ? "Requires Investigation" : "Standard Monitoring"}
              </strong>
              <span className="small faint">Priority #{summary.rank} in queue</span>
            </div>

            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <span className="small muted" style={{ display: "block", fontSize: 11, textTransform: "uppercase", marginBottom: 4 }}>EVIDENCE LAYERS</span>
              <strong style={{ display: "block", fontSize: 14 }}>
                {networkAvailable ? "4 Layers (Fused)" : "3 Layers (Ledger + ML + Patterns)"}
              </strong>
              <span className="small faint">{networkAvailable ? "Network context mapped" : "No usable network telemetry"}</span>
            </div>

            <div style={{ padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6 }}>
              <span className="small muted" style={{ display: "block", fontSize: 11, textTransform: "uppercase", marginBottom: 4 }}>TEMPORAL WINDOW</span>
              <strong style={{ display: "block", fontSize: 14 }}>
                {summary.first_timestep === summary.last_timestep
                  ? `Timestep t${summary.first_timestep}`
                  : `t${summary.first_timestep} – t${summary.last_timestep}`}
              </strong>
              <span className="small faint">Observed active span</span>
            </div>
          </div>

          <div className="banner banner-synthetic" style={{ marginTop: 16, marginBottom: 0 }}>
            <h4 style={{ margin: "0 0 4px", fontSize: 13, textTransform: "uppercase", letterSpacing: "0.03em" }}>
              Investigative Prioritization Notice
            </h4>
            <p style={{ margin: 0, fontSize: 13 }}>
              <strong>The risk score is an investigative priority indicator, not a determination of criminality or ownership.</strong>{" "}
              Signals prioritize forensic attention by measuring statistical deviation, graph topology, and network timing. Attribution requires legal and exchange verification.
            </p>
          </div>
        </div>
      </section>

      <AnalysisLayerToggle
        layer={layer}
        onChange={setLayer}
        networkAvailable={networkAvailable}
      />
      <section className="panel">
        <div className="panel-head">
          <h2>Alert {summary.cluster_id}</h2>
          <SeverityBadge severity={summary.severity} />
          <span className="spacer" style={{ flex: 1 }} />
          <span className="mono small faint">
            {data.alert_id} · run {data.run_fingerprint}
          </span>
        </div>
        <div className="panel-body">
          <div className="stat-row">
            <div className="stat">
              <span className="v" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <RiskBar value={risk.score} severity={risk.severity} />
                <Value value={risk.score} digits={4} />
              </span>
              <span className="k">Calibrated risk ({risk.ranking_aggregation})</span>
            </div>
            <div className="stat">
              <span className="v"><Value value={risk.top_member_risk} digits={4} /></span>
              <span className="k">Highest member risk</span>
            </div>
            <div className="stat">
              <span className="v">{summary.members_scored.toLocaleString()}</span>
              <span className="k">Members scored</span>
            </div>
            <div className="stat">
              <span className="v">{summary.members_total.toLocaleString()}</span>
              <span className="k">Members in cluster</span>
            </div>
            <div className="stat">
              <span className="v">
                {summary.first_timestep === summary.last_timestep
                  ? `t${summary.first_timestep}`
                  : `t${summary.first_timestep}–${summary.last_timestep}`}
              </span>
              <span className="k">Active timesteps</span>
            </div>
            <div className="stat">
              <span className="v">#{summary.rank}</span>
              <span className="k">Queue rank</span>
            </div>
          </div>

          <h3 className="small muted" style={{ margin: "18px 0 6px" }}>
            All aggregations of member risk
          </h3>
          <div className="stat-row">
            {Object.entries(risk.aggregations).map(([name, value]) => (
              <div className="stat" key={name}>
                <span className="v" style={{ fontSize: 15 }}>
                  <Value value={value} digits={4} />
                </span>
                <span className="k">
                  {name}{name === risk.ranking_aggregation ? " · ranking" : ""}
                </span>
              </div>
            ))}
          </div>

          <p className="note">{data.meaning}</p>
          <p className="note">{data.score_scope}</p>
        </div>
      </section>

      <div className="grid-2">
        <div>
          {/* CHAIN: evidence derived from the ledger and the model over it. */}
          {showChain && <WhyFlagged data={data.why_flagged} networkContext={data.network_context} />}
          {showChain && <EvidencePanel evidence={data.evidence} />}
          {showChain && <StructuralPatternsPanel alertId={data.alert_id} />}
          {showChain && <InvestigationGraph alert={data} />}
          {/* NETWORK: announcement observations. Never an ownership claim. */}
          {showNetwork && (
            <CorrelationPanel
              correlation={data.correlation}
              onSelectTxid={(txid) => setActiveTxid(txid)}
            />
          )}
          {/* Cross-Cluster Alert Correlations */}
          <RelatedAlertsPanel
            alertId={data.alert_id}
            onCompare={(targetId) => setCompareTargetAlertId(targetId)}
          />
        </div>
        <div>
          {showChain && <MembersPanel data={data} />}
          {showChain && (
            <Timeline timeline={data.timeline} observedAt={summary.last_timestep} />
          )}
          {showChain && <ClusterEvolutionPanel data={data} />}
          {showNetwork && <NetworkContextPanel context={data.network_context} />}
          {showNetwork && <SeparationEvidencePanel alertId={data.alert_id} />}
          {/* Provenance belongs to the run, not to a layer. */}
          <ProvenancePanel provenance={data.provenance} />
        </div>
      </div>

      {activeTxid && (
        <TransactionModal txid={activeTxid} onClose={() => setActiveTxid(null)} />
      )}
      {compareTargetAlertId && (
        <ClusterCompareModal
          alertAId={data.alert_id}
          alertBId={compareTargetAlertId}
          onClose={() => setCompareTargetAlertId(null)}
        />
      )}
    </>
  );
}

function RelatedAlertsPanel({
  alertId,
  onCompare,
}: {
  alertId: string;
  onCompare: (targetId: string) => void;
}) {
  const { invId } = useParams();
  const [data, setData] = useState<RelatedAlertsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    setLoading(true);
    api.getRelatedAlerts(alertId)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [alertId]);

  if (loading) {
    return (
      <section className="panel">
        <div className="panel-head"><h2>Cross-Cluster Alert Connections</h2></div>
        <div className="panel-body"><Skeleton rows={3} /></div>
      </section>
    );
  }

  if (error || !data || !Array.isArray(data.related_alerts) || data.related_alerts.length === 0) {
    return (
      <section className="panel">
        <div className="panel-head">
          <h2>Cross-Cluster Alert Connections</h2>
          <span className="small muted">0 connections</span>
        </div>
        <div className="panel-body">
          <p className="muted small" style={{ margin: 0 }}>
            No shared transactions or announcing peer overlaps were identified for this cluster against other alert candidates in the reference run.
          </p>
        </div>
      </section>
    );
  }

  const base = invId ? `/inv/${invId}/alerts` : "/alerts";

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Cross-Cluster Alert Connections</h2>
        <span className="small muted">{data.related_alerts.length} connected</span>
      </div>
      <div className="panel-body">
        <div className="banner banner-synthetic" style={{ marginTop: 0 }}>
          <h4>Analytical Limitation</h4>
          <p>{data.meaning}</p>
        </div>
      </div>
      <div className="panel-body flush" style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Connected Alert</th>
              <th>Cluster</th>
              <th>Severity</th>
              <th className="num">Risk</th>
              <th>Relationship</th>
              <th>Connection Details</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.related_alerts.map((r) => (
              <tr key={r.alert_id}>
                <td className="mono small">{r.alert_id}</td>
                <td className="mono">{r.cluster_id}</td>
                <td><SeverityBadge severity={r.severity} /></td>
                <td className="num"><Value value={r.risk_score} digits={4} /></td>
                <td>
                  <span className="runchip runchip-current" style={{ fontSize: 10 }}>
                    {r.relationship_type.replace(/_/g, " ")}
                  </span>
                </td>
                <td className="small muted">{r.detail}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button
                    className="btn btn-sm"
                    style={{ marginRight: 6 }}
                    onClick={() => onCompare(r.alert_id)}
                  >
                    Compare
                  </button>
                  <Link to={`${base}/${r.alert_id}`} className="btn btn-sm btn-ghost">
                    Open →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ClusterEvolutionPanel({ data }: { data: AlertDetailData }) {
  const points = data.timeline?.points ?? [];
  if (points.length === 0) return null;

  const totalTx = points.reduce((acc, p) => acc + (p.transactions || 0), 0);
  const maxActive = Math.max(...points.map((p) => p.active_addresses || 0));
  const totalSent = points.reduce((acc, p) => acc + (p.btc_sent || 0), 0);
  const totalReceived = points.reduce((acc, p) => acc + (p.btc_received || 0), 0);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Cluster Evolution Across Timesteps</h2>
        <span className="small muted">
          {points.length} observed timestep{points.length > 1 ? "s" : ""}
        </span>
      </div>
      <div className="panel-body">
        <div className="stat-row">
          <div className="stat">
            <span className="v">{points.length}</span>
            <span className="k">Active timesteps</span>
          </div>
          <div className="stat">
            <span className="v">{maxActive}</span>
            <span className="k">Peak active addresses</span>
          </div>
          <div className="stat">
            <span className="v">{totalTx}</span>
            <span className="k">Total cluster transactions</span>
          </div>
          <div className="stat">
            <span className="v mono"><Value value={totalSent} digits={2} /></span>
            <span className="k">Total BTC sent</span>
          </div>
          <div className="stat">
            <span className="v mono"><Value value={totalReceived} digits={2} /></span>
            <span className="k">Total BTC received</span>
          </div>
        </div>

        <p className="note" style={{ marginTop: 12 }}>
          <strong>Evolutionary behavior:</strong> Address activity spans timesteps{" "}
          t{data.summary.first_timestep} through t{data.summary.last_timestep}. Temporal clustering reveals patterns in address reuse and transaction bursts across discrete block windows.
        </p>
      </div>
    </section>
  );
}

function MembersPanel({ data }: { data: AlertDetailData }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Member addresses</h2>
        <span className="small muted">
          {data.members.shown} of {data.members.total_scored.toLocaleString()} shown
        </span>
      </div>
      <div className="panel-body flush" style={{ maxHeight: 340, overflowY: "auto" }}>
        <table>
          <thead>
            <tr><th>Address</th><th>Severity</th><th className="num">Risk</th>
              <th className="num">As-of</th></tr>
          </thead>
          <tbody>
            {data.members.rows.map((m) => (
              <tr key={m.address}>
                <td><Address value={m.address} /></td>
                <td><SeverityBadge severity={m.severity} /></td>
                <td className="num"><Value value={m.risk_score} digits={4} /></td>
                <td className="num mono">t{m.observed_at_timestep}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.members.withheld > 0 ? (
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0 }}>
            {data.members.withheld.toLocaleString()} further scored members are not
            listed. The highest-risk members are shown first.
          </p>
        </div>
      ) : null}
    </section>
  );
}
