/**
 * Investigation sub-pages: Graph, Timeline, Network, Evidence, Notes.
 *
 * All pages draw from the alerts THIS investigation has referenced.
 * Preserves strict separation between analytical truth and casework claims.
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchAlert } from "../../api/client";
import * as api from "../../api/console";
import type { AlertDetail, CaseAlertRow, InvestigatorNote } from "../../api/types";
import { Skeleton } from "../../components/ui/primitives";
import { DispositionBadge } from "../../components/layout/CaseChrome";
import { AlertMoneyFlow } from "../../components/graph/AlertMoneyFlow";
import { Timeline } from "../../components/forensics/Timeline";
import { CorrelationPanel } from "../../components/forensics/CorrelationPanel";
import { NetworkContextPanel } from "../../components/forensics/NetworkContextPanel";
import { EvidencePanel } from "../../components/forensics/EvidencePanel";
import { SeparationEvidencePanel } from "../../components/forensics/SeparationEvidencePanel";
import { StructuralPatternsPanel } from "../../components/forensics/StructuralPatternsPanel";
import { useAuth } from "../../store/auth";

function AlertSelector({
  rows, selected, onSelect,
}: {
  rows: CaseAlertRow[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="alert-selector">
      <label className="filter-label" htmlFor="case-alert-pick">
        Referenced alert
      </label>
      <select id="case-alert-pick" value={selected}
        onChange={(e) => onSelect(e.target.value)}>
        {rows.map((r) => (
          <option key={r.alert_id} value={r.alert_id}>
            {r.alert_id}
            {r.disposition ? ` · ${r.disposition.state}` : ""}
            {r.stale === true ? " · STALE" : ""}
          </option>
        ))}
      </select>
      {rows.find((r) => r.alert_id === selected)?.disposition && (
        <DispositionBadge
          state={rows.find((r) => r.alert_id === selected)?.disposition?.state}
        />
      )}
    </div>
  );
}

function useCaseAlerts() {
  const { invId = "" } = useParams();
  const [rows, setRows] = useState<CaseAlertRow[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<AlertDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    api.listCaseAlerts(invId, controller.signal)
      .then((r) => {
        setRows(r.alerts);
        const first = r.alerts.find((a) => a.stale !== true) ?? r.alerts[0];
        if (first) setSelectedId(first.alert_id);
        else setLoading(false);
      })
      .catch(() => setLoading(false));
    return () => controller.abort();
  }, [invId]);

  useEffect(() => {
    if (!selectedId) return;
    const row = rows.find((r) => r.alert_id === selectedId);
    if (row?.stale === true) { setDetail(null); setLoading(false); return; }
    setLoading(true);
    fetchAlert(selectedId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [selectedId, rows]);

  const selected = rows.find((r) => r.alert_id === selectedId);
  return { invId, rows, selectedId, setSelectedId, detail, loading, selected };
}

function SubPage({
  title,
  children,
}: {
  title: string;
  children: (ctx: ReturnType<typeof useCaseAlerts>) => React.ReactNode;
}) {
  const ctx = useCaseAlerts();
  return (
    <>
      <div className="page-header"><h1>{title}</h1></div>
      {ctx.rows.length === 0 ? (
        <section className="panel">
          <div className="panel-body">
            <p className="muted" style={{ marginTop: 0 }}>
              This investigation has not referenced any alerts, so there is
              nothing to draw here. Add alerts from the{" "}
              <Link to={`/inv/${ctx.invId}/alerts`}>alerts page</Link>.
            </p>
            <p className="note">
              This page shows only the case's own material. It does not fall
              back to the analytical run's top-ranked alerts, which belong to
              the pipeline rather than to this investigation.
            </p>
            <div style={{ marginTop: 16 }}>
              <Link to={`/inv/${ctx.invId}/alerts`} className="btn btn-sm btn-primary">
                Browse &amp; Reference Alerts →
              </Link>
            </div>
          </div>
        </section>
      ) : (
        <>
          <AlertSelector rows={ctx.rows} selected={ctx.selectedId}
            onSelect={ctx.setSelectedId} />
          {ctx.selected?.stale === true ? (
            <section className="panel">
              <div className="panel-body">
                <div className="banner banner-error" style={{ marginTop: 0 }}>
                  <h4>This reference is stale</h4>
                  <p>
                    It was referenced under run{" "}
                    <code>{ctx.selected.run_fingerprint}</code>, which is not
                    the artifact currently on disk. Analytical views cannot be
                    resolved for it. The reference and its decision history are
                    kept.
                  </p>
                </div>
              </div>
            </section>
          ) : ctx.loading ? (
            <Skeleton rows={8} />
          ) : (
            children(ctx)
          )}
        </>
      )}
    </>
  );
}

export function GraphSubPage() {
  return (
    <SubPage title="Money flow">
      {({ detail }) =>
        detail ? (
          <AlertMoneyFlow alertId={detail.alert_id} />
        ) : <p className="muted">Reference an alert in this case to trace its money flow.</p>
      }
    </SubPage>
  );
}

export function TimelineSubPage() {
  return (
    <SubPage title="Activity timeline">
      {({ detail }) =>
        detail ? (
          <Timeline timeline={detail.timeline}
            observedAt={detail.summary.last_timestep} />
        ) : <p className="muted">No timeline data available.</p>
      }
    </SubPage>
  );
}

export function NetworkSubPage() {
  return (
    <SubPage title="Network intelligence">
      {({ detail }) =>
        detail ? (
          <>
            {/* Scientific Caveat Banner */}
            <div className="banner banner-synthetic" style={{ marginBottom: 16 }}>
              <h4>P2P Propagation Forensics Notice</h4>
              <p>
                <strong>Network observations provide association and context. They do not by themselves establish wallet ownership.</strong>{" "}
                P2P network announcement observations reflect propagation vantage points reported by passive network monitors, and do not establish physical transaction origin or wallet custody without ISP subscriber records.
              </p>
            </div>
            <NetworkContextPanel context={detail.network_context} />
            {detail.correlation.transactions.length > 0 && (
              <CorrelationPanel correlation={detail.correlation} />
            )}
          </>
        ) : <p className="muted">No network data available.</p>
      }
    </SubPage>
  );
}

export function EvidenceSubPage() {
  return (
    <SubPage title="Evidence Matrix">
      {({ detail, selectedId }) =>
        detail ? (
          <>
            <div className="banner banner-synthetic" style={{ marginBottom: 16 }}>
              <h4>Multi-Layer Forensic Evidence</h4>
              <p>
                Analytical claims are anchored across four independent evidence layers: Ledger state, Structural graph topology, P2P network telemetry, and Supervised ML feature attribution.
              </p>
            </div>
            <EvidencePanel evidence={detail.evidence} />
            <StructuralPatternsPanel alertId={selectedId} />
            <SeparationEvidencePanel alertId={selectedId} />
          </>
        ) : <p className="muted">No evidence data available.</p>
      }
    </SubPage>
  );
}

export function NotesSubPage() {
  const { invId = "" } = useParams();
  const { can } = useAuth();
  const [notes, setNotes] = useState<InvestigatorNote[]>([]);
  const [body, setBody] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const loadNotes = () => {
    setLoading(true);
    api.listNotes(invId)
      .then((r) => setNotes(r.notes))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadNotes();
  }, [invId]);

  const handleSubmit = async () => {
    if (!body.trim()) return;
    setBusy(true);
    try {
      await api.addNote(invId, body);
      setBody("");
      loadNotes();
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Casework Notes & Assessments</h1>
          <p className="muted">Attributable notes and investigative hypotheses recorded by analysts</p>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 20 }}>
        <div>
          <section className="panel">
            <div className="panel-head">
              <h2>Casework Log</h2>
              <span className="small muted">{notes.length} Recorded</span>
            </div>
            <div className="panel-body">
              {loading ? (
                <Skeleton rows={5} />
              ) : notes.length === 0 ? (
                <p className="muted">No notes recorded for this investigation yet.</p>
              ) : (
                <ul className="note-list">
                  {notes.map((n) => (
                    <li key={n.id} style={{ marginBottom: 16, paddingBottom: 16, borderBottom: "1px solid var(--hairline)" }}>
                      <div className="note-meta" style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <strong>{n.author_display_name || n.author_username || "Investigator"}</strong>
                        <span className="faint small mono">{new Date(n.created_at).toLocaleString()}</span>
                      </div>
                      <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{n.body}</p>
                      {n.alert_id && (
                        <div style={{ marginTop: 6 }}>
                          <span className="small muted">Referenced Alert: </span>
                          <Link to={`/inv/${invId}/alerts/${n.alert_id}`} className="mono small">
                            {n.alert_id}
                          </Link>
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        </div>

        <div>
          {can("write_note") && (
            <section className="panel">
              <div className="panel-head">
                <h2>Add Casework Note</h2>
              </div>
              <div className="panel-body">
                <div className="form-group">
                  <label htmlFor="casework-note">Investigative Finding / Observation</label>
                  <textarea
                    id="casework-note"
                    rows={6}
                    value={body}
                    onChange={(e) => setBody(e.target.value)}
                    placeholder="Document analysis hypotheses, external exchange subpoenas, co-spending rationale, or evidence verification steps…"
                  />
                </div>
                <div className="form-actions">
                  <button
                    className="btn btn-primary"
                    onClick={handleSubmit}
                    disabled={!body.trim() || busy}
                  >
                    {busy ? "Saving to Audit Trail…" : "Save Case Note →"}
                  </button>
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    </>
  );
}
