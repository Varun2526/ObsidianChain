/**
 * The investigation report — persisted, versioned, run-bound.
 *
 * Three defects this replaces
 * ---------------------------
 * 1. The executive summary was never saved. `save()` persisted only the
 *    notes field, so the summary vanished on navigation.
 * 2. A referenced alert whose id had gone stale was dropped from the table
 *    by a `.catch(() => null)` — a report that listed six alerts silently
 *    listed four, and said nothing.
 * 3. "Priority Findings" was a heading over the GLOBAL top ten, which is the
 *    model's ranking and not anyone's finding.
 *
 * Now every save creates a new version with a content hash, stale
 * references are shown with a warning instead of removed, and the two kinds
 * of claim - analytical results and investigator conclusions - are in
 * separate, labelled sections.
 */
import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import * as api from "../../api/console";
import type { ReportPayload } from "../../api/types";
import { useAuth } from "../../store/auth";
import { DispositionBadge, RunStatusChip, StaleRunBanner } from "../../components/layout/CaseChrome";
import { ErrorState } from "../../components/ui/ErrorState";

export function ReportPage() {
  const { invId = "" } = useParams();
  const { can } = useAuth();

  const [payload, setPayload] = useState<ReportPayload | null>(null);
  const [title, setTitle] = useState("");
  const [summary, setSummary] = useState("");
  const [content, setContent] = useState("");
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const [exportedSha, setExportedSha] = useState<string | null>(null);

  const adopt = useCallback((next: ReportPayload) => {
    setPayload(next);
    setTitle(next.report?.title ?? `${next.case.case_label} - ${next.case.name}`);
    setSummary(next.report?.executive_summary ?? "");
    setContent(next.report?.content ?? "");
    setDirty(false);
  }, []);

  useEffect(() => {
    api.getReport(invId)
      .then(adopt)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [invId, adopt]);

  const save = async () => {
    setBusy(true); setError(null);
    try {
      adopt(await api.saveReport(invId, {
        title, executive_summary: summary, content,
      }));
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  };

  const finalise = async () => {
    if (!payload?.report) return;
    setBusy(true); setError(null);
    try {
      await api.finaliseReport(invId, payload.report.version);
      adopt(await api.getReport(invId));
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  };

  const transitionStatus = async (nextStatus: string) => {
    setBusy(true); setError(null);
    try {
      await api.setInvestigationStatus(invId, nextStatus);
      adopt(await api.getReport(invId));
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  };

  const downloadJsonPackage = async () => {
    setBusy(true); setError(null);
    try {
      const pkg = await api.exportInvestigation(invId);
      const blob = new Blob([JSON.stringify(pkg, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `obsidianchain-${payload?.case.case_label || invId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      const sha = (typeof pkg.export_sha256 === "string" ? pkg.export_sha256 : pkg.bundle_sha256);
      if (typeof sha === "string") {
        setExportedSha(sha);
      }
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  };

  const print = async () => {
    if (payload?.report) {
      // Export is itself an auditable act: the console cannot control what
      // happens to the document afterwards, which is why the departure is
      // recorded.
      try { await api.recordExport(invId, payload.report.version); } catch { /* audit only */ }
    }
    window.print();
  };

  if (loading) {
    return <section className="panel"><div className="panel-body">
      <p className="muted">Loading report…</p></div></section>;
  }
  if (!payload) {
    return <section className="panel"><ErrorState error={error} /></section>;
  }

  const { case: kase, analytical_run: run, report } = payload;

  return (
    <div className="report-page">
      <div className="page-header no-print">
        <div>
          <h1>Investigation report</h1>
          <p className="muted">
            {kase.case_label} · {kase.name}
            {report ? ` · version ${report.version} (${report.status})` : " · unsaved"}
            {kase.status ? ` · Case: ${kase.status}` : ""}
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {kase.status === "ACTIVE" && (
            <button
              className="btn btn-sm"
              onClick={() => transitionStatus("REVIEW")}
              disabled={busy}
              title="Submit this case and draft report for peer review"
            >
              Submit for Review
            </button>
          )}
          {kase.status === "REVIEW" && (
            <button
              className="btn btn-sm btn-ghost"
              onClick={() => transitionStatus("ACTIVE")}
              disabled={busy}
              title="Return case to active triage"
            >
              Return to Active
            </button>
          )}
          <button className="btn btn-sm" onClick={save} disabled={busy}>
            {busy ? "Saving…" : dirty ? "Save new version •" : "Save new version"}
          </button>
          {payload.may_finalise && report && report.status === "DRAFT" && (
            <button className="btn btn-sm" onClick={finalise} disabled={busy}>
              Finalise v{report.version}
            </button>
          )}
          <button
            className="btn btn-sm"
            onClick={downloadJsonPackage}
            disabled={busy}
            title="Download complete audited offline JSON package with cryptographic SHA-256"
          >
            Export Package (JSON)
          </button>
          <button className="btn btn-primary btn-sm" onClick={print}>
            Print / PDF
          </button>
        </div>
      </div>

      {exportedSha && (
        <div className="banner banner-synthetic no-print" style={{ marginBottom: 16 }}>
          <h4>Deterministic Case Package Exported</h4>
          <p>
            Complete offline JSON package downloaded. Cryptographic SHA-256:{" "}
            <span className="mono">{exportedSha}</span>. Action has been recorded in the audit ledger (<code>REPORT_EXPORTED</code>).
          </p>
        </div>
      )}

      {kase.status === "REVIEW" && (
        <div className="banner banner-synthetic no-print" style={{ marginBottom: 16 }}>
          <h4>Case is Under Formal Peer Review</h4>
          <p>
            This investigation is currently pending review. Reviewers should verify evidence traceability chains, analytical bindings, and investigator rationales before finalising the report.
          </p>
        </div>
      )}

      {error ? <div className="no-print"><ErrorState error={error} /></div> : null}

      <div className="print-only report-print-header">
        <h1>OBSIDIANCHAIN Investigation Report</h1>
        <p>{kase.case_label} · {kase.name}</p>
        <p>
          {report
            ? `Version ${report.version} (${report.status}) · generated ${new Date(report.generated_at).toLocaleString()}`
            : "Unsaved draft"}
        </p>
        {report && <p>Content SHA-256: {report.content_sha256}</p>}
      </div>

      {run.status === "STALE" && (
        <StaleRunBanner
          bound={run.bound_run_fingerprint}
          current={run.current_artifact_run}
          count={payload.stale_references.length}
        />
      )}

      <section className="panel">
        <div className="panel-head"><h2>Case information</h2></div>
        <div className="panel-body">
          <dl className="kv">
            <dt>Case</dt><dd>{kase.case_label}</dd>
            <dt>Investigation</dt><dd>{kase.name}</dd>
            <dt>Owner</dt>
            <dd>{kase.owner?.display_name ?? kase.owner?.username ?? kase.owner_id}</dd>
            <dt>Status</dt><dd>{kase.status}</dd>
            <dt>Created</dt><dd>{new Date(kase.created_at).toLocaleString()}</dd>
            {report && (
              <>
                <dt>Report version</dt><dd>{report.version} · {report.status}</dd>
                <dt>Generated at</dt>
                <dd>{new Date(report.generated_at).toLocaleString()}</dd>
                <dt>Content SHA-256</dt>
                <dd className="mono">{report.content_sha256}</dd>
                {report.finalised_at && (
                  <>
                    <dt>Finalised</dt>
                    <dd>{new Date(report.finalised_at).toLocaleString()}</dd>
                  </>
                )}
              </>
            )}
          </dl>
          {kase.description && <p style={{ marginTop: 12 }}>{kase.description}</p>}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Analytical run</h2>
          <RunStatusChip status={run.status} />
        </div>
        <div className="panel-body">
          <dl className="kv">
            <dt>Report bound to run</dt>
            <dd className="mono">
              {report?.run_fingerprint ?? run.bound_run_fingerprint ?? "—"}
            </dd>
            <dt>Artifact on disk</dt>
            <dd className="mono">{run.current_artifact_run ?? "unavailable"}</dd>
          </dl>
          <p className="note">{payload.separation_note}</p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head"><h2>Executive summary</h2></div>
        <div className="panel-body">
          <textarea className="report-textarea no-print" rows={5}
            placeholder="Write the executive summary for this investigation…"
            value={summary}
            onChange={(e) => { setSummary(e.target.value); setDirty(true); }} />
          <div className="print-only">{summary || "—"}</div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head"><h2>Report body</h2></div>
        <div className="panel-body">
          <div className="form-group no-print">
            <label htmlFor="rep-title">Title</label>
            <input id="rep-title" type="text" value={title}
              onChange={(e) => { setTitle(e.target.value); setDirty(true); }} />
          </div>
          <textarea className="report-textarea no-print" rows={8}
            placeholder="Findings, methodology, limitations…"
            value={content}
            onChange={(e) => { setContent(e.target.value); setDirty(true); }} />
          <div className="print-only">{content || "—"}</div>
        </div>
      </section>

      {/* Stale references are SHOWN, never dropped. */}
      {payload.stale_references.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2>Stale alert references</h2>
            <span className="runchip runchip-stale">
              {payload.stale_references.length}
            </span>
          </div>
          <div className="panel-body">
            <div className="banner banner-error" style={{ marginTop: 0 }}>
              <h4>These references cannot be resolved against the current run</h4>
              <p>{payload.stale_references[0]?.warning}</p>
            </div>
            <table>
              <thead>
                <tr><th>Alert</th><th>Referenced under run</th>
                  <th>Investigator decision</th></tr>
              </thead>
              <tbody>
                {payload.stale_references.map((r) => (
                  <tr key={r.alert_id}>
                    <td className="mono small">{r.alert_id}</td>
                    <td className="mono small">{r.run_fingerprint}</td>
                    <td><DispositionBadge state={r.disposition?.state} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="panel">
        <div className="panel-head">
          <h2>Investigator conclusions</h2>
          <span className="small muted">
            {payload.alert_references.length} referenced alerts
          </span>
        </div>
        {payload.alert_references.length === 0 ? (
          <div className="panel-body">
            <p className="muted">
              No alerts have been referenced into this investigation.
            </p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>Alert</th><th>Decision</th><th>Rationale</th>
                  <th>Decided by</th><th>Run</th>
                </tr>
              </thead>
              <tbody>
                {payload.alert_references.map((r) => (
                  <tr key={r.alert_id}>
                    <td className="mono small">{r.alert_id}</td>
                    <td><DispositionBadge state={r.disposition?.state} /></td>
                    <td className="small muted">
                      {r.disposition?.rationale || <span className="faint">—</span>}
                    </td>
                    <td className="small muted">
                      {r.disposition?.decided_by_display_name ?? "—"}
                    </td>
                    <td className="mono small faint">{r.run_fingerprint}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0 }}>
            {payload.disposition_meaning}
          </p>
        </div>
      </section>

      {payload.alert_references.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2>Evidence Traceability & Chain of Custody</h2>
            <span className="small muted">{payload.alert_references.length} finding lineage(s)</span>
          </div>
          <div className="panel-body">
            <p className="note" style={{ marginTop: 0 }}>
              Every investigative finding maintains unbroken provenance from frozen analytical inputs to final disposition. Network observations record gossip vantage points and never assert entity ownership.
            </p>
            {payload.alert_references.map((r) => (
              <div key={r.alert_id} style={{ marginBottom: 16, padding: "12px 14px", border: "1px solid var(--hairline, #e1e4e8)", borderRadius: "var(--radius, 6px)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, alignItems: "center" }}>
                  <span className="mono" style={{ fontWeight: 600 }}>Alert {r.alert_id}</span>
                  <DispositionBadge state={r.disposition?.state} />
                </div>
                <div className="trace-chain" style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
                  <div className="trace-step">
                    <span className="trace-step-number" style={{ fontWeight: "bold", marginRight: 6 }}>1.</span>
                    <span>Frozen Parquet (Run <code className="small">{r.run_fingerprint.slice(0, 10)}…</code>)</span>
                  </div>
                  <span className="trace-arrow">→</span>
                  <div className="trace-step">
                    <span className="trace-step-number" style={{ fontWeight: "bold", marginRight: 6 }}>2.</span>
                    <span>Analytical Feature & Risk Pipeline</span>
                  </div>
                  <span className="trace-arrow">→</span>
                  <div className="trace-step">
                    <span className="trace-step-number" style={{ fontWeight: "bold", marginRight: 6 }}>3.</span>
                    <span>Gossip Network Vantage Points</span>
                  </div>
                  <span className="trace-arrow">→</span>
                  <div className="trace-step">
                    <span className="trace-step-number" style={{ fontWeight: "bold", marginRight: 6 }}>4.</span>
                    <span>Investigator Review ({r.disposition?.decided_by_display_name || "Investigator"})</span>
                  </div>
                  <span className="trace-arrow">→</span>
                  <div className="trace-step">
                    <span className="trace-step-number" style={{ fontWeight: "bold", marginRight: 6 }}>5.</span>
                    <span>Report Binding {report ? `(v${report.version})` : "(Draft)"}</span>
                  </div>
                </div>
                {r.disposition?.rationale && (
                  <p className="small muted" style={{ margin: "8px 0 0" }}>
                    <strong>Finding Rationale:</strong> {r.disposition.rationale}
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="panel">
        <div className="panel-head">
          <h2>Investigator notes</h2>
          <span className="small muted">{payload.notes.length}</span>
        </div>
        <div className="panel-body">
          {payload.notes.length === 0 ? (
            <p className="muted">No notes recorded.</p>
          ) : (
            <ul className="note-list">
              {payload.notes.map((n) => (
                <li key={n.id}>
                  <div className="note-meta">
                    <strong>{n.author_display_name ?? n.author_username}</strong>
                    <span className="faint small">
                      {new Date(n.created_at).toLocaleString()}
                      {n.alert_id ? ` · ${n.alert_id}` : " · case note"}
                    </span>
                  </div>
                  <p>{n.body}</p>
                </li>
              ))}
            </ul>
          )}
          {can("write_note") && <CaseNoteForm invId={invId}
            onSaved={() => api.getReport(invId).then(adopt)} />}
        </div>
      </section>

      {payload.versions.length > 1 && (
        <section className="panel no-print">
          <div className="panel-head"><h2>Version history</h2></div>
          <div className="panel-body flush">
            <table>
              <thead>
                <tr><th className="num">Version</th><th>Status</th>
                  <th>Generated</th><th>By</th><th>Content SHA-256</th></tr>
              </thead>
              <tbody>
                {payload.versions.map((v) => (
                  <tr key={v.id}>
                    <td className="num">{v.version}</td>
                    <td>
                      <span className={`status-badge status-${
                        v.status === "FINAL" ? "closed" : "draft"}`}>{v.status}</span>
                    </td>
                    <td className="small muted">
                      {new Date(v.generated_at).toLocaleString()}
                    </td>
                    <td className="small muted">{v.generated_by_username ?? "—"}</td>
                    <td className="mono small faint">
                      {v.content_sha256.slice(0, 16)}…
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function CaseNoteForm({ invId, onSaved }: { invId: string; onSaved: () => void }) {
  const [body, setBody] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div className="no-print" style={{ marginTop: 16 }}>
      <div className="form-group">
        <label htmlFor="case-note">Add a case note</label>
        <textarea id="case-note" rows={3} value={body}
          onChange={(e) => setBody(e.target.value)} />
      </div>
      <div className="form-actions">
        <button className="btn" disabled={busy || !body.trim()}
          onClick={async () => {
            setBusy(true);
            try { await api.addNote(invId, body); setBody(""); onSaved(); }
            finally { setBusy(false); }
          }}>
          Save note
        </button>
      </div>
    </div>
  );
}
