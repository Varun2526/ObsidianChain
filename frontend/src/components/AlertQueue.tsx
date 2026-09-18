/**
 * The alert queue, in two modes that must not be confused.
 *
 * CASE MODE (`/inv/:invId/alerts`) shows the alerts THIS INVESTIGATION has
 * referenced, with the investigator's decision on each. A case with no
 * references shows an empty list - it does not fall back to the artifact's
 * top-ranked alerts, which belong to the pipeline and not to anyone's case.
 * A second tab exposes the reference run for adding alerts, clearly labelled
 * as the pipeline's output.
 *
 * GLOBAL MODE (`/alerts`) browses the analytical artifact directly. Text
 * search is client-side on the loaded page; every other filter is API-backed.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { fetchAlerts } from "../api/client";
import * as api from "../api/console";
import type { AlertListResponse, CaseAlertRow, Severity } from "../api/types";
import { ErrorState } from "./ErrorState";
import { DispositionBadge, RunStatusChip } from "./CaseChrome";
import { EmptyState, RiskBar, SeverityBadge, Skeleton } from "./primitives";

const SEVERITIES: Severity[] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
const PAGE_SIZES = [25, 50, 100];
type SortKey = "rank" | "risk_score" | "members_scored" | "last_timestep";

export function AlertQueue() {
  const { invId } = useParams();
  return invId ? <CaseAlertQueue invId={invId} /> : <GlobalAlertQueue />;
}

/**
 * The case's OWN alerts, plus a labelled door to the reference run.
 *
 * Every row carries the investigator decision alongside the reference, and a
 * stale reference is flagged rather than dropped: alert ids do not survive a
 * regeneration, and removing one would erase a recorded decision.
 */
function CaseAlertQueue({ invId }: { invId: string }) {
  const [tab, setTab] = useState<"case" | "reference">("case");
  const [rows, setRows] = useState<CaseAlertRow[]>([]);
  const [currentRun, setCurrentRun] = useState<string | null>(null);
  const [staleCount, setStaleCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    api.listCaseAlerts(invId, controller.signal)
      .then((r) => {
        setRows(r.alerts);
        setCurrentRun(r.current_artifact_run);
        setStaleCount(r.stale_count);
      })
      .catch((cause) => {
        if ((cause as Error)?.name !== "AbortError") setError(cause);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [invId, reloads]);

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Alerts</h1>
          <p className="muted">
            {rows.length} referenced by this investigation
            {staleCount > 0 ? ` · ${staleCount} stale` : ""}
          </p>
        </div>
      </div>

      <div className="toggles" role="tablist" style={{ marginBottom: 16 }}>
        <button className="toggle" aria-pressed={tab === "case"}
          onClick={() => setTab("case")}>This investigation</button>
        <button className="toggle" aria-pressed={tab === "reference"}
          onClick={() => setTab("reference")}>Reference run</button>
      </div>

      {tab === "reference" ? (
        <>
          <div className="banner banner-synthetic">
            <h4>These alerts belong to the pipeline, not to this case</h4>
            <p>
              They were produced by an offline analysis run over the frozen
              reference dataset, and are not the result of any file uploaded to
              this investigation. Open one and use “Add to this investigation”
              to reference it here.
            </p>
          </div>
          <GlobalAlertQueue />
        </>
      ) : loading ? (
        <section className="panel"><Skeleton rows={6} /></section>
      ) : error ? (
        <section className="panel">
          <ErrorState error={error} onRetry={() => setReloads((n) => n + 1)} />
        </section>
      ) : rows.length === 0 ? (
        <section className="panel">
          <EmptyState title="No alerts referenced yet">
            This investigation has not referenced any alerts. Browse the
            reference run above and add the ones this case is about.
          </EmptyState>
        </section>
      ) : (
        <section className="panel">
          <div className="panel-body flush" style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Alert</th><th>Investigator decision</th><th>Rationale</th>
                  <th>Assigned</th><th>Run</th><th>Added by</th><th />
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.alert_id}>
                    <td className="mono small">{row.alert_id}</td>
                    <td><DispositionBadge state={row.disposition?.state} /></td>
                    <td className="small muted">
                      {row.disposition?.rationale || <span className="faint">—</span>}
                    </td>
                    <td className="small muted">
                      {row.assigned_to_display_name ??
                        row.assigned_to_username ?? <span className="faint">unassigned</span>}
                    </td>
                    <td>
                      <RunStatusChip status={
                        row.stale === true ? "STALE"
                          : row.stale === null ? "UNVERIFIABLE" : "CURRENT"} />
                    </td>
                    <td className="small muted">{row.added_by_username ?? "—"}</td>
                    <td>
                      <Link to={`/inv/${invId}/alerts/${row.alert_id}`}
                        className="btn btn-sm">Open</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel-body">
            <p className="note" style={{ marginTop: 0 }}>
              Decisions in this table are the investigator's. The model's risk
              score and severity band are analytical output and appear on each
              alert; the two are separate assessments and may disagree.
              {currentRun ? ` Artifact on disk: run ${currentRun}.` : ""}
            </p>
          </div>
        </section>
      )}
    </>
  );
}

export function GlobalAlertQueue() {
  const { invId } = useParams();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Severity[]>([]);
  const [minRisk, setMinRisk] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [firstTimestep, setFirstTimestep] = useState("");
  const [lastTimestep, setLastTimestep] = useState("");
  const [limit, setLimit] = useState(25);
  const [offset, setOffset] = useState(0);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "rank", desc: false });

  const [data, setData] = useState<AlertListResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(null);
    fetchAlerts({
      severity: selected.length ? selected : undefined,
      minRisk: minRisk === "" ? undefined : Number(minRisk),
      firstTimestep: firstTimestep === "" ? undefined : Number(firstTimestep),
      lastTimestep: lastTimestep === "" ? undefined : Number(lastTimestep),
      limit, offset,
    }, controller.signal)
      .then((r) => { setData(r); setLoading(false); })
      .catch((cause) => {
        if ((cause as Error)?.name === "AbortError") return;
        setError(cause); setLoading(false);
      });
    return () => controller.abort();
  }, [selected, minRisk, firstTimestep, lastTimestep, limit, offset, reloads]);

  const toggle = useCallback((sev: Severity) => {
    setOffset(0);
    setSelected((c) => c.includes(sev) ? c.filter((s) => s !== sev) : [...c, sev]);
  }, []);

  // Client-side text search on loaded page
  const rows = useMemo(() => {
    if (!data) return [];
    let copy = [...data.alerts];
    if (search.trim()) {
      const q = search.toLowerCase();
      copy = copy.filter((a) =>
        a.alert_id.toLowerCase().includes(q) ||
        String(a.cluster_id).includes(q) ||
        a.top_signals.some((s) => s.toLowerCase().includes(q)),
      );
    }
    copy.sort((a, b) => {
      const left = a[sort.key] ?? 0;
      const right = b[sort.key] ?? 0;
      const delta = Number(left) - Number(right);
      return sort.desc ? -delta : delta;
    });
    return copy;
  }, [data, sort, search]);

  const header = (key: SortKey, label: string, numeric = false) => (
    <th className={`sortable${numeric ? " num" : ""}`}
      onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : true }))}
      aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}>
      {label}{sort.key === key ? (sort.desc ? " ▾" : " ▴") : ""}
    </th>
  );

  const base = invId ? `/inv/${invId}/alerts` : "/alerts";

  return (
    <>
      {!invId && (
        <div className="page-header">
          <div>
            <h1>Alerts</h1>
            <p className="muted">
              The analytical run on disk. These alerts belong to the pipeline,
              not to an investigation.
            </p>
          </div>
          {data && (
            <span className="muted">
              {data.alert_count_matched.toLocaleString()} matched of{" "}
              {data.alert_count_total.toLocaleString()} · ranked by{" "}
              {data.alerts[0]?.ranking_aggregation ?? "—"} · run{" "}
              <span className="mono">{data.run_fingerprint}</span>
            </span>
          )}
        </div>
      )}

      {/* Search */}
      <div className="search-bar">
        <input type="text" placeholder="Search cluster, address, signal…"
          value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>

      {/* Compact filters */}
      <div className="filter-bar">
        <div className="filter-group">
          <span className="filter-label">Severity</span>
          <div className="toggles" role="group">
            {SEVERITIES.map((sev) => (
              <button key={sev} className="toggle" aria-pressed={selected.includes(sev)}
                onClick={() => toggle(sev)}>{sev}</button>
            ))}
          </div>
        </div>
        <div className="filter-group">
          <span className="filter-label">Min risk</span>
          <input type="number" min={0} max={1} step={0.05} value={minRisk}
            placeholder="0.00" style={{ width: 80 }}
            onChange={(e) => { setOffset(0); setMinRisk(e.target.value); }} />
        </div>
        <button className="btn btn-ghost btn-sm"
          onClick={() => setShowAdvanced(!showAdvanced)}>
          {showAdvanced ? "Hide" : "Advanced"}
        </button>
        <button className="btn btn-ghost btn-sm" onClick={() => {
          setSelected([]); setMinRisk(""); setFirstTimestep(""); setLastTimestep(""); setOffset(0); setSearch("");
        }}>Clear</button>
      </div>

      {showAdvanced && (
        <div className="filter-bar">
          <div className="filter-group">
            <span className="filter-label">From timestep</span>
            <input type="number" min={1} max={49} value={firstTimestep}
              placeholder="1" style={{ width: 70 }}
              onChange={(e) => { setOffset(0); setFirstTimestep(e.target.value); }} />
          </div>
          <div className="filter-group">
            <span className="filter-label">To timestep</span>
            <input type="number" min={1} max={49} value={lastTimestep}
              placeholder="49" style={{ width: 70 }}
              onChange={(e) => { setOffset(0); setLastTimestep(e.target.value); }} />
          </div>
          <div className="filter-group">
            <span className="filter-label">Page size</span>
            <select value={limit} onChange={(e) => { setOffset(0); setLimit(Number(e.target.value)); }}>
              {PAGE_SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
        </div>
      )}

      {/* Table */}
      <section className="panel">
        {loading ? <Skeleton rows={8} /> : null}
        {!loading && error ? <ErrorState error={error} onRetry={() => setReloads((n) => n + 1)} /> : null}
        {!loading && !error && rows.length === 0 ? (
          <EmptyState title="No alerts match these filters">
            Adjust the filters or search to find alerts.
          </EmptyState>
        ) : null}

        {!loading && !error && rows.length > 0 && (
          <>
            <div className="panel-body flush" style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {header("rank", "#", true)}
                    <th>Severity</th>
                    {header("risk_score", "Risk", true)}
                    <th>Cluster</th>
                    {header("members_scored", "Members", true)}
                    <th className="num">Timesteps</th>
                    <th>Why flagged</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((alert) => (
                    <tr key={alert.alert_id} className="clickable" tabIndex={0}
                      onClick={() => navigate(`${base}/${alert.alert_id}`)}
                      onKeyDown={(e) => { if (e.key === "Enter") navigate(`${base}/${alert.alert_id}`); }}
                      aria-label={`Alert ${alert.alert_id}`}>
                      <td className="num">{alert.rank}</td>
                      <td><SeverityBadge severity={alert.severity} /></td>
                      <td className="num">
                        <RiskBar value={alert.risk_score} severity={alert.severity} />
                        {" "}{alert.risk_score === null ? "—" : alert.risk_score.toFixed(4)}
                      </td>
                      <td className="mono">{alert.cluster_id}</td>
                      <td className="num">
                        {alert.members_scored.toLocaleString()}
                        {alert.members_total !== alert.members_scored && (
                          <span className="faint"> / {alert.members_total.toLocaleString()}</span>
                        )}
                      </td>
                      <td className="num">
                        {alert.first_timestep === alert.last_timestep
                          ? `t${alert.first_timestep}` : `t${alert.first_timestep}–${alert.last_timestep}`}
                      </td>
                      <td className="small muted">
                        {alert.top_signals.length ? alert.top_signals.join(", ")
                          : <span className="faint">none recorded</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="table-footer">
              <span className="faint small">
                Sorting reorders the current page only.
              </span>
              <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button className="btn btn-sm" disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - limit))}>← Previous</button>
                <span className="mono small">
                  {offset + 1}–{Math.min(offset + limit, data?.alert_count_matched ?? 0)}
                </span>
                <button className="btn btn-sm"
                  disabled={!data || offset + limit >= data.alert_count_matched}
                  onClick={() => setOffset(offset + limit)}>Next →</button>
              </span>
            </div>
          </>
        )}
      </section>
    </>
  );
}
