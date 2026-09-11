/**
 * The alert queue: what an investigator sees first.
 *
 * Filtering and pagination are pushed to the API, because the backend holds
 * all 2,128 alerts and the client holds one page. Sorting is client-side and
 * applies to the CURRENT PAGE ONLY - the UI says so, because a sort that
 * silently reorders 25 of 2,128 rows while looking like a global sort would
 * mislead about what is at the top of the queue.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { fetchAlerts } from "../api/client";
import type { AlertListResponse, AlertSummary, Severity } from "../api/types";
import { ErrorState } from "./ErrorState";
import { EmptyState, RiskBar, SeverityBadge, Skeleton } from "./primitives";

const SEVERITIES: Severity[] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
const PAGE_SIZES = [25, 50, 100];

type SortKey = "rank" | "risk_score" | "members_scored" | "last_timestep";

export function AlertQueue() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Severity[]>([]);
  const [minRisk, setMinRisk] = useState<string>("");
  const [firstTimestep, setFirstTimestep] = useState<string>("");
  const [lastTimestep, setLastTimestep] = useState<string>("");
  const [limit, setLimit] = useState(25);
  const [offset, setOffset] = useState(0);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({
    key: "rank", desc: false,
  });

  const [data, setData] = useState<AlertListResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchAlerts(
      {
        severity: selected.length ? selected : undefined,
        minRisk: minRisk === "" ? undefined : Number(minRisk),
        firstTimestep: firstTimestep === "" ? undefined : Number(firstTimestep),
        lastTimestep: lastTimestep === "" ? undefined : Number(lastTimestep),
        limit,
        offset,
      },
      controller.signal,
    )
      .then((response) => { setData(response); setLoading(false); })
      .catch((cause) => {
        if ((cause as Error)?.name === "AbortError") return;
        setError(cause); setLoading(false);
      });
    return () => controller.abort();
  }, [selected, minRisk, firstTimestep, lastTimestep, limit, offset, reloads]);

  const toggle = useCallback((severity: Severity) => {
    setOffset(0);
    setSelected((current) =>
      current.includes(severity)
        ? current.filter((s) => s !== severity)
        : [...current, severity],
    );
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const copy = [...data.alerts];
    copy.sort((a, b) => {
      const left = a[sort.key] ?? 0;
      const right = b[sort.key] ?? 0;
      const delta = Number(left) - Number(right);
      return sort.desc ? -delta : delta;
    });
    return copy;
  }, [data, sort]);

  const header = (key: SortKey, label: string, numeric = false) => (
    <th
      className={`sortable${numeric ? " num" : ""}`}
      onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : true }))}
      aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}
    >
      {label}{sort.key === key ? (sort.desc ? " ▾" : " ▴") : ""}
    </th>
  );

  return (
    <>
      <section className="panel">
        <div className="panel-head"><h2>Filters</h2></div>
        <div className="panel-body">
          <div className="controls">
            <div className="field">
              <label id="sev-label">Severity</label>
              <div className="toggles" role="group" aria-labelledby="sev-label">
                {SEVERITIES.map((severity) => (
                  <button
                    key={severity}
                    className="toggle"
                    aria-pressed={selected.includes(severity)}
                    onClick={() => toggle(severity)}
                  >
                    {severity}
                  </button>
                ))}
              </div>
            </div>
            <div className="field">
              <label htmlFor="min-risk">Min risk</label>
              <input id="min-risk" type="number" min={0} max={1} step={0.05}
                value={minRisk} placeholder="0.00"
                onChange={(e) => { setOffset(0); setMinRisk(e.target.value); }} />
            </div>
            <div className="field">
              <label htmlFor="first-t">Active from timestep</label>
              <input id="first-t" type="number" min={1} max={49} value={firstTimestep}
                placeholder="1"
                onChange={(e) => { setOffset(0); setFirstTimestep(e.target.value); }} />
            </div>
            <div className="field">
              <label htmlFor="last-t">Active to timestep</label>
              <input id="last-t" type="number" min={1} max={49} value={lastTimestep}
                placeholder="49"
                onChange={(e) => { setOffset(0); setLastTimestep(e.target.value); }} />
            </div>
            <div className="field">
              <label htmlFor="page-size">Page size</label>
              <select id="page-size" value={limit}
                onChange={(e) => { setOffset(0); setLimit(Number(e.target.value)); }}>
                {PAGE_SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <button onClick={() => {
              setSelected([]); setMinRisk(""); setFirstTimestep("");
              setLastTimestep(""); setOffset(0);
            }}>Clear</button>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Alert queue</h2>
          {data ? (
            <span className="small muted">
              {data.alert_count_matched.toLocaleString()} matched of{" "}
              {data.alert_count_total.toLocaleString()} · ranked by{" "}
              {data.alerts[0]?.ranking_aggregation ?? "—"}
            </span>
          ) : null}
        </div>

        {loading ? <Skeleton rows={8} /> : null}
        {!loading && error ? (
          <ErrorState error={error} onRetry={() => setReloads((n) => n + 1)} />
        ) : null}
        {!loading && !error && rows.length === 0 ? (
          <EmptyState title="No alerts match these filters">
            Every alert in this run was excluded by the current filter
            combination. Clear the filters to see the full ranked queue.
          </EmptyState>
        ) : null}

        {!loading && !error && rows.length > 0 ? (
          <>
            <div className="panel-body flush" style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {header("rank", "Rank", true)}
                    <th>Severity</th>
                    {header("risk_score", "Risk", true)}
                    <th>Cluster</th>
                    {header("members_scored", "Members", true)}
                    <th className="num">Timesteps</th>
                    <th>Top model signals</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((alert) => (
                    <AlertRow key={alert.alert_id} alert={alert}
                      onOpen={() => navigate(`/alerts/${alert.alert_id}`)} />
                  ))}
                </tbody>
              </table>
            </div>
            <div className="legend" style={{ justifyContent: "space-between" }}>
              <span className="faint">
                Column sorting reorders the current page only; ranking across all{" "}
                {data?.alert_count_matched.toLocaleString()} matched alerts is the
                server's.
              </span>
              <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - limit))}>Previous</button>
                <span className="mono small">
                  {offset + 1}–{Math.min(offset + limit, data?.alert_count_matched ?? 0)}
                </span>
                <button
                  disabled={!data || offset + limit >= data.alert_count_matched}
                  onClick={() => setOffset(offset + limit)}>Next</button>
              </span>
            </div>
          </>
        ) : null}
      </section>
    </>
  );
}

function AlertRow({ alert, onOpen }: { alert: AlertSummary; onOpen: () => void }) {
  return (
    <tr className="clickable" onClick={onOpen} tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter") onOpen(); }}
      aria-label={`Alert ${alert.alert_id}, severity ${alert.severity}`}>
      <td className="num">{alert.rank}</td>
      <td><SeverityBadge severity={alert.severity} /></td>
      <td className="num">
        <RiskBar value={alert.risk_score} severity={alert.severity} />{" "}
        {alert.risk_score === null ? "—" : alert.risk_score.toFixed(4)}
      </td>
      <td className="mono">{alert.cluster_id}</td>
      <td className="num">
        {alert.members_scored.toLocaleString()}
        {alert.members_total !== alert.members_scored ? (
          <span className="faint"> / {alert.members_total.toLocaleString()}</span>
        ) : null}
      </td>
      <td className="num">
        {alert.first_timestep === alert.last_timestep
          ? `t${alert.first_timestep}`
          : `t${alert.first_timestep}–${alert.last_timestep}`}
      </td>
      <td className="small muted">
        {alert.top_signals.length
          ? alert.top_signals.join(", ")
          : <span className="faint">none recorded</span>}
      </td>
    </tr>
  );
}
