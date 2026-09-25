/**
 * Network propagation for an uploaded-dataset run, and its graph.
 *
 * Computed by the backend from the capture's own observations
 * (network/propagation.py); nothing here is estimated in the browser. A peer
 * is the IP that announced a transaction to an observer: a relay vantage
 * point, never the sender. These facts are investigation evidence and are
 * not part of the risk score.
 */
import { useState } from "react";

import { getRunGraph, getRunNetwork } from "../../api/intel";
import type { TxPropagation } from "../../api/intel";
import { useApi } from "../../lib/useApi";
import { FlowPreview } from "../graph/FlowPreview";
import { ErrorState } from "../ui/ErrorState";
import { EvidenceTag, Metric, pct } from "../ui/intel";
import { Skeleton } from "../ui/primitives";

function utc(ms: number | null): string {
  if (ms == null) return "n/a";
  return new Date(ms).toISOString().replace("T", " ").replace(".000Z", "Z");
}

/** Shown wherever a DB-IP-resolved country appears (CC BY 4.0 requires it). */
export const DBIP_ATTRIBUTION = "IP geolocation by DB-IP (db-ip.com), CC BY 4.0";

function countryCell(t: TxPropagation): string {
  const parts: string[] = [];
  if (t.resolved_countries?.length) parts.push(t.resolved_countries.join(", "));
  if (t.countries.length) parts.push(`capture: ${t.countries.join(", ")} (unverified)`);
  return parts.join(" · ") || "—";
}

function secs(ms: number | null): string {
  return ms == null ? "n/a" : `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

export function RunNetworkPanel({ investigationId, runId }: { investigationId: string; runId: string }) {
  const net = useApi((s) => getRunNetwork(investigationId, runId, s), [investigationId, runId]);
  const [open, setOpen] = useState<string | null>(null);

  if (net.loading) return <section className="panel"><div className="panel-head"><h2>Network propagation</h2></div><Skeleton rows={3} /></section>;
  if (net.error) return <ErrorState error={net.error} onRetry={net.reload} />;
  const d = net.data!;
  const s = d.summary;
  const observerSources = Object.entries(s.observer_source ?? {});

  return (
    <section className="panel" aria-label="Network propagation">
      <div className="panel-head">
        <h2>Network propagation</h2>
        <EvidenceTag kind="network">From this capture</EvidenceTag>
        <span className="small faint">not part of the risk score</span>
      </div>
      <div className="metric-strip" style={{ border: 0, borderRadius: 0, margin: 0 }}>
        <Metric k="Transactions observed" v={s.transactions_with_observations} d={`${s.observations} observations`} />
        <Metric k="Distinct peers" v={s.distinct_peers} d={`${s.distinct_asns} ASNs`} />
        <Metric k="Timed" v={s.with_timing} d={`${s.with_spread} with two or more timed observations`} />
        <Metric k="Median spread" v={secs(s.median_spread_ms)} d="first to last observation, per transaction" />
        <Metric k="Peer countries" v={s.country_resolution ? (s.distinct_resolved_countries ?? 0) : "n/a"}
                d={s.country_resolution ? "resolved offline (DB-IP Lite)" : "no GeoIP database installed"} />
        <Metric k="Observer identity" v={observerSources.length === 1 && observerSources[0]![0] === "observer_id" ? "observer_id" : "mixed"}
                d={observerSources.map(([k, n]) => `${n} via ${k}`).join(" · ")} />
      </div>
      <div className="panel-body flush table-wrap" style={{ maxHeight: 420 }}>
        <table>
          <thead>
            <tr>
              <th>Transaction</th><th>First seen (UTC)</th><th className="num">Spread</th><th className="num">Obs.</th>
              <th className="num">Peers</th><th className="num">Observers</th><th className="num">ASNs</th>
              <th>First-seen peer</th><th className="num">Top peer share</th><th>Countries</th><th />
            </tr>
          </thead>
          <tbody>{d.transactions.map((t) => (
            <Row key={t.txid} t={t} open={open === t.txid} onToggle={() => setOpen(open === t.txid ? null : t.txid)} />
          ))}</tbody>
        </table>
      </div>
      <div className="panel-foot">
        {d.meaning}{" "}
        {s.country_resolution
          ? <>Peer countries are resolved offline from the peer IP ({s.country_resolution.replace(` (${DBIP_ATTRIBUTION})`, "")}); a relay&apos;s country is not the sender&apos;s. {DBIP_ATTRIBUTION}. Capture-supplied <code>geo_country</code> values are shown separately and are unverified.</>
          : <>Countries are the capture&apos;s own <code>geo_country</code> values (unverified); no offline GeoIP database is installed.</>}
        {d.transactions_total > d.transactions.length && ` Showing ${d.transactions.length} of ${d.transactions_total}.`}
      </div>
    </section>
  );
}

function Row({ t, open, onToggle }: { t: TxPropagation; open: boolean; onToggle: () => void }) {
  return (
    <>
      <tr>
        <td className="mono small" title={t.txid}>{t.txid.length > 20 ? `${t.txid.slice(0, 10)}…${t.txid.slice(-6)}` : t.txid}</td>
        <td className="mono small nowrap">{utc(t.first_seen_ms)}</td>
        <td className="num">{secs(t.spread_ms)}</td>
        <td className="num">{t.observations}</td>
        <td className="num">{t.peer_count}</td>
        <td className="num" title={t.observer_source}>{t.observer_count ?? "unknown"}</td>
        <td className="num">{t.asn_count}</td>
        <td className="mono small">{t.first_seen_peers.join(", ") || "n/a"}</td>
        <td className="num">{pct(t.dominant_peer_share, 0)}</td>
        <td className="small">{countryCell(t)}</td>
        <td><button type="button" className="btn btn-sm btn-ghost" aria-expanded={open} onClick={onToggle}>{open ? "Hide" : "Peers"}</button></td>
      </tr>
      {open && (
        <tr>
          <td colSpan={11} style={{ background: "var(--oc-surface-inset)" }}>
            <table>
              <thead><tr><th>Peer IP</th><th>Class</th><th>First arrival (UTC)</th><th className="num">After first</th><th className="num">Obs.</th><th>Observers</th><th>ASN</th><th>Country</th></tr></thead>
              <tbody>{t.peers.map((p) => (
                <tr key={p.peer_ip}>
                  <td className="mono small">{p.peer_ip}</td>
                  <td className="small">{p.ip_class === "global" ? "globally routable" : p.ip_class}</td>
                  <td className="mono small">{utc(p.first_seen_ms)}</td>
                  <td className="num">{p.first_seen_ms != null && t.first_seen_ms != null ? secs(p.first_seen_ms - t.first_seen_ms) : "n/a"}</td>
                  <td className="num">{p.observations}</td>
                  <td className="mono small">{p.observers.join(", ") || "unknown"}</td>
                  <td className="mono small">{p.asns.map((a) => `AS${a}`).join(", ") || "—"}</td>
                  <td className="small">{p.country_iso ?? "—"}</td>
                </tr>
              ))}</tbody>
            </table>
            <p className="note" style={{ marginTop: 6 }}>
              Observer identity: {t.observer_source}. Non-routable peers: {pct(t.non_routable_peer_share, 0)}. A peer relayed the
              transaction to an observer; it is not the sender.
            </p>
          </td>
        </tr>
      )}
    </>
  );
}

export function RunGraphPanel({ investigationId, runId }: { investigationId: string; runId: string }) {
  const g = useApi((s) => getRunGraph(investigationId, runId, s), [investigationId, runId]);
  if (g.loading) return <section className="panel"><div className="panel-head"><h2>Run graph</h2></div><Skeleton rows={3} /></section>;
  if (g.error) return <ErrorState error={g.error} onRetry={g.reload} />;
  const r = g.data!;
  return (
    <FlowPreview
      hiddenKinds={["cluster"]}
      nodes={r.graph.nodes}
      edges={r.graph.edges}
      truncated={r.truncated}
      explorerHref={`/graph?${new URLSearchParams({ case: investigationId, run: runId }).toString()}`}
      title="Run graph: clusters, addresses, transactions, peers, ASNs"
      note={<>{r.meaning} ANNOUNCED_BY edges carry the observation time; IN_ASN links a peer to its autonomous system.</>}
    />
  );
}
