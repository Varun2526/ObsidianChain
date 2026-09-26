/**
 * One alert from this case's own uploaded-dataset run, opened.
 *
 * Read from the run's immutable artifacts (alerts.json, the run graph and
 * network_propagation.json) by console/run_alerts.py. Evidence is grouped by
 * class so a learned association (MODEL) is never read as a rule (RULE) or an
 * observation (NETWORK); network evidence is shown beside the score and is
 * not part of it.
 */
import { Link } from "react-router-dom";

import type { CrossLayerView, RunAlertView as View, RunEvidence } from "../../api/types";
import { FlowPreview } from "../graph/FlowPreview";
import { CopyButton, EvidenceTag, Metric, fixed, int, pct, short } from "../ui/intel";
import type { EvidenceKind } from "../ui/intel";

const DBIP_ATTRIBUTION = "IP geolocation by DB-IP (db-ip.com), CC BY 4.0";

const CLASS_KIND: Record<string, EvidenceKind> = {
  MODEL: "model", RULE: "rule", NETWORK: "network", WATCHLIST: "watchlist", CONTEXT: "chain",
};

const SIGNAL_LABEL: Record<string, string> = {
  supervised_risk_model: "ObsidianChain Risk Model",
  robust_mad_deviation: "Statistical outlier",
  peeling_chain: "Peeling chain",
  coinjoin_mixing: "CoinJoin / mixing structure",
  cluster_topology: "Cluster structure",
  p2p_network_telemetry: "Network propagation",
  cross_layer_relay_coherence: "Blockchain ↔ network coherence",
  seed_risk_propagation: "Risk propagated from known seeds",
};

export function runAlertWorkspaceHref(invId: string, view: View): string {
  const q = new URLSearchParams({ case: invId, run: view.run_id, focus: `addr:${view.alert.primary_address}` });
  return `/graph?${q.toString()}`;
}

export function RunAlertAnalytical({ invId, view, actions }: { invId: string; view: View; actions?: React.ReactNode }) {
  const a = view.alert;
  const model = a.evidence.find((e) => e.signal_name === "supervised_risk_model");
  const scores = (model?.details?.model_scores ?? []) as { address: string; ml_risk_score: number }[];
  const topModel = scores.length ? Math.max(...scores.map((s) => s.ml_risk_score)) : null;
  const shownApart = (e: RunEvidence) => e.evidence_class === "NETWORK" || e.signal_name === "cross_layer_relay_coherence";
  const present = a.evidence.filter((e) => e.status === "PRESENT" && !shownApart(e));
  const absent = a.evidence.filter((e) => e.status !== "PRESENT" && !shownApart(e));
  const network = a.evidence.find((e) => e.evidence_class === "NETWORK" && e.status === "PRESENT");

  return (
    <>
      <section className="panel" aria-label="Alert summary">
        <div className="panel-head">
          <h2>Alert #{a.rank} · <span className="mono">{a.primary_address}</span></h2>
          <span className={`sev sev-${a.severity}`}>{a.severity}</span>
          <EvidenceTag kind="chain">From uploaded dataset</EvidenceTag>
          <span className="spacer" />
          {actions}
        </div>
        <div className="metric-strip" style={{ border: 0, borderRadius: 0, margin: 0 }}>
          <Metric k="Fused risk" v={fixed(a.fused_risk_score, 3)} d={`rank ${a.rank} of this run`} />
          <Metric k="Model score" v={topModel == null ? "n/a" : pct(topModel, 1)} d={`highest member · ${view.model_version ?? "model"}`} />
          <Metric k="Members" v={int(a.member_count)} d="co-spend cluster" />
          <Metric k="Transactions" v={int(view.transactions.length)} d="spent from or paid to members" />
          <Metric k="Agreeing evidence" v={int(Number(a.summary.corroborating_evidence_lines ?? 0))} d="independent lines present" />
        </div>
        <div className="panel-body">
          <div className="banner" style={{ marginTop: 0 }}>
            <h4>A ranking signal, not a finding</h4>
            <p>{a.explanation_statement} It decides where to look first; it does not establish ownership or intent.</p>
          </div>
        </div>
      </section>

      <div className="grid-2">
        <section className="panel" aria-label="Why it was flagged">
          <div className="panel-head"><h2>Why it was flagged</h2><span className="small faint">evidence by class</span></div>
          <div className="panel-body" style={{ display: "grid", gap: 14 }}>
            {present.map((e, i) => <EvidenceBlock key={i} e={e} />)}
            {absent.length > 0 && (
              <p className="note" style={{ margin: 0 }}>
                Checked, no evidence: {absent.map((e) => SIGNAL_LABEL[e.signal_name] ?? e.signal_name).join(", ")}.
              </p>
            )}
          </div>
        </section>

        <NetworkEvidence view={view} network={network} />
      </div>

      {view.cross_layer && <CrossLayerPanel invId={invId} x={view.cross_layer} />}

      <FlowPreview
        nodes={view.graph.nodes}
        edges={view.graph.edges}
        explorerHref={runAlertWorkspaceHref(invId, view)}
        title="Money flow around this alert"
        height={420}
        note={<>Cluster, member addresses, the transactions they spent into or were paid by, the counterparties, and the peers and ASNs that announced those transactions. Edges follow value: SPENDS funds a transaction, RECEIVES is paid by one. Open the explorer to trace paths across the whole run.</>}
      />

      <section className="panel" aria-label="Members and provenance">
        <div className="panel-head"><h2>Members and provenance</h2></div>
        <div className="panel-body">
          <dl className="kv">
            <dt>Member addresses</dt>
            <dd className="mono small">{view.members.join(", ")}</dd>
            <dt>Transactions</dt>
            <dd className="mono small">{view.transactions.map((t) => short(t, 10, 6)).join(", ") || "none"}</dd>
            <dt>Dataset</dt>
            <dd>{view.dataset ? <><span className="mono small">{view.dataset.filename}</span> <span className="faint small">sha256 {short(view.dataset.sha256, 12, 6)}</span></> : "n/a"}</dd>
            <dt>Run</dt>
            <dd className="mono small">{view.run_id} · fingerprint {view.run_fingerprint}</dd>
            <dt>Case reference</dt>
            <dd className="mono small">{view.alert_ref} <CopyButton value={view.alert_ref} /></dd>
          </dl>
        </div>
      </section>
    </>
  );
}

function EvidenceBlock({ e }: { e: RunEvidence }) {
  const kind = CLASS_KIND[e.evidence_class] ?? "chain";
  const scores = (e.details?.model_scores ?? []) as {
    address: string; ml_risk_score: number;
    ml_explanations?: { feature_name: string; feature_value: number; contribution: number; direction: string }[];
  }[];
  const deviations = (e.details?.deviations ?? []) as { description: string; modified_z_score: number }[];
  const top = scores.slice().sort((x, y) => y.ml_risk_score - x.ml_risk_score)[0];
  const maxAbs = Math.max(0.0001, ...(top?.ml_explanations ?? []).map((x) => Math.abs(x.contribution)));
  return (
    <div>
      <div className="row" style={{ gap: 8, marginBottom: 4 }}>
        <EvidenceTag kind={kind} />
        <strong>{SIGNAL_LABEL[e.signal_name] ?? e.signal_name}</strong>
        <span className="spacer" />
        <span className="mono small faint">{fixed(e.score, 3)}</span>
      </div>
      <p className="small" style={{ margin: 0 }}>{e.explanation}</p>
      {top?.ml_explanations && top.ml_explanations.length > 0 && (
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>Feature ({short(top.address, 8, 4)})</th><th className="num">Value</th><th>Contribution (TreeSHAP)</th></tr></thead>
          <tbody>{top.ml_explanations.map((x) => (
            <tr key={x.feature_name}>
              <td className="mono small">{x.feature_name}</td>
              <td className="num small">{fixed(x.feature_value, 4)}</td>
              <td>
                <div className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
                  <span style={{
                    display: "inline-block", height: 8, borderRadius: 2,
                    width: `${Math.round((Math.abs(x.contribution) / maxAbs) * 80)}px`,
                    background: x.contribution > 0 ? "var(--oc-sev-critical)" : "var(--oc-ev-chain, var(--oc-text-2))",
                  }} />
                  <span className="small mono">{x.contribution > 0 ? "+" : ""}{fixed(x.contribution, 3)}</span>
                  <span className="xsmall faint">{x.direction === "INCREASES_RISK" ? "raises" : "lowers"}</span>
                </div>
              </td>
            </tr>
          ))}</tbody>
        </table>
      )}
      {deviations.length > 0 && (
        <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
          {deviations.slice(0, 4).map((d, i) => <li key={i}>{d.description}</li>)}
        </ul>
      )}
    </div>
  );
}

function CrossLayerPanel({ invId, x }: { invId: string; x: CrossLayerView }) {
  const d = x.details;
  const present = x.status === "PRESENT";
  return (
    <section className="panel" aria-label="Blockchain and network correlation">
      <div className="panel-head">
        <h2>Blockchain ↔ network correlation</h2>
        <EvidenceTag kind="rule">{present ? "Layers agree" : "No coherence"}</EvidenceTag>
        <span className="small faint">does the P2P layer follow the money flow?</span>
      </div>
      <div className="metric-strip" style={{ border: 0, borderRadius: 0, margin: 0 }}>
        <Metric k="On-chain hops" v={int(d.hop_pairs)} d="a transaction spending another's output" />
        <Metric k="Same first relay" v={int(d.coherent_pairs)} d={d.expected_coherent != null ? `chance predicts ${d.expected_coherent}` : "no network data"} />
        <Metric k="Chance rate" v={d.chance_rate == null ? "n/a" : pct(d.chance_rate, 1)} d="two independent txs, this capture" />
        <Metric k="p-value" v={d.p_value == null ? "n/a" : d.p_value < 1e-4 ? d.p_value.toExponential(1) : d.p_value.toFixed(4)} d="binomial, one-sided" />
        <Metric k="Linked clusters" v={int(x.flows.reduce((n, f) => n + f.linked.length, 0))} d="tied together by a relay" />
      </div>
      <div className="panel-body">
        <p className="small" style={{ marginTop: 0 }}>{x.explanation}</p>
        {d.pairs.length > 0 && (
          <table>
            <thead><tr><th>Hop (spent → spender)</th><th>Via address</th><th>First relay (both)</th><th className="num">Δ first seen</th></tr></thead>
            <tbody>{d.pairs.slice(0, 8).map((p) => (
              <tr key={`${p.parent}-${p.child}`}>
                <td className="mono small">{short(p.parent, 8, 4)} → {short(p.child, 8, 4)}</td>
                <td className="mono small">{short(p.via_address, 8, 4)}</td>
                <td className="mono small">{p.shared_peers.join(", ")}</td>
                <td className="num small">{p.delta_ms == null ? "n/a" : `${(p.delta_ms / 1000).toFixed(1)} s`}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
        {x.flows.map((f) => (
          <div key={f.relay} style={{ marginTop: 12 }}>
            <p className="small" style={{ margin: "0 0 6px" }}>
              <strong>Flow first announced by <span className="mono">{f.relay}</span></strong>: {f.relay_coherent_pairs} hops where chance
              predicts {f.expected} (p = {f.p_value.toExponential(1)}, Bonferroni-corrected across relays). It ties this cluster to:
            </p>
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
              {f.linked.slice(0, 12).map((l) => l.alert_ref ? (
                <Link key={l.cluster_id} className="chip chip-mono" to={`/inv/${invId}/alerts/${encodeURIComponent(l.alert_ref)}`}>
                  #{l.rank} {short(l.primary_address ?? l.cluster_id, 6, 4)}{l.severity ? ` · ${l.severity}` : ""}
                </Link>
              ) : <span key={l.cluster_id} className="chip chip-mono">{short(l.cluster_id, 8, 4)}</span>)}
            </div>
          </div>
        ))}
      </div>
      <div className="panel-foot">
        An ownership lead, not a risk score: the pre-registered test (exp-net2) found that adding this line to the score made ranking worse,
        because benign services broadcast their own flows too. A relay is where observers first heard a transaction, never the sender.
      </div>
    </section>
  );
}

function NetworkEvidence({ view, network }: { view: View; network?: RunEvidence }) {
  const p = (network?.details?.propagation ?? null) as null | {
    peer_count: number; observer_count: number | null; asn_count: number; asns: number[];
    resolved_countries?: string[]; countries: string[]; dominant_peer_ip: string | null;
    dominant_peer_share: number | null; non_routable_peer_share: number | null; country_resolution?: string | null;
  };
  const rows = view.network.transactions;
  return (
    <section className="panel" aria-label="Network evidence">
      <div className="panel-head">
        <h2>Network evidence</h2>
        <EvidenceTag kind="network">From this capture</EvidenceTag>
        <span className="small faint">not part of the risk score</span>
      </div>
      {!p ? (
        <div className="panel-body"><p className="muted small" style={{ margin: 0 }}>No network observations of this alert's transactions in the capture.</p></div>
      ) : (
        <>
          <div className="panel-body">
            <dl className="kv">
              <dt>Peers · observers · ASNs</dt>
              <dd className="num">{p.peer_count} · {p.observer_count ?? "unknown"} · {p.asn_count} <span className="mono small faint">{p.asns.map((x) => `AS${x}`).join(" ")}</span></dd>
              <dt>Peer countries (from IP)</dt>
              <dd>{p.resolved_countries?.length ? p.resolved_countries.join(", ") : "none resolved"}</dd>
              {p.countries.length > 0 && <><dt>Countries (capture)</dt><dd>{p.countries.join(", ")} (unverified)</dd></>}
              <dt>Dominant peer</dt>
              <dd><span className="mono small">{p.dominant_peer_ip ?? "n/a"}</span> {p.dominant_peer_share != null && `· ${pct(p.dominant_peer_share, 0)} of announcements`}</dd>
              <dt>Non-routable peers</dt>
              <dd>{pct(p.non_routable_peer_share, 0)}</dd>
            </dl>
          </div>
          <div className="panel-body flush table-wrap">
            <table>
              <thead><tr><th>Transaction</th><th>First-seen peer</th><th className="num">Peers</th><th>Countries</th><th className="num">Spread</th></tr></thead>
              <tbody>{rows.map((t) => (
                <tr key={t.txid}>
                  <td className="mono small" title={t.txid}>{short(t.txid, 10, 6)}</td>
                  <td className="mono small">{t.first_seen_peers.join(", ") || "n/a"}</td>
                  <td className="num">{t.peer_count}</td>
                  <td className="small">{t.resolved_countries?.join(", ") || "—"}</td>
                  <td className="num small">{t.spread_ms == null ? "n/a" : `${(t.spread_ms / 1000).toFixed(1)} s`}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <div className="panel-foot">
            A peer is the IP that relayed a transaction to an observer: a vantage point, never the sender.
            {p.country_resolution ? ` ${DBIP_ATTRIBUTION}.` : ""}
          </div>
        </>
      )}
    </section>
  );
}
