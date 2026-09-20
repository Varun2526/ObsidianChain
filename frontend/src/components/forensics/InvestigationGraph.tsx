/**
 * The investigation graph, drawn with Cytoscape.js.
 *
 * What is rendered
 * ----------------
 * Four node kinds, every one of them backed by a field the API returns:
 *
 *   CLUSTER      the alert - an inference from the common-input-ownership
 *                heuristic, not a person
 *   ADDRESS      a member wallet
 *   TRANSACTION  a txid from the network correlation
 *   IP           a peer that ANNOUNCED a transaction
 *
 * The IP -> TRANSACTION -> ADDRESS chain is the correlation the problem
 * statement asks for, and each link is an observed fact: an observer heard
 * this transaction from this peer, and this address took part in it.
 *
 * An IP edge is labelled ANNOUNCED_BY and never "sent". 84.3% of
 * transactions in this dataset were announced by more than one peer, which
 * is what gossip relay looks like - so a peer is a vantage point, not an
 * originator, and the panel says so.
 *
 * Readability
 * -----------
 * A cluster can hold thousands of members. The graph renders the
 * highest-risk addresses up to a cap, states how many were withheld, and
 * lets the investigator raise the cap deliberately. Drawing 11,001 nodes by
 * default would produce a hairball that answers nothing.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import cytoscape from "cytoscape";
import type { Core, ElementDefinition } from "cytoscape";

import type { AlertDetail, RelationshipEdge } from "../../api/types";
import { SeverityBadge } from "../ui/primitives";

const DEFAULT_CAP = 60;
const CAPS = [30, 60, 120, 250];

export interface GraphModel {
  elements: ElementDefinition[];
  renderedAddresses: number;
  withheldAddresses: number;
  renderedEdges: number;
  withheldEdges: number;
  renderedTransactions: number;
  renderedIps: number;
}

/** Transactions pulled into the graph. Beyond this it stops being readable. */
export const TRANSACTIONS_IN_GRAPH = 12;

/**
 * Pure: build the Cytoscape elements from an alert.
 *
 * Extracted so it can be tested without a DOM or a rendering engine - the
 * interesting logic is which nodes survive the cap and whether edges are
 * dropped when an endpoint does, not how Cytoscape paints them.
 */
export function buildGraphModel(alert: AlertDetail, cap: number): GraphModel {
  const ranked = [...alert.members.rows].sort(
    (a, b) => (b.risk_score ?? 0) - (a.risk_score ?? 0),
  );
  const scored = new Map(ranked.map((m) => [m.address, m]));

  // Node set = the highest-risk members WITH a score, plus the endpoints of
  // the relationships we are about to draw.
  //
  // The second half matters and is easy to get wrong. The API caps
  // `members.rows` (25 by default) while `relationships.edges` span the whole
  // cluster, so building nodes from members alone drops almost every edge for
  // want of an endpoint - a graph panel that reports "0 relationships" on an
  // alert that has 147 of them. Endpoints outside the scored list are still
  // real cluster members; we simply do not know their individual risk, and
  // they are drawn as UNSCORED rather than given a fabricated one.
  const chosen = new Map<string, { severity: string; risk: number | null;
                                   observedAt: number | null }>();
  for (const member of ranked.slice(0, cap)) {
    chosen.set(member.address, {
      severity: member.severity,
      risk: member.risk_score,
      observedAt: member.observed_at_timestep,
    });
  }

  const keptEdges: RelationshipEdge[] = [];
  let withheldEdges = 0;
  const seen = new Set<string>();
  for (const edge of alert.relationships.edges) {
    const id = edgeId(edge);
    if (seen.has(id)) continue;
    const room = cap - chosen.size;
    const missing = [edge.address_a, edge.address_b].filter((a) => !chosen.has(a));
    if (missing.length > room) { withheldEdges += 1; continue; }
    for (const address of missing) {
      const known = scored.get(address);
      chosen.set(address, {
        severity: known?.severity ?? "UNSCORED",
        risk: known?.risk_score ?? null,
        observedAt: known?.observed_at_timestep ?? null,
      });
    }
    seen.add(id);
    keptEdges.push(edge);
  }

  const clusterId = `cluster:${alert.summary.cluster_id}`;
  const elements: ElementDefinition[] = [
    {
      data: {
        id: clusterId,
        kind: "cluster",
        label: `Cluster ${alert.summary.cluster_id}`,
        severity: alert.summary.severity,
      },
    },
  ];

  for (const [address, info] of chosen) {
    elements.push({
      data: {
        id: address,
        kind: "address",
        label: `${address.slice(0, 8)}\u2026`,
        address,
        risk: info.risk,
        severity: info.severity,
        observedAt: info.observedAt,
        scored: info.risk !== null,
      },
    });
    elements.push({
      data: {
        id: `member:${address}`,
        source: clusterId,
        target: address,
        kind: "MEMBER_OF",
      },
    });
  }

  for (const edge of keptEdges) {
    elements.push({
      data: {
        id: edgeId(edge),
        source: edge.address_a,
        target: edge.address_b,
        kind: edge.relationship,
        timestep: edge.timestep,
      },
    });
  }

  // ---- the network correlation: IP -> TRANSACTION -> ADDRESS ----------
  //
  // Only transactions that actually touch a rendered address are drawn.
  // Pulling in a transaction whose wallets are all off-graph would add a
  // node with nothing to connect to.
  let renderedTransactions = 0;
  const ips = new Set<string>();
  if (alert.correlation?.available) {
    for (const tx of alert.correlation.transactions.slice(0, TRANSACTIONS_IN_GRAPH)) {
      const involved = tx.addresses.filter((a) => chosen.has(a.address));
      if (involved.length === 0) continue;
      const txNode = `tx:${tx.txid}`;
      elements.push({
        data: {
          id: txNode, kind: "transaction", label: `tx ${tx.txid.slice(0, 8)}`,
          txid: tx.txid, announcingPeers: tx.announcing_peers_total,
          firstSeenMs: tx.first_seen_ms,
        },
      });
      renderedTransactions += 1;
      for (const involvement of involved) {
        elements.push({
          data: {
            id: `involves:${tx.txid}:${involvement.address}:${involvement.role}`,
            source: txNode, target: involvement.address,
            kind: "INVOLVES", role: involvement.role,
          },
        });
      }
      for (const peer of tx.peers) {
        const ipNode = `ip:${peer.ip}`;
        if (!ips.has(peer.ip)) {
          ips.add(peer.ip);
          elements.push({
            data: {
              id: ipNode, kind: "ip", label: peer.ip, ip: peer.ip,
              asn: peer.asn, port: peer.port, observers: peer.observers,
            },
          });
        }
        elements.push({
          data: {
            id: `announced:${peer.ip}:${tx.txid}`,
            source: ipNode, target: txNode, kind: "ANNOUNCED_BY",
            observers: peer.observers,
          },
        });
      }
    }
  }

  return {
    elements,
    renderedAddresses: chosen.size,
    withheldAddresses: Math.max(0, alert.members.total_scored - chosen.size),
    renderedEdges: keptEdges.length,
    withheldEdges,
    renderedTransactions,
    renderedIps: ips.size,
  };
}

function edgeId(edge: RelationshipEdge): string {
  const [a, b] = [edge.address_a, edge.address_b].sort();
  return `${edge.relationship}:${a}|${b}`;
}

const SEVERITY_COLOUR: Record<string, string> = {
  CRITICAL: "#ff4d5e", HIGH: "#ff9f45", MEDIUM: "#ffd54a", LOW: "#5d6880",
  // An address pulled in as a relationship endpoint but outside the API's
  // capped member list. Its risk is unknown, not zero.
  UNSCORED: "#3b445a",
};

export function InvestigationGraph({ alert }: { alert: AlertDetail }) {
  const host = useRef<HTMLDivElement>(null);
  const core = useRef<Core | null>(null);
  const [cap, setCap] = useState(DEFAULT_CAP);
  const [inspected, setInspected] = useState<Record<string, unknown> | null>(null);

  const [showAddresses, setShowAddresses] = useState(true);
  const [showTransactions, setShowTransactions] = useState(true);
  const [showIps, setShowIps] = useState(true);

  const model = useMemo(() => buildGraphModel(alert, cap), [alert, cap]);

  const activeElements = useMemo(() => {
    return model.elements.filter((el) => {
      const kind = el.data.kind;
      if (kind === "address" && !showAddresses) return false;
      if (kind === "transaction" && !showTransactions) return false;
      if (kind === "ip" && !showIps) return false;
      if (el.data.source && el.data.target) {
        const edgeKind = el.data.kind;
        if (edgeKind === "ANNOUNCED_BY" && (!showIps || !showTransactions)) return false;
        if (edgeKind === "INVOLVES" && (!showAddresses || !showTransactions)) return false;
        if (edgeKind === "MEMBER_OF" && !showAddresses) return false;
        if (edgeKind === "CO_SPEND_COMPONENT" && !showAddresses) return false;
        if (edgeKind === "FUNDED_VIA_TRANSACTION" && !showAddresses) return false;
      }
      return true;
    });
  }, [model, showAddresses, showTransactions, showIps]);

  useEffect(() => {
    if (!host.current) return;
    const instance = cytoscape({
      container: host.current,
      elements: activeElements,
      style: [
        {
          selector: 'node[kind="address"]',
          style: {
            "background-color": (n: any) => SEVERITY_COLOUR[n.data("severity")] ?? "#5d6880",
            width: 20, height: 20, label: "data(label)",
            "font-size": 8, color: "#8b97ae", "text-valign": "bottom",
            "text-margin-y": 3, "border-width": 1, "border-color": "#0b0e14",
          },
        },
        {
          selector: 'node[kind="cluster"]',
          style: {
            "background-color": "#050508", "border-width": 2,
            "border-color": (n: any) => SEVERITY_COLOUR[n.data("severity")] ?? "#ffffff",
            shape: "round-rectangle", width: 112, height: 36,
            label: "data(label)", "font-size": 11, "font-weight": 700,
            color: "#ffffff", "text-valign": "center",
          },
        },
        {
          selector: 'edge[kind="MEMBER_OF"]',
          style: { width: 1, "line-color": "#1f1f26", "curve-style": "straight",
                   "line-style": "dotted" },
        },
        {
          selector: 'edge[kind="CO_SPEND_COMPONENT"]',
          style: { width: 2, "line-color": "#38bdf8", "curve-style": "bezier",
                   opacity: 0.8 },
        },
        {
          selector: 'edge[kind="FUNDED_VIA_TRANSACTION"]',
          style: {
            width: 2, "line-color": "#00f0aa", "curve-style": "bezier",
            "target-arrow-color": "#00f0aa", "target-arrow-shape": "triangle",
            opacity: 0.85,
          },
        },
        {
          selector: 'node[kind="transaction"]',
          style: {
            "background-color": "#0e0e14", "border-width": 1,
            "border-color": "#9ca3af", shape: "diamond",
            width: 22, height: 22, label: "data(label)",
            "font-size": 9, color: "#9ca3af", "text-valign": "bottom",
            "text-margin-y": 3,
          },
        },
        {
          selector: 'node[kind="ip"]',
          style: {
            "background-color": "#c084fc", shape: "hexagon",
            width: 18, height: 18, label: "data(label)",
            "font-size": 9, color: "#c084fc", "text-valign": "top",
            "text-margin-y": -3,
          },
        },
        {
          selector: 'edge[kind="INVOLVES"]',
          style: { width: 1.5, "line-color": "#52525e", "curve-style": "bezier",
                   opacity: 0.6 },
        },
        {
          selector: 'edge[kind="ANNOUNCED_BY"]',
          style: {
            width: 1.5, "line-color": "#c084fc", "curve-style": "bezier",
            "target-arrow-color": "#c084fc", "target-arrow-shape": "vee",
            "line-style": "dashed", opacity: 0.75,
          },
        },
        { selector: ":selected", style: { "border-width": 3, "border-color": "#ffffff" } },
      ],
      layout: {
        name: "cose", animate: false,
        // Typed as a callback in @types/cytoscape; a constant is the
        // documented usage and the cast keeps it honest rather than
        // loosening the whole layout type.
        nodeRepulsion: (() => 9000) as unknown as number,
        idealEdgeLength: (() => 65) as unknown as number,
      } as cytoscape.LayoutOptions,
      wheelSensitivity: 0.2,
    });
    // Without this the cose layout spreads nodes across the full container
    // width and the visible area can look empty on a wide screen.
    instance.ready(() => instance.fit(undefined, 30));
    instance.on("tap", "node", (event) => setInspected(event.target.data()));
    instance.on("tap", (event) => { if (event.target === instance) setInspected(null); });
    core.current = instance;
    return () => { instance.destroy(); core.current = null; };
  }, [activeElements]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Investigation graph</h2>
        <span className="small muted">
          {model.renderedAddresses} addresses · {model.renderedEdges} relationships
          {model.renderedTransactions > 0
            ? ` · ${model.renderedTransactions} transactions · ${model.renderedIps} peers`
            : ""}
        </span>
        <span className="spacer" style={{ flex: 1 }} />
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <div className="toggles" role="group" aria-label="Node types">
            <button className="toggle" aria-pressed={showAddresses}
              onClick={() => setShowAddresses(!showAddresses)}>Addresses</button>
            <button className="toggle" aria-pressed={showTransactions}
              onClick={() => setShowTransactions(!showTransactions)}>Txids</button>
            <button className="toggle" aria-pressed={showIps}
              onClick={() => setShowIps(!showIps)}>Peers</button>
          </div>
          <div className="toggles" role="group" aria-label="Node cap">
            {CAPS.map((n) => (
              <button key={n} className="toggle" aria-pressed={cap === n}
                onClick={() => setCap(n)}>{n}</button>
            ))}
          </div>
          <button className="btn btn-sm" onClick={() => core.current?.fit(undefined, 30)}>
            Fit View
          </button>
        </div>
      </div>

      <div className="graph-host" ref={host} data-testid="graph-host" />

      <div className="legend">
        <span><i style={{ background: "#151a26", border: "2px solid #4da3ff",
          borderRadius: 2, width: 14, height: 10 }} /> Cluster (inferred, not a person)</span>
        <span><i style={{ background: "#ff4d5e" }} /> Address · CRITICAL</span>
        <span><i style={{ background: "#ff9f45" }} /> HIGH</span>
        <span><i style={{ background: "#ffd54a" }} /> MEDIUM</span>
        <span><i style={{ background: "#5d6880" }} /> LOW</span>
        <span><i style={{ background: "#3b445a" }} /> Unscored (outside the capped member list)</span>
        <span><i className="edge" style={{ borderColor: "#4da3ff" }} /> CO_SPEND_COMPONENT</span>
        <span><i className="edge" style={{ borderColor: "#4ddba4" }} /> FUNDED_VIA_TRANSACTION</span>
        <span><i style={{ background: "#1b2130", border: "1px solid #8b97ae",
          transform: "rotate(45deg)", borderRadius: 0 }} /> Transaction</span>
        <span><i style={{ background: "#b388ff" }} /> Announcing peer (relay, not sender)</span>
        <span><i className="edge" style={{ borderColor: "#b388ff",
          borderTopStyle: "dashed" }} /> ANNOUNCED_BY</span>
      </div>

      <div className="panel-body">
        {inspected ? (
          <dl className="kv">
            <dt>Node</dt><dd>{String(inspected.kind)}</dd>
            {inspected.address ? (<><dt>Address</dt><dd>{String(inspected.address)}</dd></>) : null}
            {inspected.severity ? (
              <><dt>Severity</dt>
                <dd><SeverityBadge severity={inspected.severity as any} /></dd></>
            ) : null}
            <dt>Risk</dt>
            <dd>
              {inspected.risk === null || inspected.risk === undefined
                ? <span className="faint">not in the scored member list</span>
                : Number(inspected.risk).toFixed(4)}
            </dd>
            {inspected.observedAt !== undefined && inspected.observedAt !== null ? (
              <><dt>Observed at</dt><dd>t{String(inspected.observedAt)}</dd></>
            ) : null}
            {inspected.txid ? (
              <><dt>Transaction</dt><dd>{String(inspected.txid)}</dd>
                <dt>Announcing peers</dt>
                <dd>{String(inspected.announcingPeers)}{" "}
                  <span className="faint">(a peer is a relay vantage point, not a sender)</span>
                </dd></>
            ) : null}
            {inspected.ip ? (
              <><dt>Peer IP</dt><dd>{String(inspected.ip)}</dd>
                <dt>ASN</dt><dd>{String(inspected.asn ?? "n/a")}</dd>
                <dt>Port</dt><dd>{String(inspected.port ?? "n/a")}</dd>
                <dt>Seen by</dt><dd>{String(inspected.observers)} observers</dd></>
            ) : null}
          </dl>
        ) : (
          <p className="note" style={{ marginTop: 0 }}>
            Click a node to inspect it.
          </p>
        )}

        <p className="note">
          {model.withheldAddresses > 0 || model.withheldEdges > 0 ? (
            <>
              Showing the {model.renderedAddresses} highest-risk members of{" "}
              {alert.members.total_scored.toLocaleString()} scored.{" "}
              {model.withheldAddresses.toLocaleString()} addresses and{" "}
              {model.withheldEdges.toLocaleString()} relationships are not drawn —
              raise the cap above to include more.{" "}
            </>
          ) : null}
          {model.renderedTransactions > 0 ? (
            <>
              <strong>Transaction and peer nodes come from the network
              correlation.</strong> An <span style={{ color: "var(--network)" }}>
              ANNOUNCED_BY</span> edge means an observer heard that transaction
              from that peer. It does <strong>not</strong> mean the peer sent
              it, and two transactions sharing a peer are not thereby the same
              party.
            </>
          ) : (
            <>
              <strong>No transaction or peer nodes are shown.</strong> No
              announcement observations correlated to this alert's
              transactions, so there is nothing truthful to draw.
            </>
          )}
        </p>
        <p className="note">{alert.relationships.note}</p>
      </div>
    </section>
  );
}
