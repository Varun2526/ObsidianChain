/**
 * The investigation graph, drawn with Cytoscape.js.
 *
 * What is rendered, and what is NOT
 * ---------------------------------
 * The API exposes address-to-address relationships only. There are no
 * transaction ids in the alert contract, so there are no transaction nodes
 * here: inventing one per edge would draw a node the backend never
 * described. The legend says so rather than leaving a reader to assume the
 * graph is complete.
 *
 * Two node kinds are real and drawn: the CLUSTER (the alert itself - an
 * inference from the common-input-ownership heuristic, not a person) and the
 * ADDRESS nodes that belong to it.
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

import type { AlertDetail, RelationshipEdge } from "../api/types";
import { SeverityBadge } from "./primitives";

const DEFAULT_CAP = 60;
const CAPS = [30, 60, 120, 250];

export interface GraphModel {
  elements: ElementDefinition[];
  renderedAddresses: number;
  withheldAddresses: number;
  renderedEdges: number;
  withheldEdges: number;
}

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

  return {
    elements,
    renderedAddresses: chosen.size,
    withheldAddresses: Math.max(0, alert.members.total_scored - chosen.size),
    renderedEdges: keptEdges.length,
    withheldEdges,
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

  const model = useMemo(() => buildGraphModel(alert, cap), [alert, cap]);

  useEffect(() => {
    if (!host.current) return;
    const instance = cytoscape({
      container: host.current,
      elements: model.elements,
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
            "background-color": "#151a26", "border-width": 2,
            "border-color": (n: any) => SEVERITY_COLOUR[n.data("severity")] ?? "#4da3ff",
            shape: "round-rectangle", width: 108, height: 34,
            label: "data(label)", "font-size": 11, "font-weight": 600,
            color: "#dbe2f0", "text-valign": "center",
          },
        },
        {
          selector: 'edge[kind="MEMBER_OF"]',
          style: { width: 1, "line-color": "#232a3a", "curve-style": "straight",
                   "line-style": "dotted" },
        },
        {
          selector: 'edge[kind="CO_SPEND_COMPONENT"]',
          style: { width: 2, "line-color": "#4da3ff", "curve-style": "bezier",
                   opacity: 0.75 },
        },
        {
          selector: 'edge[kind="FUNDED_VIA_TRANSACTION"]',
          style: {
            width: 2, "line-color": "#4ddba4", "curve-style": "bezier",
            "target-arrow-color": "#4ddba4", "target-arrow-shape": "triangle",
            opacity: 0.8,
          },
        },
        { selector: ":selected", style: { "border-width": 3, "border-color": "#fff" } },
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
  }, [model]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Investigation graph</h2>
        <span className="small muted">
          {model.renderedAddresses} addresses · {model.renderedEdges} relationships
        </span>
        <span className="spacer" style={{ flex: 1 }} />
        <div className="toggles" role="group" aria-label="Node cap">
          {CAPS.map((n) => (
            <button key={n} className="toggle" aria-pressed={cap === n}
              onClick={() => setCap(n)}>{n}</button>
          ))}
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
            {inspected.observedAt !== undefined ? (
              <><dt>Observed at</dt><dd>t{String(inspected.observedAt)}</dd></>
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
          <strong>No transaction nodes are shown.</strong> The alert API exposes
          address-to-address relationships only, so drawing transaction nodes
          would mean inventing objects the backend never described.
        </p>
        <p className="note">{alert.relationships.note}</p>
      </div>
    </section>
  );
}
