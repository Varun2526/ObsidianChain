/**
 * The money-flow canvas: Cytoscape.js with a directed-flow layout (dagre,
 * left to right, value moving rightwards) and an exploration layout (fcose).
 *
 * Why Cytoscape and not a force-only canvas
 * -----------------------------------------
 * Investigators ask directional questions - where did this come from, where
 * did it go, what connects these two - and Cytoscape answers them with graph
 * algorithms over the same elements it draws (predecessors, successors, A*),
 * so a highlighted path is a computed path, not a picture of one.
 *
 * What the canvas does NOT do
 * ---------------------------
 * It invents no node and no edge. Every element comes from the API
 * (api/investigation.py trace, api/alerts.py get_alert_graph). Filtering
 * hides elements; it never removes them from the answer, and the counts say
 * how many are hidden.
 *
 * Accessibility: the canvas is a picture. Every action it offers has a button
 * equivalent, and the element list beside it (GraphElementList) is the text
 * alternative a keyboard or screen-reader user works from.
 */
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type cytoscape from "cytoscape";
import type { Core, ElementDefinition, EventObject, LayoutOptions, NodeSingular } from "cytoscape";

import type { GraphEdge, GraphNode } from "../../api/intel";

export type GraphLayout = "flow" | "explore";

export interface GraphFilters {
  hiddenKinds: Set<string>;
  /** Hide addresses whose model risk is below this (0 = show all, unscored always shown). */
  minRisk: number;
  minTimestep: number | null;
  maxTimestep: number | null;
  onlyFlagged: boolean;
}

export const NO_FILTERS: GraphFilters = {
  hiddenKinds: new Set(), minRisk: 0, minTimestep: null, maxTimestep: null, onlyFlagged: false,
};

export interface FlowGraphHandle {
  fit(): void;
  zoomBy(factor: number): void;
  relayout(): void;
  focus(id: string): void;
  /** Directed path first (the way value moved); undirected only if no directed one exists. */
  shortestPath(from: string, to: string): { ids: string[]; directed: boolean } | null;
  /** Everything upstream / downstream of a node inside the loaded graph. */
  reach(id: string, direction: "upstream" | "downstream"): string[];
  exportPng(): string | null;
}

export interface FlowGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  layout: GraphLayout;
  filters?: GraphFilters;
  selectedId?: string | null;
  highlightIds?: string[] | null;
  onSelect?: (id: string | null) => void;
  onExpand?: (id: string) => void;
  onHiddenCount?: (hidden: number) => void;
  label?: string;
}

/** jsdom has no canvas and no layout engine; the text alternative is what renders there. */
const CANVAS_AVAILABLE = typeof navigator !== "undefined" && !/jsdom/i.test(navigator.userAgent);

let modulesPromise: Promise<typeof import("cytoscape")> | null = null;

function loadCytoscape() {
  if (!modulesPromise) {
    modulesPromise = (async () => {
      const [{ default: cytoscape }, dagre, fcose] = await Promise.all([
        import("cytoscape"), import("cytoscape-dagre"), import("cytoscape-fcose"),
      ]);
      try {
        cytoscape.use(dagre.default);
        cytoscape.use(fcose.default);
      } catch {
        // Registering twice (hot reload) throws; the extensions are already in.
      }
      return cytoscape;
    })();
  }
  return modulesPromise;
}

function token(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function palette() {
  return {
    ground: token("--oc-surface-inset", "#0b0c0e"),
    node: token("--oc-surface-3", "#22252a"),
    line: token("--oc-hairline-strong", "#3a3e46"),
    edge: "#5a5f69",
    text: token("--oc-text-2", "#a9a69f"),
    textStrong: token("--oc-text", "#e9e7e2"),
    accent: token("--oc-accent", "#e7a33e"),
    critical: token("--oc-sev-critical", "#f06368"),
    high: token("--oc-sev-high", "#e8743b"),
    medium: token("--oc-sev-medium", "#d9b44a"),
    low: token("--oc-sev-low", "#9a9ca2"),
    chain: token("--oc-ev-chain", "#9fb98a"),
    network: token("--oc-ev-network", "#b59ad6"),
    watchlist: token("--oc-ev-watchlist", "#d98f6a"),
    model: token("--oc-ev-model", "#7aa7d9"),
  };
}

function toElements(nodes: GraphNode[], edges: GraphEdge[]): ElementDefinition[] {
  const out: ElementDefinition[] = [];
  for (const n of nodes) {
    const d = n.data;
    const sev = d.model?.severity ?? (typeof d.severity === "string" ? d.severity : null);
    out.push({
      group: "nodes",
      data: {
        id: n.id, kind: n.kind, label: n.label,
        severity: sev ?? "NONE",
        risk: d.model?.risk_score ?? (typeof d.risk_score === "number" ? d.risk_score : -1),
        watch: d.watchlist && d.watchlist.length > 0 ? 1 : 0,
        seed: d.seed ? 1 : 0,
        member: d.alert_member ? 1 : 0,
        timestep: typeof d.timestep === "number" ? d.timestep : -1,
      },
    });
  }
  for (const e of edges) {
    out.push({
      group: "edges",
      data: { id: e.id, source: e.source, target: e.target, kind: e.kind,
              timestep: typeof e.data.timestep === "number" ? e.data.timestep : -1 },
    });
  }
  return out;
}

function stylesheet(large: boolean): cytoscape.Stylesheet[] {
  const c = palette();
  // Cast: the pinned typings predate underlay-* and a few other properties.
  return ([
    { selector: "node", style: {
      "background-color": c.node, "border-width": 1.5, "border-color": c.line,
      width: 16, height: 16, label: "", color: c.text,
      "font-family": "IBM Plex Mono, ui-monospace, monospace", "font-size": 9,
      "text-valign": "bottom", "text-margin-y": 4, "min-zoomed-font-size": 7,
      "text-background-color": c.ground, "text-background-opacity": 0.85, "text-background-padding": "1px",
      "overlay-opacity": 0,
    } },
    { selector: 'node[kind = "address"]', style: { label: large ? "" : "data(label)" } },
    { selector: 'node[kind = "transaction"]', style: {
      shape: "round-rectangle", width: 9, height: 9, "background-color": c.line, "border-width": 0,
    } },
    { selector: 'node[kind = "cluster"]', style: {
      shape: "hexagon", width: 30, height: 30, "background-opacity": 0, "border-style": "dashed",
      "border-color": c.text, label: "data(label)", "font-family": "IBM Plex Sans, sans-serif", "font-size": 10,
    } },
    { selector: 'node[kind = "ip"]', style: {
      shape: "diamond", width: 13, height: 13, "background-color": c.ground, "border-color": c.network,
      label: large ? "" : "data(label)",
    } },
    { selector: 'node[severity = "CRITICAL"]', style: { "border-color": c.critical, "border-width": 2.5 } },
    { selector: 'node[severity = "HIGH"]', style: { "border-color": c.high, "border-width": 2.5 } },
    { selector: 'node[severity = "MEDIUM"]', style: { "border-color": c.medium, "border-width": 2 } },
    { selector: 'node[severity = "LOW"]', style: { "border-color": c.low } },
    { selector: "node[watch = 1]", style: {
      "underlay-color": c.watchlist, "underlay-opacity": 0.35, "underlay-padding": 5, "underlay-shape": "ellipse",
    } },
    { selector: "node[member = 1]", style: { "background-color": "#2d3036" } },
    { selector: "node[seed = 1]", style: {
      width: 24, height: 24, "background-color": c.accent, "background-opacity": 0.25,
      "border-color": c.accent, "border-width": 2.5, label: "data(label)", color: c.textStrong,
    } },
    { selector: "edge", style: {
      width: 1.1, "line-color": c.edge, "target-arrow-color": c.edge, "target-arrow-shape": "triangle",
      "arrow-scale": 0.7, "curve-style": large ? "haystack" : "bezier", opacity: 0.75,
    } },
    { selector: 'edge[kind = "MEMBER_OF"]', style: {
      "line-style": "dashed", "line-dash-pattern": [3, 3], "target-arrow-shape": "none", opacity: 0.35, width: 0.8,
    } },
    { selector: 'edge[kind = "ANNOUNCED_BY"]', style: {
      "line-style": "dotted", "line-color": c.network, "target-arrow-color": c.network, opacity: 0.55,
    } },
    { selector: ".dim", style: { opacity: 0.12 } },
    { selector: "edge.dim", style: { opacity: 0.06 } },
    { selector: "node.hl", style: { opacity: 1, label: "data(label)", color: c.textStrong } },
    { selector: "edge.hl", style: {
      opacity: 1, width: 2.2, "line-color": c.accent, "target-arrow-color": c.accent, "z-index": 10,
    } },
    { selector: "node:selected", style: {
      "border-color": c.accent, "border-width": 3, label: "data(label)", color: c.textStrong, "z-index": 20,
    } },
    { selector: ".filtered", style: { display: "none" } },
  ] as unknown) as cytoscape.Stylesheet[];
}

function layoutOptions(kind: GraphLayout, count: number, incremental: boolean): LayoutOptions {
  if (kind === "flow") {
    return {
      name: "dagre", rankDir: "LR", ranker: "network-simplex", nodeSep: 10, rankSep: 90, edgeSep: 4,
      animate: false, fit: false, padding: 32,
    } as unknown as LayoutOptions;
  }
  return {
    name: "fcose", quality: count > 800 ? "draft" : "default", randomize: !incremental,
    animate: false, fit: true, padding: 32, nodeRepulsion: 6500, idealEdgeLength: 55,
    nodeSeparation: 60, packComponents: true,
  } as unknown as LayoutOptions;
}

/**
 * Flow layout ranks only the value-flow subgraph (addresses, transactions,
 * SPENDS/PAYS). Cluster membership and relay edges are not value movement;
 * letting dagre rank them folded every alert graph into one tall column.
 * Clusters are then placed left of the flow and relay peers right of it.
 */
function runLayout(cy: Core, kind: GraphLayout, incremental: boolean) {
  if (cy.elements().length === 0) return;
  // Measured, not asserted: read with performance.getEntriesByName("oc-graph-layout").
  const t0 = performance.now();
  try { layoutOnce(cy, kind, incremental); } finally {
    performance.measure?.("oc-graph-layout", { start: t0, detail: { kind, nodes: cy.nodes().length, edges: cy.edges().length } });
  }
}

function layoutOnce(cy: Core, kind: GraphLayout, incremental: boolean) {
  if (kind !== "flow") {
    cy.layout(layoutOptions(kind, cy.nodes().length, incremental)).run();
    return;
  }
  const flow = cy.nodes('[kind = "address"], [kind = "transaction"]')
    .union(cy.edges('[kind = "SPENDS"], [kind = "PAYS"]'));
  flow.layout(layoutOptions("flow", flow.nodes().length, incremental)).run();
  const bb = flow.nodes().boundingBox({});
  const place = (sel: string, x: number) => {
    const nodes = cy.nodes(sel);
    const step = nodes.length > 1 ? Math.max(28, (bb.h || 200) / (nodes.length - 1)) : 0;
    nodes.forEach((n, i) => { n.position({ x, y: bb.y1 + i * step + (nodes.length === 1 ? bb.h / 2 : 0) }); });
  };
  place('[kind = "cluster"]', bb.x1 - 160);
  place('[kind = "ip"]', bb.x2 + 160);
  cy.fit(cy.elements(), 32);
  // A wide fan-out ranks into a column far taller than the viewport; fitting
  // it shrinks every node to a dot. Stay readable and centre on the seeds.
  if (cy.zoom() < 0.45) {
    const seeds = cy.nodes("[seed = 1]");
    cy.zoom(0.6);
    cy.center(seeds.nonempty() ? seeds : cy.nodes());
  }
}

/** Flow reads best for small traces; beyond this a force layout shows structure better. */
export function defaultLayout(nodeCount: number): GraphLayout {
  return nodeCount > 120 ? "explore" : "flow";
}

function applyFilters(cy: Core, f: GraphFilters): number {
  let hidden = 0;
  cy.batch(() => {
    cy.elements().removeClass("filtered");
    cy.nodes().forEach((n) => {
      const kind = n.data("kind") as string;
      const risk = n.data("risk") as number;
      const ts = n.data("timestep") as number;
      let hide = f.hiddenKinds.has(kind);
      if (!hide && kind === "address" && f.minRisk > 0 && risk >= 0 && risk < f.minRisk) hide = true;
      if (!hide && kind === "address" && f.onlyFlagged && risk < 0 && !n.data("watch") && !n.data("seed")) hide = true;
      if (!hide && kind === "transaction" && ts >= 0) {
        if (f.minTimestep != null && ts < f.minTimestep) hide = true;
        if (f.maxTimestep != null && ts > f.maxTimestep) hide = true;
      }
      if (hide && !n.data("seed")) { n.addClass("filtered"); hidden += 1; }
    });
    // An edge with a hidden end is hidden; a transaction left with no visible
    // address on either side is noise, and is hidden with it.
    cy.edges().forEach((e) => {
      if (e.source().hasClass("filtered") || e.target().hasClass("filtered")) e.addClass("filtered");
    });
    cy.nodes('[kind = "transaction"]').forEach((t) => {
      if (!t.hasClass("filtered") && t.connectedEdges().not(".filtered").length === 0) {
        t.addClass("filtered"); hidden += 1;
      }
    });
  });
  return hidden;
}

export const FlowGraph = forwardRef<FlowGraphHandle, FlowGraphProps>(function FlowGraph(props, ref) {
  const { nodes, edges, layout, filters = NO_FILTERS, selectedId, highlightIds,
          onSelect, onExpand, onHiddenCount, label = "Money-flow graph" } = props;
  const host = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const handlers = useRef({ onSelect, onExpand, onHiddenCount });
  handlers.current = { onSelect, onExpand, onHiddenCount };
  const layoutRef = useRef(layout);
  layoutRef.current = layout;
  const appliedLayout = useRef<GraphLayout | null>(null);

  // mount once
  useEffect(() => {
    if (!CANVAS_AVAILABLE || !host.current) return;
    let cancelled = false;
    loadCytoscape().then((cytoscape) => {
      if (cancelled || !host.current) return;
      const cy = cytoscape({
        container: host.current,
        elements: [],
        style: stylesheet(false),
        minZoom: 0.08, maxZoom: 4, wheelSensitivity: 0.25,
        boxSelectionEnabled: false, selectionType: "single",
        pixelRatio: 1, textureOnViewport: false,
      });
      cy.on("tap", "node", (evt: EventObject) => handlers.current.onSelect?.((evt.target as NodeSingular).id()));
      cy.on("tap", (evt: EventObject) => { if (evt.target === cy) handlers.current.onSelect?.(null); });
      cy.on("dbltap", "node", (evt: EventObject) => handlers.current.onExpand?.((evt.target as NodeSingular).id()));
      cyRef.current = cy;
      setReady(true);
    }).catch((err: unknown) => setFailed(String((err as Error)?.message ?? err)));
    return () => {
      cancelled = true;
      cyRef.current?.destroy();
      cyRef.current = null;
    };
  }, []);

  // elements: diff so expansion keeps what the investigator already arranged
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !ready) return;
    const wanted = toElements(nodes, edges);
    const wantedIds = new Set(wanted.map((e) => e.data.id as string));
    const had = cy.elements().length;
    const large = nodes.length > 900;
    cy.batch(() => {
      cy.elements().filter((e) => !wantedIds.has(e.id())).remove();
      const fresh = wanted.filter((e) => cy.getElementById(e.data.id as string).empty());
      // new nodes start beside a neighbour that already has a position
      for (const el of fresh) {
        if (el.group !== "nodes") continue;
        const id = el.data.id as string;
        const anchor = edges.find((e) => e.source === id || e.target === id);
        const other = anchor ? (anchor.source === id ? anchor.target : anchor.source) : null;
        const at = other ? cy.getElementById(other) : null;
        if (at && at.nonempty()) {
          const p = at.position();
          el.position = { x: p.x + (Math.random() - 0.5) * 60, y: p.y + (Math.random() - 0.5) * 60 };
        }
      }
      cy.add(fresh);
    });
    cy.style(stylesheet(large));
    runLayout(cy, layoutRef.current, had > 0);
    appliedLayout.current = layoutRef.current;
    handlers.current.onHiddenCount?.(applyFilters(cy, filters));
    // filters handled below; layout changes handled below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges, ready]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !ready || appliedLayout.current === layout) return;
    runLayout(cy, layout, false);
    appliedLayout.current = layout;
  }, [layout, ready]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !ready) return;
    handlers.current.onHiddenCount?.(applyFilters(cy, filters));
  }, [filters, ready, nodes, edges]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !ready) return;
    cy.batch(() => {
      cy.elements().unselect();
      if (selectedId) cy.getElementById(selectedId).select();
    });
  }, [selectedId, ready, nodes]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !ready) return;
    cy.batch(() => {
      cy.elements().removeClass("dim hl");
      if (highlightIds && highlightIds.length > 0) {
        const keep = new Set(highlightIds);
        cy.elements().forEach((el) => { el.addClass(keep.has(el.id()) ? "hl" : "dim"); });
      }
    });
  }, [highlightIds, ready, nodes, edges]);

  useImperativeHandle(ref, () => ({
    fit() { cyRef.current?.fit(cyRef.current.elements(":visible"), 32); },
    zoomBy(factor: number) {
      const cy = cyRef.current;
      if (!cy) return;
      cy.zoom({ level: cy.zoom() * factor, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
    },
    relayout() {
      const cy = cyRef.current;
      if (cy) runLayout(cy, layoutRef.current, false);
    },
    focus(id: string) {
      const cy = cyRef.current;
      const el = cy?.getElementById(id);
      if (cy && el && el.nonempty()) cy.animate({ center: { eles: el }, zoom: Math.max(cy.zoom(), 1.2) }, { duration: 0 });
    },
    shortestPath(from: string, to: string) {
      const cy = cyRef.current;
      if (!cy) return null;
      const root = cy.getElementById(from);
      const goal = cy.getElementById(to);
      if (root.empty() || goal.empty()) return null;
      const visible = cy.elements().not(".filtered");
      for (const directed of [true, false]) {
        const r = visible.aStar({ root, goal, directed });
        if (r.found) return { ids: r.path.map((e) => e.id()), directed };
      }
      return null;
    },
    reach(id: string, direction: "upstream" | "downstream") {
      const cy = cyRef.current;
      const el = cy?.getElementById(id);
      if (!cy || !el || el.empty()) return [];
      const set = direction === "upstream" ? el.predecessors() : el.successors();
      return [id, ...set.not('[kind = "cluster"]').filter((x) => !x.isEdge() || x.data("kind") !== "MEMBER_OF").map((x) => x.id())];
    },
    exportPng() {
      const cy = cyRef.current;
      return cy ? cy.png({ full: true, scale: 2, bg: palette().ground }) : null;
    },
  }), []);

  if (!CANVAS_AVAILABLE) {
    return (
      <div className="graph-canvas" role="img" aria-label={`${label}: ${nodes.length} nodes, ${edges.length} edges`}>
        <div className="graph-empty">{nodes.length} nodes · {edges.length} edges (canvas not available here; use the element list)</div>
      </div>
    );
  }
  return (
    <>
      <div ref={host} className="graph-canvas" role="img"
           aria-label={`${label}: ${nodes.length} nodes and ${edges.length} edges. The element list beside the graph is the text alternative.`} />
      {failed && <div className="graph-empty">The graph renderer could not start: {failed}</div>}
      {!failed && nodes.length === 0 && ready && <div className="graph-empty">Nothing to draw yet.</div>}
    </>
  );
});
