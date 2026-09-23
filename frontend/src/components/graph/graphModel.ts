/**
 * Pure helpers over the API's graph payloads, kept out of the components so
 * they are testable without a canvas.
 */
import type { GraphEdge, GraphNode } from "../../api/intel";

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/**
 * Merge an expansion into the loaded graph. Existing nodes keep their
 * identity; a node's annotations are refreshed if the new payload has them;
 * a seed flag is never lost. Edges whose ends are missing are dropped, so a
 * merge cannot produce a dangling edge even if a payload had one.
 */
export function mergeGraph(base: GraphData, add: GraphData): GraphData {
  const nodes = new Map(base.nodes.map((n) => [n.id, n]));
  for (const n of add.nodes) {
    const had = nodes.get(n.id);
    nodes.set(n.id, had ? { ...had, data: { ...had.data, ...n.data, seed: Boolean(had.data.seed || n.data.seed) } } : n);
  }
  const edges = new Map(base.edges.map((e) => [e.id, e]));
  for (const e of add.edges) edges.set(e.id, e);
  const kept = [...edges.values()].filter((e) => nodes.has(e.source) && nodes.has(e.target));
  return { nodes: [...nodes.values()], edges: kept };
}

export function countKinds(nodes: GraphNode[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const n of nodes) out[n.kind] = (out[n.kind] ?? 0) + 1;
  return out;
}

export function neighbours(id: string, edges: GraphEdge[]): { incoming: GraphEdge[]; outgoing: GraphEdge[] } {
  return {
    incoming: edges.filter((e) => e.target === id),
    outgoing: edges.filter((e) => e.source === id),
  };
}

export function timestepRange(nodes: GraphNode[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const n of nodes) {
    const t = n.data.timestep;
    if (n.kind === "transaction" && typeof t === "number") { lo = Math.min(lo, t); hi = Math.max(hi, t); }
  }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

export function matchNodes(nodes: GraphNode[], q: string, limit = 12): GraphNode[] {
  const needle = q.trim().toLowerCase();
  if (!needle) return [];
  return nodes.filter((n) => n.id.toLowerCase().includes(needle) || n.label.toLowerCase().includes(needle)).slice(0, limit);
}

export const KIND_LABEL: Record<string, string> = {
  address: "Addresses",
  transaction: "Transactions",
  cluster: "Clusters",
  ip: "Relay peers",
};
