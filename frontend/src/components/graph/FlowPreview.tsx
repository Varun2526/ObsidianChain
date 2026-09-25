/**
 * A bounded money-flow view embedded in a detail page, with a way out to
 * the full explorer. Same canvas, same data, fewer controls.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import type { GraphEdge, GraphNode } from "../../api/intel";
import { Icon } from "../ui/Icon";
import { FlowGraph, NO_FILTERS, defaultLayout } from "./FlowGraph";
import type { FlowGraphHandle, GraphLayout } from "./FlowGraph";
import { NodeInspector } from "./NodeInspector";

export function FlowPreview({ nodes, edges, explorerHref, truncated, note, height = 440, title = "Money flow", hiddenKinds }: {
  nodes: GraphNode[]; edges: GraphEdge[]; explorerHref: string; truncated?: boolean; note?: React.ReactNode;
  height?: number; title?: string;
  /** Node kinds drawn hidden (e.g. the many single-address clusters of a run graph). */
  hiddenKinds?: string[];
}) {
  const filters = useMemo(() => ({ ...NO_FILTERS, hiddenKinds: new Set(hiddenKinds ?? []) }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [(hiddenKinds ?? []).join(",")]);
  const [layout, setLayout] = useState<GraphLayout>(() => defaultLayout(nodes.length));
  const [selected, setSelected] = useState<string | null>(null);
  const canvas = useRef<FlowGraphHandle>(null);
  const node = nodes.find((n) => n.id === selected) ?? null;
  const [full, setFull] = useState(false);
  useEffect(() => {
    if (!full) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setFull(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [full]);
  useEffect(() => { const t = window.setTimeout(() => canvas.current?.fit(), 120); return () => window.clearTimeout(t); }, [full]);
  const h = full ? "calc(100vh - 88px)" : height;

  return (
    <section className={`panel${full ? " panel-fullscreen" : ""}`} aria-label={title}>
      <div className="panel-head">
        <h2>{title}</h2>
        <span className="small faint">{nodes.length} nodes · {edges.length} edges</span>
        {truncated && <span className="chip" style={{ color: "var(--oc-warn)" }}>Truncated</span>}
        <span className="spacer" />
        <div className="btn-group" role="group" aria-label="Layout">
          <button type="button" className="btn btn-sm" aria-pressed={layout === "flow"} onClick={() => setLayout("flow")}>Flow</button>
          <button type="button" className="btn btn-sm" aria-pressed={layout === "explore"} onClick={() => setLayout("explore")}>Explore</button>
        </div>
        <button type="button" className="btn btn-sm btn-icon" aria-label="Fit graph to view" onClick={() => canvas.current?.fit()}><Icon name="fit" size={14} /></button>
        <button type="button" className="btn btn-sm btn-icon" aria-pressed={full} aria-label={full ? "Exit fullscreen (Esc)" : "Fullscreen"}
                title={full ? "Exit fullscreen (Esc)" : "Fullscreen"} onClick={() => setFull((v) => !v)}><Icon name="expand" size={14} /></button>
        <Link className="btn btn-sm" to={explorerHref}><Icon name="graph" size={14} />Open in explorer</Link>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: node ? "minmax(0,1fr) 300px" : "minmax(0,1fr)" }}>
        <div className="graph-host" style={{ height: h, borderRadius: 0 }}>
          <FlowGraph ref={canvas} nodes={nodes} edges={edges} layout={layout} filters={filters} selectedId={selected} onSelect={setSelected} />
        </div>
        {node && (
          <div style={{ borderLeft: "1px solid var(--oc-hairline)", maxHeight: h, overflowY: "auto" }}>
            <NodeInspector node={node} edges={edges} actions={{ chainIndex: !explorerHref.includes("run=") }} />
          </div>
        )}
      </div>
      {note && <div className="panel-foot">{note}</div>}
    </section>
  );
}
