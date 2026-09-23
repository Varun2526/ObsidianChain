/**
 * A bounded money-flow view embedded in a detail page, with a way out to
 * the full explorer. Same canvas, same data, fewer controls.
 */
import { useRef, useState } from "react";
import { Link } from "react-router-dom";

import type { GraphEdge, GraphNode } from "../../api/intel";
import { Icon } from "../ui/Icon";
import { FlowGraph, defaultLayout } from "./FlowGraph";
import type { FlowGraphHandle, GraphLayout } from "./FlowGraph";
import { NodeInspector } from "./NodeInspector";

export function FlowPreview({ nodes, edges, explorerHref, truncated, note, height = 440, title = "Money flow" }: {
  nodes: GraphNode[]; edges: GraphEdge[]; explorerHref: string; truncated?: boolean; note?: React.ReactNode;
  height?: number; title?: string;
}) {
  const [layout, setLayout] = useState<GraphLayout>(() => defaultLayout(nodes.length));
  const [selected, setSelected] = useState<string | null>(null);
  const canvas = useRef<FlowGraphHandle>(null);
  const node = nodes.find((n) => n.id === selected) ?? null;

  return (
    <section className="panel">
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
        <Link className="btn btn-sm" to={explorerHref}><Icon name="graph" size={14} />Open in explorer</Link>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: node ? "minmax(0,1fr) 300px" : "minmax(0,1fr)" }}>
        <div className="graph-host" style={{ height, borderRadius: 0 }}>
          <FlowGraph ref={canvas} nodes={nodes} edges={edges} layout={layout} selectedId={selected} onSelect={setSelected} />
        </div>
        {node && (
          <div style={{ borderLeft: "1px solid var(--oc-hairline)", maxHeight: height, overflowY: "auto" }}>
            <NodeInspector node={node} edges={edges} actions={{}} />
          </div>
        )}
      </div>
      {note && <div className="panel-foot">{note}</div>}
    </section>
  );
}
