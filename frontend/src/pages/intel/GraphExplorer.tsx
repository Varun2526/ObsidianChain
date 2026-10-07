/**
 * Graph explorer: trace money flow from any address, transaction or alert.
 *
 * Seeds and trace settings live in the URL, so a view can be shared and the
 * back button works. Expansions are incremental: each fetches one more hop
 * from the chain index and merges it into what is already on screen.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { getAlertGraph, getRunGraph, traceGraph } from "../../api/intel";
import type { AlertGraphResponse, RunGraphResponse, TraceDirection, TraceResponse } from "../../api/intel";
import { FlowGraph, NO_FILTERS, defaultLayout } from "../../components/graph/FlowGraph";
import type { FlowGraphHandle, GraphFilters, GraphLayout } from "../../components/graph/FlowGraph";
import { KIND_LABEL, countKinds, matchNodes, mergeGraph, timestepRange } from "../../components/graph/graphModel";
import type { GraphData } from "../../components/graph/graphModel";
import { NodeInspector } from "../../components/graph/NodeInspector";
import { WorkspaceResizer, usePanelLayout } from "../../components/graph/WorkspaceResizer";
import { ErrorState } from "../../components/ui/ErrorState";
import { Icon } from "../../components/ui/Icon";
import { EvidenceTag, short } from "../../components/ui/intel";

const KIND_SWATCH: Record<string, string> = {
  address: "var(--oc-surface-3)",
  transaction: "var(--oc-hairline-strong)",
  cluster: "transparent",
  ip: "var(--oc-ev-network)",
  asn: "var(--oc-ev-network)",
};

type Meta = {
  truncated: boolean;
  hubs: { txid: number; participants: number }[];
  hubThreshold: number;
  meaning: string;
  layers?: Record<string, string>;
  seedNote?: string;
  fingerprint?: string;
  run?: { caseId: string; runId: string; nodesTotal: number };
};

export function GraphExplorer() {
  const [params, setParams] = useSearchParams();
  const addresses = params.getAll("address");
  const txids = params.getAll("txid").map(Number).filter(Number.isFinite);
  const alertId = params.get("alert");
  // Run mode: the whole graph one uploaded-dataset run wrote, read in place.
  const caseId = params.get("case");
  const runId = params.get("run");
  const runMode = Boolean(caseId && runId);
  const focusId = params.get("focus");
  const direction = (params.get("direction") as TraceDirection) || "both";
  const hops = Number(params.get("hops") || (alertId ? 2 : 1));
  const maxNodes = Number(params.get("max") || 300);
  const minT = params.get("tmin") ? Number(params.get("tmin")) : null;
  const maxT = params.get("tmax") ? Number(params.get("tmax")) : null;
  const hasSeed = runMode || addresses.length > 0 || txids.length > 0 || Boolean(alertId);
  const seedKey = `${caseId}|${runId}|${alertId}|${addresses.join(",")}|${txids.join(",")}|${direction}|${hops}|${maxNodes}|${minT}|${maxT}`;

  const [graph, setGraph] = useState<GraphData>({ nodes: [], edges: [] });
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [expanding, setExpanding] = useState(false);
  const [layout, setLayout] = useState<GraphLayout>("flow");
  const [filters, setFilters] = useState<GraphFilters>(NO_FILTERS);
  const [hidden, setHidden] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<{ ids: string[]; label: string } | null>(null);
  const [pathStart, setPathStart] = useState<string | null>(null);
  const [find, setFind] = useState("");
  const [showList, setShowList] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [seedDraft, setSeedDraft] = useState("");
  const canvas = useRef<FlowGraphHandle>(null);
  const panels = usePanelLayout();

  const load = useCallback((signal?: AbortSignal) => {
    if (!hasSeed) { setGraph({ nodes: [], edges: [] }); setMeta(null); return; }
    setLoading(true); setError(null); setSelected(null); setHighlight(null); setPathStart(null);
    if (runMode) {
      getRunGraph(caseId!, runId!, signal).then((r: RunGraphResponse) => {
        setLayout(defaultLayout(r.graph.nodes.length));
        setGraph({ nodes: r.graph.nodes, edges: r.graph.edges });
        // Most run clusters are single addresses; drawn by default they bury
        // the flow. They stay one click away under Show.
        setFilters({ ...NO_FILTERS, hiddenKinds: new Set(["cluster"]) });
        setMeta({ truncated: r.truncated, hubs: [], hubThreshold: 0, meaning: r.meaning,
                  run: { caseId: caseId!, runId: runId!, nodesTotal: r.nodes_total } });
        setLoading(false);
        if (focusId && r.graph.nodes.some((n) => n.id === focusId)) {
          setSelected(focusId);
          window.setTimeout(() => canvas.current?.focus(focusId), 900);
        }
      }).catch((e: unknown) => {
        if ((e as Error)?.name === "AbortError") return;
        setError(e); setLoading(false);
      });
      return;
    }
    const req: Promise<TraceResponse | AlertGraphResponse> = alertId
      ? getAlertGraph(alertId, { hops, direction, maxNodes }, signal)
      : traceGraph({ addresses, txids, direction, hops, maxNodes, minTimestep: minT, maxTimestep: maxT }, signal);
    req.then((r) => {
      setLayout(defaultLayout(r.graph.nodes.length));
      setGraph({ nodes: r.graph.nodes, edges: r.graph.edges });
      const alertSeeds = "alert_id" in r ? r.seeds : null;
      setMeta({
        truncated: r.truncated, hubs: r.hub_transactions_skipped, hubThreshold: r.hub_threshold, meaning: r.meaning,
        layers: "layers" in r ? r.layers : undefined,
        seedNote: alertSeeds && alertSeeds.omitted > 0
          ? `Traced from the ${alertSeeds.used} highest-scored of ${alertSeeds.members_total} members.` : undefined,
        fingerprint: r.provenance?.chain_index_fingerprint,
      });
      setLoading(false);
    }).catch((e: unknown) => {
      if ((e as Error)?.name === "AbortError") return;
      setError(e); setLoading(false);
    });
    // seedKey captures every input
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seedKey]);

  useEffect(() => {
    const ctrl = new AbortController();
    load(ctrl.signal);
    return () => ctrl.abort();
  }, [load]);

  // This is an in-product workspace mode rather than browser fullscreen: it
  // keeps the inspector, keyboard escape hatch and responsive layout under
  // the application's control.
  useEffect(() => {
    if (!isFullscreen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsFullscreen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isFullscreen]);

  const update = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) { if (v == null || v === "") next.delete(k); else next.set(k, v); }
    setParams(next);
  };

  const addSeed = () => {
    const v = seedDraft.trim();
    if (!v) return;
    const next = new URLSearchParams(params);
    next.delete("alert");
    if (/^\d+$/.test(v)) next.append("txid", v);
    else if (/^[0-9a-f]{16}:\d+$/.test(v)) { next.delete("address"); next.delete("txid"); next.set("alert", v); }
    else next.append("address", v);
    setSeedDraft("");
    setParams(next);
  };

  const removeSeed = (kind: "address" | "txid", value: string) => {
    const next = new URLSearchParams();
    for (const [k, v] of params.entries()) if (!(k === kind && v === value)) next.append(k, v);
    setParams(next);
  };

  const expand = (id: string, dir: TraceDirection) => {
    const node = graph.nodes.find((n) => n.id === id);
    if (!node) return;
    setExpanding(true);
    const req = node.kind === "address"
      ? traceGraph({ addresses: [String(node.data.address)], direction: dir, hops: 1, maxNodes: 200 })
      : traceGraph({ txids: [Number(node.data.txid)], direction: dir, hops: 1, maxNodes: 200 });
    req.then((r) => {
      // an expansion seed is not a seed of the investigation
      const nodes = r.graph.nodes.map((n) => (n.id === id ? n : { ...n, data: { ...n.data, seed: false } }));
      setGraph((g) => mergeGraph(g, { nodes, edges: r.graph.edges }));
      setMeta((m) => m && ({ ...m, truncated: m.truncated || r.truncated, hubs: [...m.hubs, ...r.hub_transactions_skipped] }));
      setExpanding(false);
    }).catch((e: unknown) => { setError(e); setExpanding(false); });
  };

  const reach = (id: string, dir: "upstream" | "downstream") => {
    const ids = canvas.current?.reach(id, dir) ?? [];
    window.setTimeout(() => canvas.current?.fitTo(ids), 50);
    setHighlight({ ids, label: `${dir === "upstream" ? "Sources of" : "Destinations of"} ${labelOf(id)} in the loaded graph: ${ids.filter((x) => x.startsWith("addr:") && x !== id).length} addresses` });
  };

  const pathTo = (id: string) => {
    if (!pathStart) return;
    const r = canvas.current?.shortestPath(pathStart, id);
    if (!r) { setHighlight({ ids: [pathStart, id], label: runMode ? "No path between these two in this run's graph." : "No path between these two in the loaded graph. Expand further and try again." }); return; }
    const hopsN = r.ids.filter((x) => x.startsWith("tx:")).length;
    const kind = !r.flowOnly ? "Link path (not a money flow; goes through membership or relay links)"
      : r.directed ? "Money-flow path" : "Money-flow path, ignoring direction (no directed path exists)";
    setHighlight({ ids: r.ids, label: `${kind}: ${hopsN} transaction${hopsN === 1 ? "" : "s"}` });
    window.setTimeout(() => canvas.current?.fitTo(r.ids), 50);
  };

  const labelOf = (id: string) => {
    const n = graph.nodes.find((x) => x.id === id);
    return n ? (n.kind === "address" ? short(String(n.data.address)) : n.label) : id;
  };

  const kinds = useMemo(() => countKinds(graph.nodes), [graph.nodes]);
  const range = useMemo(() => timestepRange(graph.nodes), [graph.nodes]);
  const matches = useMemo(() => matchNodes(graph.nodes.filter((n) => !filters.hiddenKinds.has(n.kind)), find),
    [graph.nodes, find, filters.hiddenKinds]);
  const selectedNode = graph.nodes.find((n) => n.id === selected) ?? null;

  const toggleKind = (k: string) => setFilters((f) => {
    const s = new Set(f.hiddenKinds);
    if (s.has(k)) s.delete(k); else s.add(k);
    return { ...f, hiddenKinds: s };
  });

  return (
    <div className={`workspace${isFullscreen ? " workspace-fullscreen" : ""}${panels.state.leftHidden ? " ws-left-hidden" : ""}${panels.state.rightHidden ? " ws-right-hidden" : ""}`}
         style={panels.style} ref={panels.container} aria-busy={loading}>
      {/* ---- left: seeds, trace, filters ---- */}
      <aside className="ws-panel ws-panel-left" aria-label="Trace settings">
        <div className="ws-head"><Icon name="graph" /><h2>Graph explorer</h2></div>
        {runMode && (
          <section className="ws-section">
            <h3>Source</h3>
            <p className="small" style={{ margin: 0 }}>Uploaded-dataset run <span className="mono">{runId}</span>: every cluster, address, transaction, relay peer and ASN the pipeline projected from the capture.</p>
            <p className="note">Money flow: select a node, choose <strong>Set path start</strong>, select another, then <strong>Path to here</strong>. <strong>Show sources</strong> and <strong>Show destinations</strong> follow value edges only.</p>
            <Link className="btn btn-sm" to={`/inv/${caseId}`}><Icon name="arrowLeft" size={14} />Back to the investigation</Link>
          </section>
        )}
        {!runMode && <>
        <section className="ws-section">
          <h3>Seeds</h3>
          <form className="row" style={{ flexWrap: "nowrap" }} onSubmit={(e) => { e.preventDefault(); addSeed(); }}>
            <label className="sr-only" htmlFor="seed-input">Add an address, transaction id or alert id</label>
            <input id="seed-input" placeholder="Address, txid or alert id" value={seedDraft}
                   onChange={(e) => setSeedDraft(e.target.value)} />
            <button className="btn btn-sm" type="submit" aria-label="Add seed"><Icon name="plus" size={14} /></button>
          </form>
          <ul className="node-list" style={{ marginTop: 8 }}>
            {alertId && <li><span className="chip chip-mono">alert {alertId.split(":")[1]}</span> <button className="linkish small" type="button" onClick={() => update({ alert: null })}>remove</button></li>}
            {addresses.map((a) => (
              <li key={a} className="row" style={{ flexWrap: "nowrap" }}>
                <span className="mono small truncate" title={a}>{short(a, 10, 6)}</span>
                <button className="linkish small" type="button" onClick={() => removeSeed("address", a)} aria-label={`Remove seed ${a}`}>remove</button>
              </li>
            ))}
            {txids.map((t) => (
              <li key={t} className="row"><span className="mono small">tx {t}</span>
                <button className="linkish small" type="button" onClick={() => removeSeed("txid", String(t))} aria-label={`Remove seed ${t}`}>remove</button></li>
            ))}
          </ul>
          {!hasSeed && <p className="note">Add a seed, or open the explorer from an alert, an address or a transaction.</p>}
        </section>

        <section className="ws-section">
          <h3>Trace</h3>
          <div className="field">
            <span className="field-label" id="dir-label">Direction</span>
            <div className="btn-group" role="group" aria-labelledby="dir-label">
              {(["upstream", "both", "downstream"] as const).map((d) => (
                <button key={d} type="button" className="btn btn-sm" aria-pressed={direction === d} onClick={() => update({ direction: d })}>
                  {d === "both" ? "Both" : d === "upstream" ? "Upstream" : "Downstream"}
                </button>
              ))}
            </div>
          </div>
          <div className="inline-fields">
            <label className="field"><span className="field-label">Hops</span>
              <select value={hops} onChange={(e) => update({ hops: e.target.value })}>
                {[1, 2, 3, 4].map((h) => <option key={h} value={h}>{h}</option>)}
              </select>
            </label>
            <label className="field"><span className="field-label">Node cap</span>
              <select value={maxNodes} onChange={(e) => update({ max: e.target.value })}>
                {[150, 300, 600, 1000, 1500].map((h) => <option key={h} value={h}>{h}</option>)}
              </select>
            </label>
          </div>
          {!alertId && (
            <div className="inline-fields">
              <label className="field"><span className="field-label">From step</span>
                <input type="number" min={1} max={49} value={minT ?? ""} onChange={(e) => update({ tmin: e.target.value })} />
              </label>
              <label className="field"><span className="field-label">To step</span>
                <input type="number" min={1} max={49} value={maxT ?? ""} onChange={(e) => update({ tmax: e.target.value })} />
              </label>
            </div>
          )}
          <p className="note">Upstream follows value back to its sources; downstream follows it forward. Transactions with more than {meta?.hubThreshold ?? 200} participants are not crossed.</p>
        </section>
        </>}

        <section className="ws-section">
          <h3>Show</h3>
          {Object.entries(KIND_LABEL).map(([k, lab]) => (
            <label key={k} className="check-row">
              <input type="checkbox" checked={!filters.hiddenKinds.has(k)} onChange={() => toggleKind(k)} />
              <span className="swatch" style={{ background: KIND_SWATCH[k], border: k === "cluster" ? "1px dashed var(--oc-text-2)" : undefined }} />
              {lab}<span className="count">{kinds[k] ?? 0}</span>
            </label>
          ))}
          {!runMode && <>
          <label className="check-row">
            <input type="checkbox" checked={filters.onlyFlagged} onChange={(e) => setFilters((f) => ({ ...f, onlyFlagged: e.target.checked }))} />
            Only scored or attributed addresses
          </label>
          <label className="field" style={{ marginTop: 6 }}>
            <span className="field-label">Minimum model risk: {Math.round(filters.minRisk * 100)}%</span>
            <input type="range" min={0} max={1} step={0.05} value={filters.minRisk}
                   onChange={(e) => setFilters((f) => ({ ...f, minRisk: Number(e.target.value) }))} />
          </label>
          </>}
          {range && (
            <div className="inline-fields">
              <label className="field"><span className="field-label">Show from</span>
                <input type="number" min={range[0]} max={range[1]} placeholder={String(range[0])} value={filters.minTimestep ?? ""}
                       onChange={(e) => setFilters((f) => ({ ...f, minTimestep: e.target.value ? Number(e.target.value) : null }))} />
              </label>
              <label className="field"><span className="field-label">Show to</span>
                <input type="number" min={range[0]} max={range[1]} placeholder={String(range[1])} value={filters.maxTimestep ?? ""}
                       onChange={(e) => setFilters((f) => ({ ...f, maxTimestep: e.target.value ? Number(e.target.value) : null }))} />
              </label>
            </div>
          )}
          <button type="button" className="btn btn-sm btn-ghost" onClick={() => setFilters(NO_FILTERS)}>Clear filters</button>
        </section>

        <section className="ws-section">
          <h3>Legend</h3>
          <ul className="node-list small" style={{ display: "grid", gap: 4 }}>
            <li className="row"><span className="swatch" style={{ width: 12, height: 12, borderRadius: "50%", border: "2px solid var(--oc-accent)", background: "var(--oc-accent-wash)" }} /> Seed</li>
            <li className="row"><span className="swatch" style={{ width: 12, height: 12, borderRadius: "50%", border: "2px solid var(--oc-sev-critical)" }} /> Ring = model severity</li>
            <li className="row"><span className="swatch" style={{ width: 12, height: 12, borderRadius: "50%", boxShadow: "0 0 0 3px rgba(217,143,106,.4)" }} /> Halo = watchlist attribution</li>
            <li className="row"><span className="swatch" style={{ width: 8, height: 8, background: "var(--oc-hairline-strong)" }} /> Transaction</li>
            <li className="row"><span style={{ width: 16, borderTop: "1px solid #5a5f69" }} /> SPENDS / PAYS (value flow)</li>
            <li className="row"><span style={{ width: 16, borderTop: "1px dotted var(--oc-ev-network)" }} /> ANNOUNCED_BY (relay)</li>
            {runMode && <li className="row"><span style={{ width: 16, borderTop: "2px dashed var(--oc-ev-network)" }} /> SAME_FIRST_RELAY (hop first seen from one relay)</li>}
          </ul>
        </section>
      </aside>

      <WorkspaceResizer side="left" width={panels.state.left} hidden={panels.state.leftHidden}
                        onResize={panels.setWidth} onReset={panels.reset} onToggle={panels.toggle} />

      {/* ---- center: canvas ---- */}
      <section className="ws-center" aria-label="Graph">
        <div className="graph-host">
          <FlowGraph
            ref={canvas}
            nodes={graph.nodes}
            edges={graph.edges}
            layout={layout}
            filters={filters}
            selectedId={selected}
            highlightIds={highlight?.ids ?? null}
            onSelect={setSelected}
            onExpand={runMode ? undefined : (id) => expand(id, "both")}
            onHiddenCount={setHidden}
            hideEmptyMessage={!hasSeed}
          />
          {!hasSeed && (
            <div className="graph-empty">
              <div>
                <p className="strong">Start from something you know.</p>
                <p>Add an address, a transaction id or an alert id on the left, or search from the top bar.</p>
              </div>
            </div>
          )}
          <div className="graph-overlay-tl">
            <div className="graph-toolbar" role="group" aria-label="Layout">
              <button type="button" className="btn btn-sm" aria-pressed={layout === "flow"} onClick={() => setLayout("flow")} title="Directed flow, left to right">Flow</button>
              <button type="button" className="btn btn-sm" aria-pressed={layout === "explore"} onClick={() => setLayout("explore")} title="Force-directed">Explore</button>
            </div>
            <div className="graph-toolbar">
              <label className="sr-only" htmlFor="find-node">Find in graph</label>
              <input id="find-node" placeholder="Find in graph" value={find} onChange={(e) => setFind(e.target.value)}
                     style={{ border: 0, height: 28, width: 180, background: "transparent" }} />
            </div>
          </div>
          {matches.length > 0 && (
            <div className="graph-overlay-tl" style={{ top: 44 }}>
              <ul className="node-list panel" style={{ margin: 0, padding: 4, width: 280, background: "var(--oc-surface-2)" }}>
                {matches.map((n) => (
                  <li key={n.id}><button type="button" onClick={() => { setSelected(n.id); canvas.current?.focus(n.id); setFind(""); }}>
                    <span className="faint xsmall" style={{ width: 70 }}>{n.kind}</span><span className="mono truncate">{n.kind === "address" ? n.data.address : n.label}</span>
                  </button></li>
                ))}
              </ul>
            </div>
          )}
          <div className="graph-overlay-br">
            <div className="graph-toolbar" style={{ flexDirection: "column" }}>
              <button type="button" className="btn btn-sm btn-icon" aria-label={isFullscreen ? "Exit full-screen graph workspace" : "Open full-screen graph workspace"} aria-pressed={isFullscreen} onClick={() => setIsFullscreen((value) => !value)} title={isFullscreen ? "Exit full screen (Esc)" : "Full-screen workspace"}><Icon name={isFullscreen ? "close" : "expand"} size={14} /></button>
              <button type="button" className="btn btn-sm btn-icon" aria-label="Zoom in" onClick={() => canvas.current?.zoomBy(1.3)}><Icon name="plus" size={14} /></button>
              <button type="button" className="btn btn-sm btn-icon" aria-label="Zoom out" onClick={() => canvas.current?.zoomBy(1 / 1.3)}><Icon name="minus" size={14} /></button>
              <button type="button" className="btn btn-sm btn-icon" aria-label="Fit graph to view" onClick={() => canvas.current?.fit()}><Icon name="fit" size={14} /></button>
              <button type="button" className="btn btn-sm btn-icon" aria-label="Re-run layout" onClick={() => canvas.current?.relayout()}><Icon name="layers" size={14} /></button>
              <button type="button" className="btn btn-sm btn-icon" aria-label="Reset to the original trace" onClick={() => load()}><Icon name="reset" size={14} /></button>
            </div>
          </div>
          <div className="graph-overlay-bl" style={{ flexWrap: "wrap", maxWidth: "calc(100% - 60px)" }}>
            <span className="graph-status">{graph.nodes.length} nodes · {graph.edges.length} edges{hidden ? ` · ${hidden} hidden by filters` : ""}</span>
            {loading && <span className="graph-status">Tracing…</span>}
            {expanding && <span className="graph-status">Expanding…</span>}
            {meta?.truncated && <span className="graph-status warn">Node cap reached: the trace is incomplete. Raise the cap or narrow it.</span>}
            {meta?.hubs.length ? <span className="graph-status" title={meta.hubs.map((h) => `${h.txid} (${h.participants})`).join(", ")}>{meta.hubs.length} hub transaction{meta.hubs.length === 1 ? "" : "s"} not crossed</span> : null}
            {highlight && (
              <span className="graph-status" style={{ color: "var(--oc-accent)" }}>
                {highlight.label} <button type="button" className="linkish" onClick={() => { setHighlight(null); setPathStart(null); }}>clear</button>
              </span>
            )}
          </div>
        </div>
        {error != null && <div style={{ position: "absolute", inset: "auto 12px 48px 12px" }}><ErrorState error={error} onRetry={() => load()} /></div>}
      </section>

      <WorkspaceResizer side="right" width={panels.state.right} hidden={panels.state.rightHidden}
                        onResize={panels.setWidth} onReset={panels.reset} onToggle={panels.toggle} />

      {/* ---- right: inspector ---- */}
      <aside className="ws-panel ws-panel-right" aria-label="Inspector">
        <div className="ws-head">
          <h2>{selectedNode ? "Selected" : "Graph"}</h2>
          <span className="spacer" />
          <button type="button" className="btn btn-sm btn-ghost" aria-pressed={showList} onClick={() => setShowList((v) => !v)}>
            <Icon name="list" size={14} />Element list
          </button>
        </div>
        {showList ? (
          <ElementList graph={graph} onPick={(id) => { setSelected(id); canvas.current?.focus(id); setShowList(false); }} />
        ) : selectedNode ? (
          <NodeInspector node={selectedNode} edges={graph.edges} actions={{
            onExpand: runMode ? undefined : expand, onReach: reach, busy: expanding, chainIndex: !runMode,
            pathStart, onPathStart: (id) => setPathStart(id), onPathTo: pathTo,
          }} />
        ) : (
          <div>
            <div className="ws-section">
              <h3>How to read this</h3>
              <p className="small muted">{meta?.meaning ?? "Edges follow value: SPENDS means an address funded a transaction, PAYS means a transaction paid an address."}</p>
              <p className="small muted">Select a node to inspect it. Double-click an address or transaction to expand it one hop.</p>
            </div>
            <div className="ws-section">
              <h3>Evidence in this view</h3>
              {meta?.run ? (
              <ul className="node-list small" style={{ display: "grid", gap: 6 }}>
                <li><EvidenceTag kind="chain" /> Addresses, transactions and SPENDS / RECEIVES value edges from the uploaded capture.</li>
                <li><EvidenceTag kind="heuristic" /> Clusters: common-input ownership. Can merge owners.</li>
                <li><EvidenceTag kind="network" /> Relay peers and ASNs from the capture&apos;s own observations. A peer is never the sender.</li>
              </ul>
              ) : (
              <ul className="node-list small" style={{ display: "grid", gap: 6 }}>
                <li><EvidenceTag kind="chain" /> Addresses, transactions and flow edges from the Elliptic++ chain index.</li>
                <li><EvidenceTag kind="model" /> Severity rings: the reference model's risk band. A lead, not proof.</li>
                <li><EvidenceTag kind="watchlist" /> Halos: OFAC SDN or analyst watchlist attribution.</li>
                <li><EvidenceTag kind="heuristic" /> Clusters: common-input ownership. Can merge owners.</li>
                {meta?.layers && <li><EvidenceTag kind="network" /> Relay peers: synthetic network overlay. Never the sender.</li>}
              </ul>
              )}
            </div>
            {meta?.seedNote && <div className="ws-section"><p className="note">{meta.seedNote}</p></div>}
            {meta?.fingerprint && <div className="ws-section"><p className="note">Chain index <span className="mono">{meta.fingerprint}</span>. No class labels are served.</p></div>}
            {alertId && <div className="ws-section"><Link className="btn btn-sm" to={`/alerts/${encodeURIComponent(alertId)}`}><Icon name="arrowLeft" size={14} />Back to alert</Link></div>}
          </div>
        )}
      </aside>
    </div>
  );
}

/** The text alternative to the canvas: every node, sortable by kind, selectable by keyboard. */
function ElementList({ graph, onPick }: { graph: GraphData; onPick: (id: string) => void }) {
  const [kind, setKind] = useState<string>("address");
  const rows = graph.nodes.filter((n) => n.kind === kind);
  return (
    <div>
      <div className="ws-section">
        <div className="btn-group" role="group" aria-label="Element kind">
          {Object.entries(KIND_LABEL).map(([k, lab]) => (
            <button key={k} type="button" className="btn btn-sm" aria-pressed={kind === k} onClick={() => setKind(k)}>{lab}</button>
          ))}
        </div>
      </div>
      <table>
        <thead><tr><th>{KIND_LABEL[kind]}</th><th className="num">Hop</th><th>Notes</th></tr></thead>
        <tbody>
          {rows.slice(0, 400).map((n) => (
            <tr key={n.id}>
              <td><button type="button" className="linkish mono small" onClick={() => onPick(n.id)}>{n.kind === "address" ? short(String(n.data.address)) : n.label}</button></td>
              <td className="num">{n.data.hop ?? ""}</td>
              <td className="small">
                {n.data.seed ? "seed " : ""}
                {n.data.model ? `model ${n.data.model.severity.toLowerCase()} ` : ""}
                {n.data.watchlist?.length ? "watchlist " : ""}
                {n.kind === "transaction" && typeof n.data.timestep === "number" ? `t${n.data.timestep}` : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 400 && <p className="note" style={{ padding: 12 }}>First 400 of {rows.length} shown.</p>}
    </div>
  );
}
