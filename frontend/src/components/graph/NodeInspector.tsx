/**
 * What is known about the selected element, grouped by where it comes from.
 * On-chain facts first; attribution and model output after, each labelled.
 */
import { Link } from "react-router-dom";

import type { GraphEdge, GraphNode } from "../../api/intel";
import { Icon } from "../ui/Icon";
import { AnnotationChips, CopyButton, EvidenceTag, SevTag, btc, int, pct } from "../ui/intel";
import { neighbours } from "./graphModel";

export interface InspectorActions {
  onExpand?: (id: string, direction: "upstream" | "downstream" | "both") => void;
  onReach?: (id: string, direction: "upstream" | "downstream") => void;
  onPathStart?: (id: string) => void;
  onPathTo?: (id: string) => void;
  pathStart?: string | null;
  busy?: boolean;
}

export function NodeInspector({ node, edges, actions }: { node: GraphNode; edges: GraphEdge[]; actions: InspectorActions }) {
  const d = node.data;
  const { incoming, outgoing } = neighbours(node.id, edges);
  const flowIn = incoming.filter((e) => e.kind === "PAYS" || e.kind === "SPENDS").length;
  const flowOut = outgoing.filter((e) => e.kind === "PAYS" || e.kind === "SPENDS").length;
  const canExpand = node.kind === "address" || node.kind === "transaction";

  return (
    <div>
      <div className="ws-section">
        <div className="eyebrow">{node.kind === "ip" ? "relay peer" : node.kind}{d.seed ? " · seed" : ""}{typeof d.hop === "number" ? ` · hop ${d.hop}` : ""}</div>
        <div className="inspector-title">
          <span className="mono">{node.kind === "address" ? d.address : node.kind === "transaction" ? d.txid : node.label}</span>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          {node.kind === "address" && d.address && (
            <>
              <Link className="btn btn-sm" to={`/entity/${encodeURIComponent(d.address)}`}><Icon name="external" size={14} />Profile</Link>
              <CopyButton value={d.address} />
            </>
          )}
          {node.kind === "transaction" && d.txid != null && (
            <Link className="btn btn-sm" to={`/tx/${d.txid}`}><Icon name="external" size={14} />Transaction</Link>
          )}
          {node.kind === "cluster" && typeof d.alert_id === "string" && (
            <Link className="btn btn-sm" to={`/alerts/${encodeURIComponent(d.alert_id)}`}><Icon name="external" size={14} />Alert</Link>
          )}
        </div>
      </div>

      <div className="ws-section">
        <h3><EvidenceTag kind={node.kind === "ip" ? "network" : node.kind === "cluster" ? "heuristic" : "chain"} /> Observed</h3>
        <dl className="kv">
          {node.kind === "transaction" && (
            <>
              <dt>Timestep</dt><dd className="num">{d.timestep ?? "n/a"}</dd>
              <dt>Inputs · outputs</dt><dd className="num">{int(d.n_inputs as number)} · {int(d.n_outputs as number)}</dd>
              <dt>Value in</dt><dd className="num">{btc(d.in_btc as number)}</dd>
              <dt>Value out</dt><dd className="num">{btc(d.out_btc as number)}</dd>
              <dt>Fee</dt><dd className="num">{btc(d.fee_btc as number)}</dd>
            </>
          )}
          {node.kind === "cluster" && (
            <>
              <dt>Cluster</dt><dd className="num">{String(d.cluster_id)}</dd>
              <dt>Members</dt><dd className="num">{int(d.members_total as number)}</dd>
            </>
          )}
          {node.kind === "ip" && (
            <>
              <dt>ASN</dt><dd className="num">{d.asn == null ? "n/a" : String(d.asn)}</dd>
              <dt>Meaning</dt><dd>Relayed a transaction to an observer. Not the sender.</dd>
            </>
          )}
          <dt>Flow edges in loaded graph</dt><dd className="num">{flowIn} in · {flowOut} out</dd>
        </dl>
      </div>

      {node.kind === "address" && (
        <div className="ws-section">
          <h3>Attribution and model</h3>
          <AnnotationChips model={d.model} cluster={d.cluster} watchlist={d.watchlist} />
          {d.model && (
            <p className="note" style={{ marginTop: 8 }}>
              Model risk {pct(d.model.risk_score)} from the Phase 7 reference run
              (alert <Link to={`/alerts/${encodeURIComponent(d.model.alert_id)}`}>{d.model.alert_id.split(":")[1]}</Link>).
              A learned association, not proof of illicit activity.
            </p>
          )}
        </div>
      )}

      {node.kind === "cluster" && typeof d.severity === "string" && (
        <div className="ws-section">
          <h3>Alert</h3>
          <div className="row"><EvidenceTag kind="model">Model {pct(d.risk_score as number)}</EvidenceTag><SevTag severity={d.severity} /></div>
        </div>
      )}

      {(canExpand || actions.onPathStart) && (
        <div className="ws-section">
          <h3>Trace</h3>
          {canExpand && actions.onExpand && (
            <div className="action-list" style={{ marginBottom: 6 }}>
              <button type="button" className="btn btn-sm" disabled={actions.busy} onClick={() => actions.onExpand?.(node.id, "upstream")}>
                <Icon name="upstream" size={14} />Expand upstream
              </button>
              <button type="button" className="btn btn-sm" disabled={actions.busy} onClick={() => actions.onExpand?.(node.id, "downstream")}>
                <Icon name="downstream" size={14} />Expand downstream
              </button>
            </div>
          )}
          {actions.onReach && (
            <div className="action-list" style={{ marginBottom: 6 }}>
              <button type="button" className="btn btn-sm" onClick={() => actions.onReach?.(node.id, "upstream")}>Show sources</button>
              <button type="button" className="btn btn-sm" onClick={() => actions.onReach?.(node.id, "downstream")}>Show destinations</button>
            </div>
          )}
          {actions.onPathStart && (
            <div className="action-list">
              <button type="button" className="btn btn-sm" onClick={() => actions.onPathStart?.(node.id)}
                      aria-pressed={actions.pathStart === node.id}>
                <Icon name="path" size={14} />{actions.pathStart === node.id ? "Path start set" : "Set path start"}
              </button>
              <button type="button" className="btn btn-sm" disabled={!actions.pathStart || actions.pathStart === node.id}
                      onClick={() => actions.onPathTo?.(node.id)}>
                Path to here
              </button>
            </div>
          )}
          <p className="note" style={{ marginTop: 8 }}>
            Sources and destinations are computed over the loaded graph only. Expanding fetches one more hop from the chain index.
          </p>
        </div>
      )}
    </div>
  );
}
