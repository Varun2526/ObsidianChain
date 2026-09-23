/**
 * Advanced investigative modals (T2):
 * - CrossAlertModal (T2.2)
 * - ClusterCompareModal (T2.3 & T2.8)
 * - TransactionModal (T2.4)
 */
import { useEffect, useState } from "react";
import { fetchAlert } from "../../api/client";
import * as api from "../../api/console";
import type { AlertDetail, AlertPatterns, TransactionDrilldown } from "../../api/types";
import { RiskBar, SeverityBadge } from "../ui/primitives";

export function TransactionModal({
  txid,
  onClose,
}: {
  txid: number | string | null;
  onClose: () => void;
}) {
  const [data, setData] = useState<TransactionDrilldown | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (!txid) return;
    setLoading(true);
    setError(null);
    api.getTransactionDrilldown(txid)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [txid]);

  if (!txid) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Transaction Drill-Down — {txid}</h3>
          <button type="button" className="btn btn-sm btn-ghost" aria-label="Close" onClick={onClose}>Close</button>
        </div>
        <div className="modal-body">
          {loading ? (
            <p className="muted">Loading transaction metadata…</p>
          ) : error ? (
            <p className="note">Could not load details for transaction {txid}.</p>
          ) : data ? (
            <>
              <div className="stat-row" style={{ marginBottom: 16 }}>
                <div className="stat">
                  <span className="v mono">{data.txid}</span>
                  <span className="k">TXID</span>
                </div>
                <div className="stat">
                  <span className="v">{data.input_count}</span>
                  <span className="k">Inputs</span>
                </div>
                <div className="stat">
                  <span className="v">{data.output_count}</span>
                  <span className="k">Outputs</span>
                </div>
                <div className="stat">
                  <span className="v">{data.announcing_peers.length}</span>
                  <span className="k">Announcing peers</span>
                </div>
                <div className="stat">
                  <span className="v mono">
                    {data.mixing?.mixing_class ?? (data.mixing?.available ? "STANDARD" : "n/a")}
                  </span>
                  <span className="k">Mixing classification</span>
                </div>
              </div>

              <div className="banner banner-synthetic">
                <h4>Evidentiary Limitation</h4>
                <p>{data.limitation}</p>
              </div>

              <div className="compare-grid" style={{ marginTop: 16 }}>
                <div className="compare-col">
                  <h4 style={{ margin: "0 0 8px" }}>Inputs ({data.inputs.length})</h4>
                  <ul className="note-list">
                    {data.inputs.map((inp, i) => (
                      <li key={i}>
                        <span className="mono small">{inp.address}</span>
                        <span className="faint small" style={{ display: "block" }}>
                          Alert: {inp.alert_id}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="compare-col">
                  <h4 style={{ margin: "0 0 8px" }}>Outputs ({data.outputs.length})</h4>
                  <ul className="note-list">
                    {data.outputs.map((out, i) => (
                      <li key={i}>
                        <span className="mono small">{out.address}</span>
                        <span className="faint small" style={{ display: "block" }}>
                          Alert: {out.alert_id}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {data.announcing_peers.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h4 style={{ margin: "0 0 8px" }}>Announcing Peers (Relay Vantage Points)</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Peer IP</th><th>Port</th><th>ASN</th><th>Observers</th><th>Announcing Peers</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.announcing_peers.map((p, i) => (
                        <tr key={i}>
                          <td className="mono">{p.ip}</td>
                          <td>{p.port ?? "—"}</td>
                          <td className="mono">{p.asn ? `AS${p.asn}` : "—"}</td>
                          <td>{p.observers}</td>
                          <td>{p.announcing_peers}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          ) : null}
        </div>
        <div className="modal-foot">
          <button className="btn btn-sm" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

export function CrossAlertModal({
  alertIds,
  onClose,
}: {
  alertIds: string[];
  onClose: () => void;
}) {
  const [alerts, setAlerts] = useState<AlertDetail[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all(alertIds.map((id) => fetchAlert(id).catch(() => null)))
      .then((results) => setAlerts(results.filter((a): a is AlertDetail => a !== null)))
      .finally(() => setLoading(false));
  }, [alertIds]);

  // Compute common entities, common txids, common peers
  const txOccurrences = new Map<string, Set<string>>();
  const peerOccurrences = new Map<string, Set<string>>();
  const addressOccurrences = new Map<string, Set<string>>();

  alerts.forEach((a) => {
    a.correlation.transactions.forEach((tx) => {
      if (!txOccurrences.has(tx.txid)) txOccurrences.set(tx.txid, new Set());
      txOccurrences.get(tx.txid)!.add(a.alert_id);
    });
    a.correlation.transactions.forEach((tx) => {
      tx.peers.forEach((p) => {
        if (!peerOccurrences.has(p.ip)) peerOccurrences.set(p.ip, new Set());
        peerOccurrences.get(p.ip)!.add(a.alert_id);
      });
    });
    a.members.rows.forEach((m) => {
      if (!addressOccurrences.has(m.address)) addressOccurrences.set(m.address, new Set());
      addressOccurrences.get(m.address)!.add(a.alert_id);
    });
  });

  const commonTxs = Array.from(txOccurrences.entries()).filter(([_, set]) => set.size > 1);
  const commonPeers = Array.from(peerOccurrences.entries()).filter(([_, set]) => set.size > 1);
  const commonAddrs = Array.from(addressOccurrences.entries()).filter(([_, set]) => set.size > 1);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Cross-Alert Investigation ({alerts.length} Selected)</h3>
          <button type="button" className="btn btn-sm btn-ghost" aria-label="Close" onClick={onClose}>Close</button>
        </div>
        <div className="modal-body">
          <div className="banner banner-synthetic" style={{ marginTop: 0 }}>
            <h4>Analysis Assistance — Not an Automatic Conclusion</h4>
            <p>
              Shared transactions demonstrate counterparty fund flow. Shared announcing peers indicate propagation through the same gossip vantage point. Neither establishes that different clusters belong to the same person or entity.
            </p>
          </div>

          {loading ? (
            <p className="muted">Loading alert details…</p>
          ) : (
            <>
              <h4 style={{ margin: "14px 0 8px" }}>Selected Alerts</h4>
              <table>
                <thead>
                  <tr>
                    <th>Alert</th><th>Cluster</th><th>Severity</th><th>Risk</th><th>Members</th><th>Top Signals</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((a) => (
                    <tr key={a.alert_id}>
                      <td className="mono small">{a.alert_id}</td>
                      <td className="mono">{a.summary.cluster_id}</td>
                      <td><SeverityBadge severity={a.summary.severity} /></td>
                      <td className="mono">{a.risk.score?.toFixed(4) ?? "—"}</td>
                      <td>{a.summary.members_scored}</td>
                      <td className="small muted">{Object.keys(a.evidence.groups).slice(0, 2).join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="card-grid-4" style={{ marginTop: 16 }}>
                <div className="stat-card">
                  <span className="stat-card-value">{commonTxs.length}</span>
                  <span className="stat-card-label">Shared transactions</span>
                </div>
                <div className="stat-card">
                  <span className="stat-card-value">{commonPeers.length}</span>
                  <span className="stat-card-label">Shared announcing peers</span>
                </div>
                <div className="stat-card">
                  <span className="stat-card-value">{commonAddrs.length}</span>
                  <span className="stat-card-label">Shared member addresses</span>
                </div>
                <div className="stat-card">
                  <span className="stat-card-value">{alerts.length}</span>
                  <span className="stat-card-label">Alerts in group</span>
                </div>
              </div>

              {commonTxs.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h4>Common Transactions Connecting These Alerts</h4>
                  <table>
                    <thead>
                      <tr><th>TXID</th><th>Alerts Connected</th></tr>
                    </thead>
                    <tbody>
                      {commonTxs.map(([txid, alertSet]) => (
                        <tr key={txid}>
                          <td className="mono">{txid}</td>
                          <td className="small muted">{Array.from(alertSet).join(", ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {commonPeers.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h4>Shared Announcing Peers (Gossip Relay)</h4>
                  <table>
                    <thead>
                      <tr><th>Peer IP</th><th>Alerts Reached</th></tr>
                    </thead>
                    <tbody>
                      {commonPeers.map(([peer, alertSet]) => (
                        <tr key={peer}>
                          <td className="mono">{peer}</td>
                          <td className="small muted">{Array.from(alertSet).join(", ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
        <div className="modal-foot">
          <button className="btn btn-sm" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

export function ClusterCompareModal({
  alertAId,
  alertBId,
  onClose,
}: {
  alertAId: string;
  alertBId: string;
  onClose: () => void;
}) {
  const [a, setA] = useState<AlertDetail | null>(null);
  const [b, setB] = useState<AlertDetail | null>(null);
  const [patA, setPatA] = useState<AlertPatterns | null>(null);
  const [patB, setPatB] = useState<AlertPatterns | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetchAlert(alertAId).catch(() => null),
      fetchAlert(alertBId).catch(() => null),
      api.getAlertPatterns(alertAId).catch(() => null),
      api.getAlertPatterns(alertBId).catch(() => null),
    ]).then(([resA, resB, pA, pB]) => {
      setA(resA);
      setB(resB);
      setPatA(pA);
      setPatB(pB);
      setLoading(false);
    });
  }, [alertAId, alertBId]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Entity / Cluster Comparison</h3>
          <button type="button" className="btn btn-sm btn-ghost" aria-label="Close" onClick={onClose}>Close</button>
        </div>
        <div className="modal-body">
          <div className="banner banner-synthetic" style={{ marginTop: 0 }}>
            <h4>Objective Comparative Metric Display</h4>
            <p>
              This view presents empirical differences across blockchain, network, and model evidence. It does not declare that one entity is more or less criminal than another.
            </p>
          </div>

          {loading || !a || !b ? (
            <p className="muted">Loading comparison data…</p>
          ) : (
            <div className="compare-grid" style={{ marginTop: 16 }}>
              <div className="compare-col">
                <h3>Cluster {a.summary.cluster_id}</h3>
                <span className="mono small faint">{a.alert_id}</span>
                <div style={{ margin: "10px 0" }}>
                  <SeverityBadge severity={a.summary.severity} />
                  {" "}
                  <RiskBar value={a.risk.score} severity={a.risk.severity} />
                  {" "}
                  <span className="mono">{a.risk.score?.toFixed(4) ?? "—"}</span>
                </div>

                <dl className="kv" style={{ marginTop: 12 }}>
                  <dt>Scored members</dt><dd>{a.summary.members_scored.toLocaleString()}</dd>
                  <dt>Total in cluster</dt><dd>{a.summary.members_total.toLocaleString()}</dd>
                  <dt>Timestep window</dt><dd>t{a.summary.first_timestep}–t{a.summary.last_timestep}</dd>
                  <dt>Correlated txs</dt><dd>{a.correlation.transactions.length}</dd>
                  <dt>Announcing peers</dt><dd>{a.correlation.summary?.announcing_peers ?? 0}</dd>
                  <dt>Peeling chain</dt>
                  <dd>{patA?.peeling.available ? `max depth ${patA.peeling.max_chain_depth}` : "not present"}</dd>
                  <dt>Mixing class</dt>
                  <dd>{patA?.mixing.available ? `${patA.mixing.pattern_count} patterns` : "not scanned"}</dd>
                </dl>
              </div>

              <div className="compare-col">
                <h3>Cluster {b.summary.cluster_id}</h3>
                <span className="mono small faint">{b.alert_id}</span>
                <div style={{ margin: "10px 0" }}>
                  <SeverityBadge severity={b.summary.severity} />
                  {" "}
                  <RiskBar value={b.risk.score} severity={b.risk.severity} />
                  {" "}
                  <span className="mono">{b.risk.score?.toFixed(4) ?? "—"}</span>
                </div>

                <dl className="kv" style={{ marginTop: 12 }}>
                  <dt>Scored members</dt><dd>{b.summary.members_scored.toLocaleString()}</dd>
                  <dt>Total in cluster</dt><dd>{b.summary.members_total.toLocaleString()}</dd>
                  <dt>Timestep window</dt><dd>t{b.summary.first_timestep}–t{b.summary.last_timestep}</dd>
                  <dt>Correlated txs</dt><dd>{b.correlation.transactions.length}</dd>
                  <dt>Announcing peers</dt><dd>{b.correlation.summary?.announcing_peers ?? 0}</dd>
                  <dt>Peeling chain</dt>
                  <dd>{patB?.peeling.available ? `max depth ${patB.peeling.max_chain_depth}` : "not present"}</dd>
                  <dt>Mixing class</dt>
                  <dd>{patB?.mixing.available ? `${patB.mixing.pattern_count} patterns` : "not scanned"}</dd>
                </dl>
              </div>
            </div>
          )}
        </div>
        <div className="modal-foot">
          <button className="btn btn-sm" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
