/**
 * IP <-> transaction <-> wallet, the correlation the PS asks for.
 *
 * The one thing this panel must never imply is that a peer IP is a sender.
 * 84.3% of transactions in this dataset were announced by more than one
 * peer - that is gossip relay - so every row carries how many peers
 * announced it, and the caveat is stated above the table rather than buried
 * in a tooltip.
 *
 * GeoIP is reported as the registry actually resolves it. These addresses
 * are RFC 5737 documentation ranges, so no country exists for them and the
 * panel says exactly that instead of leaving an empty column.
 */
import { Fragment, useState } from "react";
import type { Correlation } from "../../api/types";
import { Address, CategoryChip, Value } from "../ui/primitives";

export function CorrelationPanel({
  correlation,
  onSelectTxid,
}: {
  correlation: Correlation;
  onSelectTxid?: (txid: string) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!correlation.available) {
    return (
      <section className="panel">
        <div className="panel-head">
          <h2>Network correlation</h2>
          <CategoryChip category="INSUFFICIENT_EVIDENCE" />
        </div>
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0 }}>
            <strong>Insufficient evidence.</strong>{" "}
            No announcement observations could be correlated to this alert's
            transactions. {correlation.insufficient_evidence_meaning}
          </p>
        </div>
      </section>
    );
  }

  const geo = correlation.summary?.geo;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Network correlation</h2>
        <CategoryChip category={correlation.category} />
        <span className="small muted">IP → transaction → wallet</span>
      </div>

      <div className="panel-body">
        <div className="banner banner-synthetic">
          <h4>An announcing peer is not a sender</h4>
          <p>{correlation.meaning}</p>
        </div>

        {correlation.summary ? (
          <div className="stat-row">
            <div className="stat">
              <span className="v">{correlation.summary.transactions}</span>
              <span className="k">Transactions correlated</span>
            </div>
            <div className="stat">
              <span className="v">{correlation.summary.announcing_peers}</span>
              <span className="k">Distinct announcing peers</span>
            </div>
            <div className="stat">
              <span className="v">{correlation.summary.asns}</span>
              <span className="k">ASNs</span>
            </div>
            <div className="stat">
              <span className="v">{correlation.summary.observers}</span>
              <span className="k">Observers</span>
            </div>
            <div className="stat">
              <span className="v">
                {geo?.countries.length
                  ? geo.countries.join(", ")
                  : <span className="faint" style={{ fontSize: 15 }}>n/a</span>}
              </span>
              <span className="k">Countries (GeoIP)</span>
            </div>
          </div>
        ) : null}

        {geo ? (
          <p className="note">
            <strong>GeoIP / ASN.</strong> {geo.country_note}
            {geo.reserved_ranges.length ? (
              <> Observed ranges: {geo.reserved_ranges.join("; ")}.</>
            ) : null}
            {geo.all_private_asns ? (
              <> All observed ASNs are RFC 6996 private-use numbers, which
              identify no real network operator.</>
            ) : null}
            {geo.geoip_database_installed
              ? " Countries were resolved from the installed GeoIP database."
              : " No GeoIP database is installed; countries are reported as unavailable rather than guessed."}
          </p>
        ) : null}
      </div>

      <div className="panel-body flush" style={{ maxHeight: 420, overflowY: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Transaction</th>
              <th className="num">Announcing peers</th>
              <th className="num">Wallets</th>
              <th>First seen</th>
              <th>Drilldown</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {correlation.transactions.map((tx) => (
              <Fragment key={tx.txid}>
                <tr className="clickable"
                  onClick={() => setExpanded(expanded === tx.txid ? null : tx.txid)}>
                  <td className="mono">{tx.txid}</td>
                  <td className="num">
                    {tx.announcing_peers_total}
                    {tx.peers_shown < tx.announcing_peers_total ? (
                      <span className="faint"> ({tx.peers_shown} shown)</span>
                    ) : null}
                  </td>
                  <td className="num">{tx.addresses.length}</td>
                  <td className="mono small">
                    {tx.first_seen_ms === null
                      ? <span className="faint">n/a</span>
                      : new Date(tx.first_seen_ms).toISOString().replace("T", " ").slice(0, 19)}
                  </td>
                  <td onClick={(e) => e.stopPropagation()}>
                    {onSelectTxid && (
                      <button
                        className="btn btn-sm"
                        style={{ padding: "2px 8px", fontSize: 11 }}
                        onClick={() => onSelectTxid(tx.txid)}
                      >
                        Drill-down
                      </button>
                    )}
                  </td>
                  <td className="faint small">{expanded === tx.txid ? "▾" : "▸"}</td>
                </tr>
                {expanded === tx.txid ? (
                  <tr key={`${tx.txid}-detail`}>
                    <td colSpan={6} style={{ background: "var(--bg)" }}>
                      <div className="tx-detail">
                        <div>
                          <h4 className="small muted">Announcing peers (relay vantage points)</h4>
                          <table>
                            <thead>
                              <tr><th>IP</th><th className="num">Port</th>
                                <th className="num">ASN</th><th className="num">Observers</th></tr>
                            </thead>
                            <tbody>
                              {tx.peers.map((p) => (
                                <tr key={p.ip}>
                                  <td className="mono">{p.ip}</td>
                                  <td className="num"><Value value={p.port} digits={0} /></td>
                                  <td className="num"><Value value={p.asn} digits={0} /></td>
                                  <td className="num">{p.observers}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <div>
                          <h4 className="small muted">Wallets in this transaction</h4>
                          <table>
                            <thead><tr><th>Address</th><th>Role</th></tr></thead>
                            <tbody>
                              {tx.addresses.map((a) => (
                                <tr key={`${a.address}-${a.role}`}>
                                  <td><Address value={a.address} /></td>
                                  <td className="small muted">{a.role}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      {onSelectTxid && (
                        <div style={{ marginTop: 12, padding: "8px 12px", borderTop: "1px solid var(--color-border, #e1e4e8)", display: "flex", justifyContent: "flex-end" }}>
                          <button
                            className="btn btn-sm btn-primary"
                            onClick={() => onSelectTxid(tx.txid)}
                          >
                            Open Full Transaction Drill-down (Inputs, Outputs & Mixing) →
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
