/**
 * M0-M3 evidence, grouped exactly as the backend groups it.
 *
 * Each group carries the backend's own category, so an investigator can see
 * at a glance that M0-M2 are statements about the ledger and M3 is network
 * context. Features the backend could not compute are listed as unavailable
 * rather than rendered as zero.
 */
import { useState } from "react";
import type { Evidence, FeatureGroup } from "../api/types";
import { Address, CategoryChip, Value } from "./primitives";

const GROUP_LABEL: Record<FeatureGroup, string> = {
  M0: "M0 · address behaviour",
  M1: "M1 · graph",
  M2: "M2 · PEEL-1 chain structure",
  M3: "M3 · network",
};

export function EvidencePanel({ evidence }: { evidence: Evidence }) {
  const groups = Object.keys(evidence.groups) as FeatureGroup[];
  const [active, setActive] = useState<FeatureGroup>(groups[0] ?? "M0");
  const group = evidence.groups[active];

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Evidence</h2>
        <span className="small muted">aggregate</span>
      </div>
      <div className="panel-body">
        <div className="stat-row">
          <div className="stat">
            <span className="v"><Value value={evidence.aggregate.transactions} digits={0} /></span>
            <span className="k">Transactions</span>
          </div>
          <div className="stat">
            <span className="v"><Value value={evidence.aggregate.btc_sent_total} digits={4} /></span>
            <span className="k">BTC sent</span>
          </div>
          <div className="stat">
            <span className="v"><Value value={evidence.aggregate.btc_received_total} digits={4} /></span>
            <span className="k">BTC received</span>
          </div>
          <div className="stat">
            <span className="v"><Value value={evidence.aggregate.unique_counterparties} digits={0} /></span>
            <span className="k">Counterparties</span>
          </div>
          <div className="stat">
            <span className="v">{evidence.aggregate.peel_chain_members}</span>
            <span className="k">In PEEL-1 chain</span>
          </div>
        </div>
      </div>

      <div className="panel-head" style={{ borderTop: "1px solid var(--border)" }}>
        <div className="toggles" role="group" aria-label="Feature group">
          {groups.map((g) => (
            <button key={g} className="toggle" aria-pressed={active === g}
              onClick={() => setActive(g)}>{g}</button>
          ))}
        </div>
        {group ? (
          <>
            <span className="small muted">{GROUP_LABEL[active]}</span>
            <CategoryChip category={group.category} />
          </>
        ) : null}
      </div>

      {group ? (
        <div className="panel-body flush" style={{ overflowX: "auto", maxHeight: 400 }}>
          <table>
            <thead>
              <tr>
                <th>Address</th>
                {group.features.map((f) => (
                  <th key={f} className="num" title={f}>
                    {f.replace(/_asof_t$/, "").replace(/^net_/, "")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {group.members.map((m) => (
                <tr key={m.address}>
                  <td><Address value={m.address} /></td>
                  {group.features.map((f) => (
                    <td key={f} className="num">
                      <Value value={m.values[f] ?? null} digits={3} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className="panel-body">
        <p className="note" style={{ marginTop: 0 }}>
          {evidence.insufficient_evidence_meaning}
        </p>
      </div>
    </section>
  );
}
