/**
 * Activity over time, in Elliptic timesteps.
 *
 * The as-of nature is stated rather than implied: every feature in this
 * system is computed as of an address's last active timestep, so a timeline
 * point is what was observable BY that timestep, not a live feed. An
 * investigator reading a rising line needs to know it cannot include
 * anything after the observation point.
 */
import type { Timeline as TimelineData } from "../../api/types";
import { CategoryChip, Value } from "../ui/primitives";

export function Timeline({ timeline, observedAt }:
  { timeline: TimelineData; observedAt: number }) {
  const points = timeline.points;
  const peak = Math.max(...points.map((p) => p.transactions), 1);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Activity timeline</h2>
        <CategoryChip category={timeline.category} />
        <span className="small muted">{timeline.unit}</span>
      </div>

      {points.length === 0 ? (
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0 }}>
            No activity points were returned for this alert.
          </p>
        </div>
      ) : (
        <div className="panel-body flush" style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Timestep</th>
                <th className="num">Transactions</th>
                <th className="num">Active addresses</th>
                <th className="num">BTC sent</th>
                <th className="num">BTC received</th>
                <th style={{ width: "34%" }}>Relative volume</th>
              </tr>
            </thead>
            <tbody>
              {points.map((p) => (
                <tr key={p.timestep}>
                  <td className="mono">
                    t{p.timestep}
                    {p.timestep === observedAt ? (
                      <span className="faint small" title="Observation point: no feature may use anything after this timestep">
                        {" "}· as-of
                      </span>
                    ) : null}
                  </td>
                  <td className="num">{p.transactions.toLocaleString()}</td>
                  <td className="num">{p.active_addresses.toLocaleString()}</td>
                  <td className="num"><Value value={p.btc_sent} digits={4} /></td>
                  <td className="num"><Value value={p.btc_received} digits={4} /></td>
                  <td>
                    <span className="riskbar" style={{ width: "100%" }}>
                      <span style={{ width: `${(p.transactions / peak) * 100}%`,
                                     background: "var(--blockchain)" }} />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="panel-body">
        <p className="note" style={{ marginTop: 0 }}>
          Each Elliptic timestep covers roughly two weeks. Every feature on
          this alert is computed <strong>as of timestep {observedAt}</strong>,
          using only transactions at or before it — nothing later is visible
          to the model, by construction.
        </p>
      </div>
    </section>
  );
}
