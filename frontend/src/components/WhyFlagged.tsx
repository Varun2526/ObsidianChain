/**
 * "Why was this flagged?" - the model's own SHAP output, not a paraphrase.
 *
 * Two things are kept separate on purpose, because conflating them is how a
 * ranking signal turns into an accusation:
 *
 *   the CONTRIBUTION is a MODEL_SIGNAL - it describes the model's behaviour;
 *   the feature's VALUE is BLOCKCHAIN_CONTEXT or NETWORK_CONTEXT, or
 *   INSUFFICIENT_EVIDENCE when it was never computable.
 *
 * Contributions are log-odds. The panel says so rather than letting a reader
 * take +3.32 for 332 percentage points of anything.
 */
import { useState } from "react";
import type { WhyFlagged as WhyFlaggedData } from "../api/types";
import { CategoryChip, Value } from "./primitives";

export function WhyFlagged({ data }: { data: WhyFlaggedData }) {
  const [index, setIndex] = useState(0);
  const member = data.per_member[index];

  if (!member) {
    return (
      <section className="panel">
        <div className="panel-head"><h2>Why was this flagged?</h2></div>
        <div className="panel-body">
          <p className="note">No per-member explanation was returned for this alert.</p>
        </div>
      </section>
    );
  }

  const widest = Math.max(...member.contributions.map((c) => Math.abs(c.contribution)), 1e-9);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Why was this flagged?</h2>
        <span className="small muted">{data.units}</span>
      </div>
      <div className="panel-body">
        <div className="field" style={{ marginBottom: 14 }}>
          <label htmlFor="why-member">Member address</label>
          <select id="why-member" value={index}
            onChange={(e) => setIndex(Number(e.target.value))}>
            {data.per_member.map((m, i) => (
              <option key={m.address} value={i}>{m.address}</option>
            ))}
          </select>
        </div>

        <p className="small muted" style={{ marginTop: 0 }}>
          Model base value{" "}
          <span className="mono"><Value value={member.base_value} /></span>
          {" "}— contributions below are relative to it.
        </p>

        {member.contributions.map((c) => {
          const share = (Math.abs(c.contribution) / widest) * 100;
          const positive = c.contribution >= 0;
          return (
            <div className="contrib" key={c.feature}>
              <div className="contrib-bar">
                <span className={`fill ${positive ? "pos" : "neg"}`}
                  style={positive
                    ? { left: "50%", width: `${share / 2}%` }
                    : { right: "50%", width: `${share / 2}%` }} />
                <span className="zero" style={{ left: "50%" }} />
                <span className="contrib-label">
                  <span className="mono">{c.feature}</span>
                  <span className="faint small">[{c.feature_group}]</span>
                </span>
              </div>
              <span style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <span className="mono small"
                  style={{ color: positive ? "var(--critical)" : "var(--model)",
                           minWidth: 64, textAlign: "right" }}>
                  {positive ? "+" : ""}{c.contribution.toFixed(4)}
                </span>
                <span className="small muted" style={{ minWidth: 92, textAlign: "right" }}>
                  value <Value value={c.feature_value} />
                </span>
                <CategoryChip category={c.value_category} />
              </span>
            </div>
          );
        })}

        <p className="note">{data.meaning}</p>
        <p className="note">{data.insufficient_evidence_meaning}</p>
      </div>
    </section>
  );
}
