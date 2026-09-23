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
import type { NetworkContext, WhyFlagged as WhyFlaggedData } from "../../api/types";
import { CategoryChip, Value } from "../ui/primitives";

export function WhyFlagged({
  data,
  networkContext,
}: {
  data: WhyFlaggedData;
  networkContext?: NetworkContext;
}) {
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
  const hasNetwork = Boolean(networkContext?.available && networkContext.members_with_observations > 0);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Why was this flagged?</h2>
        <span className="small muted">{data.units}</span>
      </div>
      <div className="panel-body">
        {/* 1. MODEL SIGNAL */}
        <div style={{ marginBottom: 18 }}>
          <h3 className="small muted" style={{ margin: "0 0 6px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--cyan)" }}>
            1. Model Signal
          </h3>
          <p className="small muted" style={{ margin: "0 0 10px" }}>
            Elevated transaction and graph behaviour relative to learned baseline log-odds.
          </p>

          <div className="field" style={{ marginBottom: 12 }}>
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
            {" "}— feature contributions below are relative to base log-odds:
          </p>

          {member.contributions.map((c) => {
            const share = (Math.abs(c.contribution) / widest) * 100;
            const positive = c.contribution >= 0;
            return (
              <div className="contrib" key={c.feature} style={{ gridTemplateColumns: "minmax(140px, 34%) minmax(0, 1fr) auto" }}>
                <span className="contrib-label" title={c.feature}>
                  {c.feature} <span className="faint">[{c.feature_group}]</span>
                </span>
                <span className="contrib-bar" aria-hidden="true">
                  <span className={`fill ${positive ? "pos" : "neg"}`} style={{ width: `${share / 2}%` }} />
                </span>
                <span style={{ display: "flex", gap: 10, alignItems: "center", justifyContent: "flex-end" }}>
                  <span className="mono small" style={{ minWidth: 64, textAlign: "right" }}>
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
        </div>

        {/* 5. NETWORK EVIDENCE */}
        <div style={{ marginBottom: 18, borderTop: "1px solid var(--hairline)", paddingTop: 14 }}>
          <h3 className="small muted" style={{ margin: "0 0 6px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--network)" }}>
            2. Network evidence
          </h3>
          {hasNetwork ? (
            <p className="small muted" style={{ margin: 0, lineHeight: 1.6 }}>
              Network observations associated with passive peer propagation vantage points ({networkContext?.members_with_observations} member observations recorded).
            </p>
          ) : (
            <div style={{ padding: "8px 12px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 4 }}>
              <p className="small muted mono" style={{ margin: 0 }}>
                No usable network observations were available for this finding.
              </p>
            </div>
          )}
        </div>

        {/* 6. EVIDENTIARY LIMITATIONS */}
        <div style={{ borderTop: "1px solid var(--hairline)", paddingTop: 14 }}>
          <h3 className="small muted" style={{ margin: "0 0 8px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Evidentiary Guardrails & Scope
          </h3>
          <div className="banner banner-synthetic" style={{ margin: 0 }}>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: "12px", lineHeight: 1.6 }}>
              <li><strong>Investigative priority only:</strong> Risk scores prioritize investigative attention; they do not establish identity or criminality.</li>
              <li><strong>Association, not ownership:</strong> Network observations reflect propagation vantage points and do not by themselves establish wallet ownership.</li>
              <li><strong>Negative evidence distinction:</strong> Absence of network observations indicates unmonitored routes, not proof of non-involvement.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
