/**
 * Network evidence, as investigative context and nothing more.
 *
 * This panel is the one most likely to be misread, so it states its own
 * limits in the backend's words rather than a paraphrase: the network layer
 * emits cannot-link only, one server broadcasts for tens of thousands of
 * unrelated users, and nothing here identifies a sender or an owner.
 *
 * Where no observations were pooled the panel says INSUFFICIENT EVIDENCE
 * rather than showing zeros that would read as measurements.
 */
import type { NetworkContext } from "../../api/types";
import { CategoryChip } from "../ui/primitives";

export function NetworkContextPanel({ context }: { context: NetworkContext }) {
  const insufficient = context.status === "INSUFFICIENT_EVIDENCE" || !context.available;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Network context</h2>
        <CategoryChip category={context.category} />
        {insufficient ? <CategoryChip category="INSUFFICIENT_EVIDENCE" /> : null}
      </div>

      <div className="panel-body">
        <div className="banner banner-synthetic">
          <h4>Network observations</h4>
          <p>{context.synthetic_warning}</p>
        </div>

        {insufficient ? (
          <p className="note" style={{ marginTop: 0 }}>
            <strong>Insufficient evidence.</strong>{" "}
            {context.insufficient_evidence_meaning ??
              "No network observations were pooled for the members of this alert."}
          </p>
        ) : (
          <div className="stat-row">
            <div className="stat">
              <span className="v">{context.members_with_observations.toLocaleString()}</span>
              <span className="k">Members with observations</span>
            </div>
            <div className="stat">
              <span className="v">
                {context.members_reaching_production_minimum.toLocaleString()}
              </span>
              <span className="k">Reaching production minimum</span>
            </div>
          </div>
        )}

        <p className="note">{context.meaning}</p>
        <p className="note" style={{ fontSize: 11, color: "var(--text-dim)" }}>
          Network observations provide association and context. They do not by themselves establish wallet ownership.
        </p>
      </div>
    </section>
  );
}
