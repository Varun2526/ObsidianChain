/**
 * Network separation evidence — the project's contribution, finally visible.
 *
 * `GET /api/evidence/{id}` has existed since Phase 5.2 and nothing ever
 * called it, because no route connected an alert to the separation records
 * inside its cluster. So the question the whole network layer exists to ask -
 * can network-derived constraints stop a harmful merge? - had no answer
 * anywhere in the product.
 *
 * The vocabulary is NOT simplified here. Four things this panel must never
 * let a reader conclude, each of which is stated in the pipeline's own frozen
 * words rather than a paraphrase:
 *
 *   NOT_SEPARATED  is not evidence that two components are the same entity.
 *   NO_EVIDENCE    is not evidence of anything; the test could not be run.
 *   SEPARATED      is the only verdict carrying a constraint, and that
 *                  constraint is CANNOT-LINK - never MUST-LINK.
 *   shared network characteristics are not identity: one server broadcasts
 *                  for tens of thousands of unrelated users.
 */
import { useEffect, useState } from "react";

import * as api from "../../api/console";
import type { SeparationEvidence } from "../../api/types";

const VERDICT_TONE: Record<string, string> = {
  SEPARATED: "separated",
  NOT_SEPARATED: "not-separated",
  NO_EVIDENCE: "no-evidence",
};

export function SeparationEvidencePanel({ alertId }: { alertId: string }) {
  const [data, setData] = useState<SeparationEvidence | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    api.getSeparationEvidence(alertId, controller.signal)
      .then(setData)
      .catch((cause) => {
        if ((cause as Error)?.name !== "AbortError") setError(cause);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [alertId]);

  if (loading) {
    return (
      <section className="panel">
        <div className="panel-head"><h2>Network separation evidence</h2></div>
        <div className="panel-body"><p className="muted">Loading…</p></div>
      </section>
    );
  }

  // Shape-checked, not just null-checked. This panel renders inside the alert
  // page, so a response it does not recognise must degrade to "unavailable"
  // rather than throw - an exception here would blank the entire alert,
  // taking the analytical assessment down with it.
  const usable =
    !!data && Array.isArray(data.rows) && typeof data.verdicts === "object";

  if (error || !usable) {
    return (
      <section className="panel">
        <div className="panel-head">
          <h2>Network separation evidence</h2>
          <span className="cat cat-INSUFFICIENT_EVIDENCE">unavailable</span>
        </div>
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0 }}>
            The separation funnel could not be joined to this cluster. Rather
            than showing an empty table — which would read as “no separation
            evidence exists for this cluster” — nothing is claimed.
          </p>
        </div>
      </section>
    );
  }

  const payload = data as SeparationEvidence;
  const rows = expanded ? payload.rows : payload.rows.slice(0, 8);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Network separation evidence</h2>
        <span className="cat cat-NETWORK_CONTEXT">NETWORK_CONTEXT</span>
        <span className="small muted">
          {payload.proposed_merges_total.toLocaleString()} proposed merges
        </span>
      </div>

      <div className="panel-body">
        <p className="small muted" style={{ marginTop: 0 }}>{payload.statement}</p>

        <div className="stat-row">
          {Object.entries(payload.verdicts).map(([verdict, count]) => (
            <div className="stat" key={verdict}>
              <span className="v">{count.toLocaleString()}</span>
              <span className="k">
                <span className={`verdict verdict-${VERDICT_TONE[verdict] ?? "other"}`}>
                  {verdict.replace("_", " ")}
                </span>
              </span>
            </div>
          ))}
        </div>

        {payload.separated_count === 0 && (
          <div className="banner banner-synthetic">
            <h4>No merge in this cluster was SEPARATED</h4>
            <p>{payload.frozen_run_limitation}</p>
          </div>
        )}

        <p className="note">{payload.cannot_link_meaning}</p>
        <p className="note">{payload.not_separated_meaning}</p>
      </div>

      <div className="panel-body flush" style={{ overflowX: "auto", maxHeight: 340 }}>
        <table>
          <thead>
            <tr>
              <th>Verdict</th><th>Reason</th>
              <th className="num">Pooled</th>
              <th className="num">|A|</th><th className="num">|B|</th>
              <th className="num">p</th><th>Evidence id</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.edge_index}>
                <td>
                  <span className={`verdict verdict-${VERDICT_TONE[row.verdict] ?? "other"}`}>
                    {row.verdict.replace("_", " ")}
                  </span>
                </td>
                <td className="small muted" title={row.reason}>{row.reason_code}</td>
                <td className="num">{row.min_pooled ?? "—"}</td>
                <td className="num">{row.size_a ?? "—"}</td>
                <td className="num">{row.size_b ?? "—"}</td>
                <td className="num mono small">
                  {/* Not measured and measured-as-zero are different answers. */}
                  {row.p_value === null ? <span className="faint">n/a</span>
                    : row.p_value.toExponential(2)}
                </td>
                <td className="mono small faint">{row.evidence_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel-body">
        {payload.rows.length > 8 && (
          <button className="btn btn-sm btn-ghost"
            onClick={() => setExpanded((v) => !v)}>
            {expanded ? "Show fewer" : `Show all ${payload.rows.length} shown rows`}
          </button>
        )}
        {payload.rows_withheld > 0 && (
          <p className="note">
            {payload.rows_withheld.toLocaleString()} further merge records are not
            listed.
          </p>
        )}
        <p className="note" style={{ marginBottom: 0 }}>
          <strong>Provenance.</strong> Alert run{" "}
          <span className="mono">{payload.alert_run_fingerprint}</span>; evidence
          run <span className="mono">{payload.evidence_run_fingerprint}</span>.{" "}
          {payload.join_basis}
        </p>
      </div>
    </section>
  );
}
