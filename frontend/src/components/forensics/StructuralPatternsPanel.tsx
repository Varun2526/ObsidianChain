/**
 * Peeling-chain and mixing-like structure, for one alert.
 *
 * Both are OBSERVATIONS about the shape of transactions. The panel is
 * careful about two things in particular.
 *
 * A mixing-like pattern is not a mixer, and using a mixer is not unlawful.
 * The heading says "pattern", the backend's frozen wording is rendered
 * verbatim, and the four benign shapes the detector suppressed are shown
 * WITH their counts - because "we looked at 40 transactions, 32 were
 * ordinary payments and 4 were batches" is a far more useful thing to put in
 * front of an investigator than a bare "no mixing found".
 *
 * A peeling-chain-like structure is not laundering. It is a repeated forward
 * hop pattern, which is structural candidate generation, and the panel
 * reports depth and how many members sit in a chain rather than a verdict.
 */
import { useEffect, useState } from "react";

import * as api from "../../api/console";
import type { AlertPatterns, MixingClass } from "../../api/types";

const MIXING_TONE: Record<MixingClass, string> = {
  MIXING_PATTERN: "pattern",
  MIXING_LIKELIHOOD: "likelihood",
  NO_MIXING_SIGNAL: "none",
  INSUFFICIENT_DATA: "insufficient",
};

const MIXING_LABEL: Record<MixingClass, string> = {
  MIXING_PATTERN: "Mixing-like pattern",
  MIXING_LIKELIHOOD: "Partial mixing signal",
  NO_MIXING_SIGNAL: "No mixing signal",
  INSUFFICIENT_DATA: "Insufficient data",
};

export function StructuralPatternsPanel({ alertId }: { alertId: string }) {
  const [data, setData] = useState<AlertPatterns | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    api.getAlertPatterns(alertId, controller.signal)
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
        <div className="panel-head"><h2>Structural patterns</h2></div>
        <div className="panel-body"><p className="muted">Loading…</p></div>
      </section>
    );
  }

  // Shape-checked, not just null-checked: an unrecognised payload must
  // degrade to "unavailable" rather than throw and blank the alert page.
  const usable = !!data && !!data.peeling && !!data.mixing;
  if (error || !usable) {
    return (
      <section className="panel">
        <div className="panel-head">
          <h2>Structural patterns</h2>
          <span className="cat cat-INSUFFICIENT_EVIDENCE">unavailable</span>
        </div>
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0 }}>
            The structural pattern scan could not be read for this alert.
            Nothing is claimed either way — this is “not measured”, not
            “measured and clean”.
          </p>
        </div>
      </section>
    );
  }

  const { peeling, mixing } = data as AlertPatterns;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Structural patterns</h2>
        <span className="cat cat-BLOCKCHAIN_CONTEXT">BLOCKCHAIN_CONTEXT</span>
      </div>

      {/* ---- peeling ---- */}
      <div className="panel-body">
        <h3 className="small muted" style={{ marginTop: 0 }}>
          Peeling-chain-like structure
        </h3>
        {peeling.available ? (
          <>
            <div className="stat-row">
              <div className="stat">
                <span className="v">
                  {peeling.members_in_chain.toLocaleString()}
                  <span className="faint" style={{ fontSize: 13 }}>
                    {" / "}{peeling.members_scored.toLocaleString()}
                  </span>
                </span>
                <span className="k">Members in a chain</span>
              </div>
              <div className="stat">
                <span className="v">
                  {peeling.max_chain_depth ?? <span className="faint">n/a</span>}
                </span>
                <span className="k">Deepest chain</span>
              </div>
            </div>
            {peeling.members_in_chain === 0 && (
              <p className="muted small">
                No member of this cluster sits in a PEEL-1 chain at the
                configured depth.
              </p>
            )}
          </>
        ) : (
          <p className="note" style={{ marginTop: 0 }}>
            <strong>Insufficient evidence.</strong> {peeling.detail}
          </p>
        )}
        <p className="note">{peeling.meaning}</p>
      </div>

      {/* ---- mixing ---- */}
      <div className="panel-body" style={{ borderTop: "1px solid var(--hairline)" }}>
        <h3 className="small muted" style={{ marginTop: 0 }}>
          Mixing-like transaction pattern
        </h3>

        {!mixing.available ? (
          <p className="note" style={{ marginTop: 0 }}>
            <strong>Insufficient evidence.</strong> {mixing.detail}
          </p>
        ) : (
          <>
            <div className="stat-row">
              <div className="stat">
                <span className="v">{mixing.pattern_count}</span>
                <span className="k">Transactions showing the pattern</span>
              </div>
              <div className="stat">
                <span className="v">{mixing.transactions_measured}</span>
                <span className="k">Transactions measured</span>
              </div>
            </div>

            <div className="chips" style={{ marginTop: 10 }}>
              {Object.entries(mixing.classes).map(([name, count]) => (
                <span key={name}
                  className={`mix mix-${MIXING_TONE[name as MixingClass] ?? "none"}`}>
                  {MIXING_LABEL[name as MixingClass] ?? name} · {count}
                </span>
              ))}
            </div>

            {/* What the detector REFUSED to flag, and why. An investigator
                learns more from "32 were ordinary payments" than from a
                bare absence. */}
            {Object.keys(mixing.suppressed).length > 0 && (
              <>
                <h4 className="small muted" style={{ margin: "16px 0 6px" }}>
                  Recognised as benign shapes
                </h4>
                <ul className="suppressor-list">
                  {Object.entries(mixing.suppressed).map(([code, count]) => (
                    <li key={code}>
                      <span className="mono small">{code}</span>
                      <span className="faint small"> × {count}</span>
                      <p className="small muted">
                        {mixing.suppressor_meanings[code]}
                      </p>
                    </li>
                  ))}
                </ul>
              </>
            )}

            {mixing.transactions.some(
              (t) => t.mixing_class === "MIXING_PATTERN",
            ) && (
              <div className="panel-body flush" style={{ overflowX: "auto" }}>
                <table>
                  <thead>
                    <tr>
                      <th>Transaction</th><th className="num">In</th>
                      <th className="num">Out</th>
                      <th className="num">Output uniformity</th>
                      <th className="num">Input variety</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mixing.transactions
                      .filter((t) => t.mixing_class === "MIXING_PATTERN")
                      .map((t) => (
                        <tr key={t.txid}>
                          <td className="mono small">{t.txid.slice(0, 16)}…</td>
                          <td className="num">{t.n_inputs}</td>
                          <td className="num">{t.n_outputs}</td>
                          <td className="num mono small">
                            {t.output_uniformity === null
                              ? <span className="faint">n/a</span>
                              : t.output_uniformity.toFixed(3)}
                          </td>
                          <td className="num mono small">
                            {t.input_heterogeneity === null
                              ? <span className="faint">n/a</span>
                              : t.input_heterogeneity.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}

            <p className="note">{mixing.meaning}</p>
            <p className="note" style={{ marginBottom: 0 }}>
              <strong>Provenance.</strong> Scan{" "}
              <span className="mono">{mixing.scan_id}</span>. {mixing.join_basis}
            </p>
          </>
        )}
      </div>
    </section>
  );
}
