/**
 * Evidence traceability, made visible.
 *
 * An investigator has to be able to answer "where did this number come
 * from?" without leaving the page: which dataset, which run, which model,
 * and under what as-of semantics the features were computed.
 */
import type { Provenance } from "../../api/types";

export function ProvenancePanel({ provenance }: { provenance: Provenance }) {
  const p = provenance;
  return (
    <section className="panel">
      <div className="panel-head"><h2>Provenance</h2></div>
      <div className="panel-body">
        <dl className="kv">
          <dt>Provenance type</dt><dd>{p.provenance_type ?? "—"}</dd>
          <dt>Dataset</dt><dd>{p.dataset_id ?? "—"}</dd>
          <dt>Dataset SHA-256</dt><dd>{p.dataset_sha256 ?? "—"}</dd>
          <dt>Run fingerprint</dt><dd>{p.run_fingerprint ?? "—"}</dd>
          <dt>Training dataset</dt><dd>{p.phase6_dataset_fingerprint ?? "—"}</dd>
          <dt>Artifact schema</dt><dd>{p.artifact_schema ?? "—"}</dd>
          {p.git_revision ? (<><dt>Git revision</dt><dd>{p.git_revision}</dd></>) : null}
        </dl>

        {p.model ? (
          <>
            <h3 className="small muted" style={{ margin: "18px 0 8px" }}>Model</h3>
            <dl className="kv">
              <dt>Family</dt><dd>{p.model.family}</dd>
              <dt>Features</dt><dd>{p.model.n_features}</dd>
              <dt>Calibration</dt><dd style={{ fontFamily: "var(--sans)" }}>{p.model.calibration}</dd>
              <dt>Explainability</dt><dd style={{ fontFamily: "var(--sans)" }}>{p.model.explainability}</dd>
              {p.ranking_aggregation ? (<><dt>Ranking</dt><dd>{p.ranking_aggregation}</dd></>) : null}
              {p.scored_split ? (<><dt>Scored split</dt><dd>{p.scored_split}</dd></>) : null}
            </dl>
          </>
        ) : null}

        {p.feature_semantics ? (
          <>
            <h3 className="small muted" style={{ margin: "18px 0 8px" }}>
              Feature as-of semantics
            </h3>
            <p className="note" style={{ marginTop: 0 }}>{p.feature_semantics}</p>
            {p.split ? (
              <dl className="kv" style={{ marginTop: 10 }}>
                {Object.entries(p.split).map(([k, v]) => (
                  <span key={k} style={{ display: "contents" }}>
                    <dt>{k}</dt><dd>{v}</dd>
                  </span>
                ))}
              </dl>
            ) : null}
          </>
        ) : null}

        {p.severity_bands?.length ? (
          <>
            <h3 className="small muted" style={{ margin: "18px 0 8px" }}>
              Severity bands (selected on the validation split only)
            </h3>
            <table>
              <thead>
                <tr><th>Band</th><th className="num">Precision target</th>
                  <th className="num">Threshold</th><th className="num">Support</th>
                  <th className="num">Validation precision</th></tr>
              </thead>
              <tbody>
                {p.severity_bands.map((b) => (
                  <tr key={b.band}>
                    <td>{b.band}</td>
                    <td className="num">{b.target.toFixed(2)}</td>
                    <td className="num">
                      {b.populated && b.threshold !== null
                        ? b.threshold.toFixed(6)
                        : <span className="faint">UNPOPULATED</span>}
                    </td>
                    <td className="num">{b.support.toLocaleString()}</td>
                    <td className="num">
                      {b.validation_precision !== null
                        ? b.validation_precision.toFixed(4) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : null}

        {p.notes?.length ? (
          <>
            <h3 className="small muted" style={{ margin: "18px 0 8px" }}>Recorded limitations</h3>
            <ul className="note" style={{ paddingLeft: 18, margin: 0 }}>
              {p.notes.map((n, i) => <li key={i} style={{ marginBottom: 5 }}>{n}</li>)}
            </ul>
          </>
        ) : null}
      </div>
    </section>
  );
}
