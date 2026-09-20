/**
 * The controlled synthetic evaluation.
 *
 * This is the one page in the console that shows numbers which are NOT about
 * the production analytical run, so it is also the page most able to mislead.
 * Three things guard against that.
 *
 * The banner is the first thing rendered and cannot be dismissed. It says
 * plainly that every figure is a property of a generated world.
 *
 * It lives on its own route, reads its own endpoint, and no case, alert or
 * report draws on any of it.
 *
 * Where the experiment did not measure something - false splits, constraint
 * precision - the page says so rather than leaving a gap that invites a
 * reader to assume a zero.
 */
import { useEffect, useState } from "react";

import { ApiError } from "../api/client";
import * as api from "../api/console";

export function EvaluationPage() {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    api.getSyntheticEvaluation(controller.signal)
      .then(setData)
      .catch((cause) => {
        if ((cause as Error)?.name !== "AbortError") setError(cause);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <section className="panel"><div className="panel-body">
        <p className="muted">Loading evaluation…</p></div></section>
    );
  }

  if (error || !data) {
    return (
      <section className="panel"><div className="panel-body">
        <p className="muted" style={{ marginTop: 0 }}>
          {error instanceof ApiError
            ? error.detail
            : "The synthetic evaluation could not be read."}
        </p></div></section>
    );
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Synthetic evaluation</h1>
          <p className="muted">
            Controlled results on a generated world, kept apart from the
            production analytical run
          </p>
        </div>
        <span className="status-badge status-draft">SYNTHETIC_CONTROL</span>
      </div>

      {/* Not dismissible, and first. */}
      <div className="banner banner-synthetic" style={{ marginTop: 0 }}>
        <h4>These figures are not measurements of Bitcoin</h4>
        <p>{data.banner}</p>
      </div>

      {!data.available ? (
        <section className="panel"><div className="panel-body">
          <p className="muted" style={{ marginTop: 0 }}>{data.detail}</p>
        </div></section>
      ) : (
        <>
          <WorldPanel world={data.world} />
          {data.layer_comparison
            ? <ComparisonPanel comparison={data.layer_comparison} />
            : null}
          {data.overlap
            ? <OverlapPanel overlap={data.overlap} />
            : <NotRunPanel />}
        </>
      )}
    </>
  );
}

function NotRunPanel() {
  return (
    <section className="panel">
      <div className="panel-head"><h2>Overlap experiment</h2></div>
      <div className="panel-body">
        <p className="muted" style={{ marginTop: 0 }}>
          The overlap/degradation sweep has not been run for this world.
          Nothing is claimed in its absence.
        </p>
      </div>
    </section>
  );
}

function WorldPanel({ world }: { world: Record<string, any> }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>The world</h2>
        <span className="small muted mono">
          {String(world.world_fingerprint ?? "").slice(0, 16)}
        </span>
      </div>
      <div className="panel-body">
        <div className="card-grid-4">
          {Object.entries(world.counts ?? {}).map(([key, value]) => (
            <div className="stat-mini" key={key}>
              <span className="stat-mini-v">{Number(value).toLocaleString()}</span>
              <span className="stat-mini-k">{key}</span>
            </div>
          ))}
        </div>
        <dl className="kv" style={{ marginTop: 16 }}>
          <dt>Generator</dt><dd className="mono">{world.generator_version}</dd>
          <dt>Network generator</dt>
          <dd className="mono">{world.network_generator_version}</dd>
          <dt>Seed</dt><dd className="mono">{world.seed}</dd>
          <dt>Created</dt><dd>{world.created_at}</dd>
        </dl>

        <h3 className="small muted" style={{ margin: "18px 0 8px" }}>Behaviours</h3>
        <div className="chips">
          {(world.behaviours ?? []).map((name: string) => (
            <span key={name}
              className={`cat cat-${
                (world.adversarial_behaviours ?? []).includes(name)
                  ? "INSUFFICIENT_EVIDENCE" : "BLOCKCHAIN_CONTEXT"}`}>
              {name}
              {(world.adversarial_behaviours ?? []).includes(name)
                ? " · adversarial" : ""}
            </span>
          ))}
        </div>
        <p className="note">{world.behaviour_meaning}</p>
        <p className="note">
          <strong>Positive class:</strong>{" "}
          {(world.positive_class ?? []).join(", ")}.{" "}
          {world.positive_class_meaning}
        </p>
      </div>
    </section>
  );
}

function ComparisonPanel({ comparison }: { comparison: Record<string, any> }) {
  const chain = comparison.chain ?? {};
  const network = comparison.network ?? {};
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Chain / Network / Fused</h2>
        <span className="cat cat-BLOCKCHAIN_CONTEXT">SYNTHETIC</span>
      </div>
      <div className="panel-body">
        <p className="small muted" style={{ marginTop: 0 }}>
          {comparison.meaning}
        </p>

        <h3 className="small muted" style={{ margin: "16px 0 6px" }}>
          Chain layer
        </h3>
        <dl className="kv">
          {Object.entries(chain).map(([key, value]) => (
            <span key={key} style={{ display: "contents" }}>
              <dt>{key.replace(/_/g, " ")}</dt>
              <dd className="mono">{Number(value).toLocaleString()}</dd>
            </span>
          ))}
        </dl>

        <h3 className="small muted" style={{ margin: "16px 0 6px" }}>
          Network layer
        </h3>
        {network.available === false ? (
          <p className="note" style={{ marginTop: 0 }}>
            Network evidence unavailable for this world. {network.reason}
          </p>
        ) : (
          <dl className="kv">
            {Object.entries(network)
              .filter(([key]) => key !== "available")
              .map(([key, value]) => (
                <span key={key} style={{ display: "contents" }}>
                  <dt>{key.replace(/_/g, " ")}</dt>
                  <dd className="mono">{Number(value).toLocaleString()}</dd>
                </span>
              ))}
          </dl>
        )}

        <p className="note">{comparison.fused?.note}</p>
        {/* Named absences, so a gap is never read as a zero. */}
        <p className="note">
          <strong>Not measured here.</strong> {comparison.not_measured_here}
        </p>
      </div>
    </section>
  );
}

function OverlapPanel({ overlap }: { overlap: Record<string, any> }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Overlap and degradation</h2>
        <span className="small muted">
          {overlap.levels?.length ?? 0} conditions
        </span>
      </div>
      <div className="panel-body">
        <p className="small muted" style={{ marginTop: 0 }}>{overlap.meaning}</p>
      </div>
      <div className="panel-body flush" style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th className="num">Overlap</th>
              <th className="num">Transactions observed</th>
              <th className="num">Coverage</th>
              <th className="num">Abstention</th>
              <th>Network evidence</th>
            </tr>
          </thead>
          <tbody>
            {(overlap.levels ?? []).map((level: Record<string, any>) => (
              <tr key={level.requested_overlap}>
                <td className="num mono">
                  {(level.observed_overlap * 100).toFixed(0)}%
                </td>
                <td className="num">
                  {Number(level.transactions_with_observations).toLocaleString()}
                </td>
                <td className="num mono">{level.coverage.toFixed(3)}</td>
                <td className="num mono">{level.abstention.toFixed(3)}</td>
                <td>
                  {level.evidence_available ? (
                    <span className="runchip runchip-current">available</span>
                  ) : (
                    <span className="runchip runchip-unverifiable">
                      unavailable
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="panel-body">
        <p className="note" style={{ marginTop: 0 }}>
          {overlap.abstention_meaning}
        </p>
      </div>
    </section>
  );
}
