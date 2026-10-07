/**
 * Model intelligence: which model serves, how it was measured, and what is
 * not yet known. Every number carries its result type. There is no
 * production performance, and the page says so instead of leaving a gap a
 * research number could be mistaken for.
 */
import { Fragment } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { getModel, listModels } from "../../api/intel";
import type { ModelDetail, ModelList, ResultType } from "../../api/intel";
import { Reliability, ScoreBars } from "../../components/ui/charts";
import { ErrorState } from "../../components/ui/ErrorState";
import { Metric, PageHeader, ResultTypeTag, fixed, int, pct } from "../../components/ui/intel";
import { Skeleton } from "../../components/ui/primitives";
import { useApi } from "../../lib/useApi";
import { featureLabel, featureSetName, modelBlurb, modelName, modelRole, protocolLabel, statusLabel } from "../../lib/labels";

export function ModelsPage() {
  const list = useApi((s) => listModels(s), []);
  const [params, setParams] = useSearchParams();
  const champion = list.data?.roles.champion ?? null;
  const version = params.get("v") ?? champion;
  const detail = useApi(version ? (s) => getModel(version, s) : null, [version]);

  return (
    <>
      <PageHeader
        eyebrow="Intelligence"
        title="Model intelligence"
        sub="Which model scores uploaded datasets, how it was measured, and where it is known to fail. Research results are labelled by the data they were measured on."
        actions={<Link className="btn btn-sm" to="/evaluation">Synthetic control evaluation</Link>}
      />
      <div className="banner banner-model" role="note">
        <h4>A model score is a ranking signal, not a finding</h4>
        <p>
          The model in service ranks addresses by learned association with the Elliptic++ illicit class. A high score is a reason to look,
          never evidence on its own. Performance on live traffic is unknown until delayed labels arrive.
        </p>
      </div>

      {list.error != null && <ErrorState error={list.error} onRetry={list.reload} />}
      {list.loading && !list.data && <div className="panel"><Skeleton rows={5} /></div>}
      {list.data && <Registry list={list.data} selected={version} onSelect={(v) => setParams({ v })} />}

      {detail.error != null && <ErrorState error={detail.error} onRetry={detail.reload} />}
      {detail.loading && !detail.data && version && <div className="panel"><Skeleton rows={8} /></div>}
      {detail.data && <ModelView m={detail.data} />}
    </>
  );
}

function Registry({ list, selected, onSelect }: { list: ModelList; selected: string | null; onSelect: (v: string) => void }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Registry</h2>
        {Object.entries(list.roles).map(([role, v]) => (
          <span key={role} className="chip" title={v ?? undefined}>{modelRole(role)}: <span style={{ textTransform: "none", letterSpacing: 0 }}>{v ? modelName(v) : "none"}</span></span>
        ))}
      </div>
      <div className="panel-body flush table-wrap">
        <table>
          <thead><tr><th>Model</th><th>Role</th><th>Features</th><th>Attested commit</th><th className="num">Holdout nAP</th><th className="num">Holdout P@100</th><th>Status</th></tr></thead>
          <tbody>
            {[...list.models].reverse().map((m) => (
              <tr key={m.version} className="clickable" aria-selected={m.version === selected} tabIndex={0}
                  onClick={() => onSelect(m.version)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(m.version); } }}>
                <td title={m.version}><strong style={{ fontWeight: 600 }}>{modelName(m.version)}</strong></td>
                <td>{m.role ? <span className="chip" style={m.role === "champion" ? { color: "var(--oc-accent)", borderColor: "var(--oc-accent-line)" } : undefined}>{modelRole(m.role)}</span> : <span className="faint">—</span>}</td>
                <td className="small" title={m.feature_schema_version ?? undefined}>{featureSetName(m.feature_schema_version)}</td>
                <td className="mono small">{m.attested_source_commit ? m.attested_source_commit.slice(0, 10) : <span className="faint">not attested</span>}</td>
                <td className="num">{m.holdout?.nap != null ? fixed(m.holdout.nap) : <span className="faint">not opened</span>}</td>
                <td className="num">{m.holdout?.["P@100"] != null ? fixed(m.holdout["P@100"], 2) : ""}</td>
                <td className="small">{m.withdrawn ? <span style={{ color: "var(--oc-danger)" }}>Withdrawn</span> : statusLabel(((m.notes ?? "").split(".")[0] ?? "").trim() || null)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="panel-foot">
        Only the model marked <strong>In service</strong> scores uploaded data. A model <strong>under evaluation</strong> is scored
        alongside it but never shown as a result, and the <strong>backup</strong> takes over only if the main model fails its
        integrity check. Select a model to see its record.
      </div>
    </section>
  );
}

function mean(s: Record<string, { mean: number; sd: number }> | undefined, k: string) {
  const v = s?.[k];
  return v ? `${v.mean.toFixed(3)} ± ${v.sd.toFixed(3)}` : "n/a";
}

function ModelView({ m }: { m: ModelDetail }) {
  const ev = m.evaluation;
  const ho = m.holdout;
  const confirmNap = ev?.summary.confirm?.address.nap?.mean;
  const steps = ho?.by_first_seen_step ? Object.entries(ho.by_first_seen_step) : [];
  const features = Array.isArray(m.manifest.features) ? (m.manifest.features as string[]) : [];

  return (
    <>
      <div className="section-title">
        <span title={m.version} style={{ textTransform: "none", letterSpacing: 0, color: "var(--oc-text)", fontSize: 14, fontWeight: 600 }}>{modelName(m.version)}</span>
        {m.roles.map((r) => <span key={r} className="chip">{modelRole(r)}</span>)}
        <span style={{ textTransform: "none", letterSpacing: 0 }}>{String(m.manifest.model_type ?? "")} · {protocolLabel(String(m.manifest.evaluation_protocol ?? ev?.protocol ?? ""))}</span>
      </div>

      <div className="grid-2" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", marginBottom: 16 }}>
        <ResultBlock type="DEVELOPMENT" note={m.result_types.DEVELOPMENT}>
          <Metric k="nAP (mean ± sd)" v={mean(ev?.summary.tune?.address, "nap")} />
          <Metric k="P@100" v={mean(ev?.summary.tune?.address, "P@100")} />
        </ResultBlock>
        <ResultBlock type="CONFIRMATION" note={m.result_types.CONFIRMATION}>
          <Metric k="nAP (mean ± sd)" v={mean(ev?.summary.confirm?.address, "nap")} />
          <Metric k="P@100" v={mean(ev?.summary.confirm?.address, "P@100")} />
        </ResultBlock>
        <ResultBlock type="HOLDOUT" note={ho ? `${m.result_types.HOLDOUT}. ${int(ho.n_addresses)} addresses, ${int(ho.positives)} positive (${pct(ho.prevalence)}).` : "Not opened for this version."}>
          <Metric k="nAP" v={fixed(ho?.address.nap)} />
          <Metric k="P@100" v={fixed(ho?.address["P@100"], 2)} />
          <Metric k="ECE" v={fixed(ho?.calibration.ece, 4)} />
        </ResultBlock>
        <ResultBlock type="PRODUCTION" note={m.production.meaning}>
          <Metric k="nAP" v="unknown" d="no labelled production traffic" />
        </ResultBlock>
      </div>

      {steps.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2>Holdout by timestep</h2><ResultTypeTag type="HOLDOUT" />
            <span className="small faint">nAP for addresses first seen at each step; dashed line is the confirmation mean</span>
          </div>
          <div className="panel-body">
            <ScoreBars
              ariaLabel="Holdout nAP per timestep"
              data={steps.map(([k, v]) => ({ label: k, value: v.nap, tone: v.nap != null && v.nap < 0.3 ? "var(--oc-sev-critical)" : "var(--oc-accent)",
                                              note: `n ${v.n}, positives ${v.positives}` }))}
              reference={confirmNap} refLabel="confirmation mean"
            />
            <table style={{ marginTop: 12 }}>
              <thead><tr><th>Step</th><th className="num">Addresses</th><th className="num">Positives</th><th className="num">nAP</th><th className="num">P@100</th></tr></thead>
              <tbody>{steps.map(([k, v]) => (
                <tr key={k}><td className="mono">{k}</td><td className="num">{int(v.n)}</td><td className="num">{int(v.positives)}</td>
                  <td className="num" style={v.nap != null && v.nap < 0.3 ? { color: "var(--oc-sev-critical)" } : undefined}>{fixed(v.nap)}</td>
                  <td className="num">{fixed(v["P@100"] ?? null, 2)}</td></tr>
              ))}</tbody>
            </table>
            <div className="banner banner-error" style={{ marginTop: 12, marginBottom: 0 }}>
              <h4>Known failure</h4>
              <p>
                Ranking collapses in some windows (red bars). The collapse comes from concept drift that input monitoring does not detect,
                so a quiet drift monitor does not mean the model is working. Deployment requires the delayed-label review loop.
              </p>
            </div>
          </div>
        </section>
      )}

      <div className="grid-2">
        {ev && (
          <section className="panel">
            <div className="panel-head"><h2>Rolling folds</h2><ResultTypeTag type="DEVELOPMENT" /><ResultTypeTag type="CONFIRMATION" /></div>
            <div className="panel-body">
              <ScoreBars
                ariaLabel="nAP per rolling fold"
                data={ev.folds.map((f, i) => ({ label: String(i + 1), value: f.nap, tone: f.part === "tune" ? "var(--oc-text-3)" : "var(--oc-ev-model)", note: f.fold }))}
              />
              <p className="note">Grey: development folds used for selection. Blue: confirmation folds never used for selection.</p>
            </div>
            <div className="panel-body flush table-wrap" style={{ maxHeight: 300 }}>
              <table>
                <thead><tr><th>#</th><th>Window</th><th>Type</th><th className="num">n</th><th className="num">Prev.</th><th className="num">nAP</th><th className="num">P@100</th></tr></thead>
                <tbody>{ev.folds.map((f, i) => (
                  <tr key={i}><td className="num">{i + 1}</td><td className="mono small">{f.fold}</td><td><ResultTypeTag type={f.result_type as ResultType} /></td>
                    <td className="num">{int(f.n)}</td><td className="num">{pct(f.prevalence)}</td><td className="num">{fixed(f.nap)}</td><td className="num">{fixed(f["P@100"], 2)}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </section>
        )}

        {ho?.reliability && (
          <section className="panel">
            <div className="panel-head"><h2>Calibration</h2><ResultTypeTag type="HOLDOUT" /></div>
            <div className="panel-body" style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
              <Reliability bins={ho.reliability} ariaLabel="Holdout reliability: predicted against observed rate" />
              <dl className="kv" style={{ flex: 1, minWidth: 180 }}>
                {Object.entries(ho.calibration).map(([k, v]) => (
                  <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd className="num">{fixed(v, 4)}</dd></Fragment>
                ))}
              </dl>
            </div>
            <div className="panel-foot">The displayed probability is calibrated; ranking uses the raw score.</div>
          </section>
        )}
      </div>

      <div className="grid-2">
        {m.gate && (
          <section className="panel">
            <div className="panel-head">
              <h2>Production gate</h2>
              <span className="chip" style={{ color: m.gate.decision === "PASS" ? "var(--oc-ok)" : "var(--oc-danger)" }}>{m.gate.decision}</span>
              <span className="small faint">thresholds committed before the holdout was opened</span>
            </div>
            <div className="panel-body flush table-wrap" style={{ maxHeight: 420 }}>
              <table>
                <thead><tr><th>Criterion</th><th>Status</th><th>Evidence</th></tr></thead>
                <tbody>{Object.entries(m.gate.criteria).map(([k, c]) => (
                  <tr key={k}>
                    <td className="small">{k.replace(/^\d+_/, "").replace(/_/g, " ")}</td>
                    <td><span className="chip" style={{ color: c.status === "PASS" ? "var(--oc-ok)" : c.status === "FAIL" ? "var(--oc-danger)" : "var(--oc-warn)" }}>{c.status}</span></td>
                    <td className="small mono" style={{ maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={JSON.stringify(c.evidence)}>
                      {typeof c.evidence === "string" ? c.evidence : JSON.stringify(c.evidence)}
                    </td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </section>
        )}

        {ho && Object.keys(ho.slices_nap).length > 0 && (
          <section className="panel">
            <div className="panel-head"><h2>Where it is weaker</h2><ResultTypeTag type="HOLDOUT" /><span className="small faint">nAP by address slice</span></div>
            <div className="panel-body flush">
              <table>
                <thead><tr><th>Slice</th><th className="num">nAP</th><th /></tr></thead>
                <tbody>{Object.entries(ho.slices_nap).sort((a, b) => (a[1] ?? 0) - (b[1] ?? 0)).map(([k, v]) => (
                  <tr key={k}><td className="small">{k.replace(/_/g, " ")}</td><td className="num">{fixed(v)}</td>
                    <td style={{ width: "40%" }}><span className="bar bar-wide"><span style={{ width: `${(v ?? 0) * 100}%`, background: (v ?? 0) < 0.3 ? "var(--oc-sev-critical)" : "var(--oc-ev-model)" }} /></span></td></tr>
                ))}</tbody>
              </table>
            </div>
          </section>
        )}
      </div>

      {m.drift_baseline && (
        <section className="panel">
          <div className="panel-head"><h2>Drift baseline</h2><ResultTypeTag type="DEVELOPMENT" /><span className="small faint">score PSI per development step; alert threshold p{Number(m.drift_baseline.quantile) * 100} = {String(m.drift_baseline.score_psi_p95)}</span></div>
          <div className="panel-body">
            <ScoreBars ariaLabel="Score PSI per development step"
                       data={m.drift_baseline.per_step.map((p) => ({ label: String(p.step), value: Math.min(1, p.score_psi / 2), note: `PSI ${p.score_psi}` }))}
                       color="var(--oc-text-3)" height={100} />
            <p className="note">Bars are PSI / 2, clipped at 1. {m.drift_baseline.finding}</p>
          </div>
        </section>
      )}

      <div className="grid-2">
        <section className="panel">
          <div className="panel-head"><h2>Model record</h2></div>
          <div className="panel-body">
            <dl className="kv">
              <dt>Model</dt><dd>{modelName(m.version)} <span className="mono small faint">{m.version}</span></dd>
              <dt>What it is</dt><dd>{modelBlurb(m.version)}</dd>
              <dt>Algorithm</dt><dd>{String(m.manifest.model_type ?? "n/a")}</dd>
              <dt>Status</dt><dd>{statusLabel(m.manifest.status as string | undefined)}</dd>
              <dt>How it was tested</dt><dd>{protocolLabel(m.manifest.evaluation_protocol as string | undefined)}</dd>
              <dt>Trained on</dt><dd>{String(m.manifest.final_training_set ?? "n/a")}</dd>
              <dt>Features</dt><dd>{features.length} ({featureSetName(m.feature_schema_version)})</dd>
              <dt>Ranking score</dt><dd>The model's probability, used to order addresses</dd>
              <dt>Displayed probability</dt><dd>Calibrated so that 0.30 means about 30% were illicit in testing</dd>
              <dt>Explanations</dt><dd>Per-feature contributions (TreeSHAP) for every score</dd>
              <dt>Artifact SHA-256</dt><dd className="mono small">{String(m.manifest.model_sha256 ?? "n/a")}</dd>
              <dt>Attested commit</dt><dd className="mono small">{m.attested_source_commit ?? "not attested"}</dd>
              <dt>Holdout record</dt><dd className="mono small">{ho?.sha256 ?? "n/a"}</dd>
            </dl>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head"><h2>Features</h2><span className="small faint">{features.length}</span></div>
          <div className="panel-body">
            <div className="chips">{features.map((f) => <span key={f} className="chip" title={f}>{featureLabel(f)}</span>)}</div>
            {m.evaluation?.performance && (
              <p className="note" style={{ marginTop: 12 }}>
                Scoring latency p50 {String((m.evaluation.performance.single_row_latency_ms as Record<string, number> | undefined)?.p50 ?? "n/a")} ms per row;
                {" "}{int(m.evaluation.performance.rows_per_second_at_10k as number)} rows/s at 10k (recorded at training time).
              </p>
            )}
          </div>
        </section>
      </div>
      {m.notes && <p className="note">Registry note: {m.notes}</p>}
    </>
  );
}

function ResultBlock({ type, note, children }: { type: ResultType; note: string; children: React.ReactNode }) {
  return (
    <section className="panel" style={{ margin: 0 }}>
      <div className="panel-head"><ResultTypeTag type={type} /></div>
      <div className="metric-strip" style={{ border: 0, borderRadius: 0, margin: 0 }}>{children}</div>
      <div className="panel-foot">{note}</div>
    </section>
  );
}
