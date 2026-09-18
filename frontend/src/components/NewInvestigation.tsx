/**
 * Create an investigation: name → upload → validate → analysis state.
 *
 * The step this rewrite changes is the last one.
 *
 * It used to be called "Analysis". It fetched the GLOBAL alert list, copied
 * that run's fingerprint onto the new case, set the case ACTIVE, and
 * reported "2,128 precomputed alerts" - so a case whose data had never been
 * scored looked like a case with 2,128 results. That was the single most
 * dangerous ambiguity in the product.
 *
 * Now the step reports the case's own AnalysisRun, which the backend creates
 * as NOT_RUN with a NULL fingerprint. The case is not bound to any run, and
 * it says so.
 */
import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import * as api from "../api/console";
import type { AnalysisRun, IngestResult, Investigation, UploadedDataset } from "../api/types";
import { ErrorState } from "./ErrorState";

type Step = "create" | "upload" | "validation" | "analysis";

const ACCEPTED = ".csv,.json,.xml";
const STEPS: Step[] = ["create", "upload", "validation", "analysis"];

export function NewInvestigation() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<Step>("create");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [investigation, setInvestigation] = useState<Investigation | null>(null);

  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [dataset, setDataset] = useState<UploadedDataset | null>(null);
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const validation = dataset?.validation as IngestResult | undefined;

  const handleCreate = async () => {
    setBusy(true);
    setError(null);
    try {
      setInvestigation(await api.createInvestigation(name, description));
      setStep("upload");
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      setBusy(false);
    }
  };

  const handleFile = useCallback(
    async (file: File) => {
      if (!investigation) return;
      setBusy(true);
      setError(null);
      try {
        const result = await api.uploadDataset(investigation.id, file);
        setDataset(result.dataset);
        setRun(result.analysis_run as unknown as AnalysisRun);
        setStep("validation");
      } catch (cause) {
        setError(cause instanceof Error ? cause : new Error(String(cause)));
      } finally {
        setBusy(false);
      }
    },
    [investigation],
  );

  const openInvestigation = () => {
    if (investigation) navigate(`/inv/${investigation.id}`);
  };

  return (
    <>
      <div className="page-header">
        <div>
          <h1>New Investigation</h1>
          <p className="muted">Step {STEPS.indexOf(step) + 1} of 4</p>
        </div>
      </div>

      <div className="step-progress">
        {STEPS.map((s, i) => (
          <div
            key={s}
            className={`step-dot${step === s ? " active" : ""}${
              STEPS.indexOf(step) > i ? " done" : ""
            }`}
          >
            <span className="step-dot-n">{i + 1}</span>
            <span className="step-dot-label">
              {["Create", "Upload", "Validate", "Analysis"][i]}
            </span>
          </div>
        ))}
      </div>

      {/* STEP 1 */}
      {step === "create" && (
        <section className="panel">
          <div className="panel-head"><h2>Investigation details</h2></div>
          <div className="panel-body">
            <div className="form-group">
              <label htmlFor="inv-name">Investigation name</label>
              <input id="inv-name" type="text" value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Bitcoin financial flow investigation" />
            </div>
            <div className="form-group">
              <label htmlFor="inv-desc">Description</label>
              <textarea id="inv-desc" rows={3} value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of the investigation scope…" />
            </div>
            {error ? <ErrorState error={error} /> : null}
            <div className="form-actions">
              <button className="btn btn-primary" onClick={handleCreate}
                disabled={!name.trim() || busy}>
                {busy ? "Creating…" : "Create investigation"}
              </button>
            </div>
          </div>
        </section>
      )}

      {/* STEP 2 */}
      {step === "upload" && investigation && (
        <section className="panel">
          <div className="panel-head">
            <h2>Data source</h2>
            <span className="small muted">
              {investigation.case_label} · {investigation.name}
            </span>
          </div>
          <div className="panel-body">
            <p className="muted small" style={{ marginTop: 0 }}>
              Upload Bitcoin transaction and network metadata as{" "}
              <strong>CSV, JSON or XML</strong>. The file is parsed, normalised,
              validated and correlated locally, then <strong>stored with this
              investigation</strong> and addressed by its SHA-256.
            </p>
            <div
              className={`dropzone${dragging ? " active" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault(); setDragging(false);
                const f = e.dataTransfer.files?.[0];
                if (f) void handleFile(f);
              }}
              onClick={() => input.current?.click()}
              role="button" tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter") input.current?.click(); }}
            >
              <strong>{busy ? "Processing…" : "Drop a file here, or click to browse"}</strong>
              <span className="small faint">CSV · JSON · XML</span>
              <input ref={input} type="file" accept={ACCEPTED} hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleFile(f);
                }} />
            </div>
            {error ? <ErrorState error={error} /> : null}
          </div>
        </section>
      )}

      {/* STEP 3 */}
      {step === "validation" && dataset && (
        <>
          <section className="panel">
            <div className="panel-head">
              <h2>Data validation</h2>
              <span className={`status-badge status-${
                dataset.status === "VALIDATED" ? "active" : "closed"}`}>
                {dataset.status}
              </span>
              <span className="small muted">
                {dataset.filename} · {dataset.format.toUpperCase()}
              </span>
            </div>
            <div className="panel-body">
              <dl className="kv" style={{ marginBottom: 16 }}>
                <dt>Stored SHA-256</dt><dd className="mono">{dataset.sha256}</dd>
                <dt>Size</dt><dd>{dataset.size_bytes.toLocaleString()} bytes</dd>
              </dl>

              {validation?.validation ? (
                <div className="card-grid-4">
                  <StatMini k="Records" v={validation.validation.rows_read} />
                  <StatMini k="Valid" v={validation.validation.rows_valid} />
                  <StatMini k="Rejected" v={validation.validation.rows_rejected} />
                  <StatMini k="Transactions" v={validation.correlation?.transactions ?? 0} />
                  <StatMini k="Addresses" v={validation.correlation?.addresses ?? 0} />
                  <StatMini k="Source IPs" v={validation.correlation?.source_ips ?? 0} />
                  <StatMini k="ASNs" v={validation.correlation?.asns ?? 0} />
                  <StatMini k="Correlatable" v={validation.correlation?.correlatable_records ?? 0} />
                </div>
              ) : null}

              {validation?.validation?.errors?.length ? (
                <div className="banner banner-error" style={{ marginTop: 16 }}>
                  <h4>Validation errors</h4>
                  <ul>{validation.validation.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
                </div>
              ) : null}

              {validation?.validation?.warnings?.length ? (
                <div className="banner banner-synthetic" style={{ marginTop: 16 }}>
                  <h4>Warnings</h4>
                  <ul>{validation.validation.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                </div>
              ) : null}

              {validation?.canonical_columns ? (
                <>
                  <h3 className="small muted" style={{ margin: "18px 0 8px" }}>
                    Field coverage
                  </h3>
                  <div className="chips">
                    {validation.canonical_columns.map((col) => {
                      const present =
                        validation.validation.columns_present.includes(col);
                      return (
                        <span key={col} className={`cat cat-${
                          present ? "BLOCKCHAIN_CONTEXT" : "INSUFFICIENT_EVIDENCE"}`}>
                          {present ? "✓" : "✗"} {col}
                        </span>
                      );
                    })}
                  </div>
                </>
              ) : null}
            </div>
          </section>

          <div className="form-actions">
            <button className="btn" onClick={() => setStep("upload")}>← Back</button>
            <button className="btn btn-primary" onClick={() => setStep("analysis")}>
              Continue
            </button>
          </div>
        </>
      )}

      {/* STEP 4 - the honest one */}
      {step === "analysis" && (
        <section className="panel">
          <div className="panel-head">
            <h2>Analysis</h2>
            <span className={`status-badge status-${
              run?.status === "COMPLETE" ? "active" : "draft"}`}>
              {run?.status ?? "NOT_RUN"}
            </span>
          </div>
          <div className="panel-body">
            <div className="banner banner-synthetic" style={{ marginTop: 0 }}>
              <h4>This dataset has not been scored</h4>
              <p>{run?.meaning ??
                "The upload was received, parsed and validated. Producing risk " +
                "requires an offline pipeline run, not a request."}</p>
            </div>

            <div className="analysis-steps">
              <div className="analysis-step done">
                ✓ Data ingestion — {validation?.validation?.rows_read ?? 0} rows parsed
              </div>
              <div className="analysis-step done">
                ✓ Schema validation — {validation?.validation?.rows_valid ?? 0} valid records
              </div>
              <div className="analysis-step done">
                ✓ Stored with this investigation — sha256 {dataset?.sha256.slice(0, 16)}…
              </div>
              <div className="analysis-step pending">○ Entity clustering — requires offline pipeline</div>
              <div className="analysis-step pending">○ ML risk scoring — requires offline pipeline</div>
              <div className="analysis-step pending">○ Alert generation — requires offline pipeline</div>
            </div>

            <dl className="kv" style={{ marginTop: 18 }}>
              <dt>Analysis run status</dt><dd className="mono">{run?.status ?? "NOT_RUN"}</dd>
              <dt>Run fingerprint</dt>
              <dd className="faint">
                {run?.run_fingerprint ?? "none — no run has produced results for this dataset"}
              </dd>
              <dt>Alerts from this dataset</dt><dd className="faint">none</dd>
            </dl>

            <div style={{ marginTop: 16 }}>
              <span className="filter-label">Offline pipeline command</span>
              <pre className="cmd">{run?.command ??
                'make run ARGS="phase6-dataset" && make run ARGS="phase7-alerts"'}</pre>
            </div>

            <div className="form-actions" style={{ marginTop: 24 }}>
              <button className="btn btn-primary" onClick={openInvestigation}>
                Open investigation workspace →
              </button>
            </div>
          </div>
        </section>
      )}
    </>
  );
}

function StatMini({ k, v }: { k: string; v: number }) {
  return (
    <div className="stat-mini">
      <span className="stat-mini-v">{v.toLocaleString()}</span>
      <span className="stat-mini-k">{k}</span>
    </div>
  );
}
