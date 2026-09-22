/**
 * New Investigation Wizard (5 Steps)
 * 
 * 1. Details: Name, description, priority
 * 2. Data Source: Dual dropzones (Blockchain Ledger & P2P Network observations) or unified file
 * 3. Forensic Validation: Structural validation checklist & field coverage
 * 4. Analysis Configuration: Standard Investigation profile with toggles + Collapsible Advanced Config
 * 5. Live Backend-Driven 17-Stage Execution: Real polling from /progress with stage tracker
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import * as api from "../../api/console";
import type {
  AnalysisRun,
  IngestResult,
  Investigation,
  RunProgressResponse,
  UploadedDataset,
} from "../../api/types";
import { ErrorState } from "../../components/ui/ErrorState";

type Step = "create" | "upload" | "validation" | "config" | "execution";

const ACCEPTED = ".csv,.json,.xml";
const STEPS: Step[] = ["create", "upload", "validation", "config", "execution"];

const CANONICAL_STAGES = [
  "Ingestion",
  "Validation & Deduplication",
  "GeoIP / ASN",
  "Blockchain Analysis",
  "Network Analysis",
  "Cross-Layer Correlation",
  "Transaction Graph",
  "Entity Clustering",
  "Feature Generation",
  "ML Detection",
  "Anomaly Detection",
  "Pattern Detection",
  "Evidence Fusion",
  "Alert Ranking",
  "Explainability",
  "Investigation Graph",
  "Report / Integrity",
];

const STEP_META: Record<Step, { num: string; title: string; doing: string; next: string }> = {
  create: {
    num: "01",
    title: "CASE",
    doing: "Define case parameters, title, and investigative hypothesis",
    next: "Next: Ingest Bitcoin transaction ledger and network telemetry dataset",
  },
  upload: {
    num: "02",
    title: "DATA",
    doing: "Upload Bitcoin transaction records and optional P2P network observations",
    next: "Next: Validate dataset schema integrity and cross-layer correlation readiness",
  },
  validation: {
    num: "03",
    title: "VALIDATE",
    doing: "Review data quality, canonical field coverage, and network observation readiness",
    next: "Next: Choose analysis profile and analytical detection modules",
  },
  config: {
    num: "04",
    title: "CONFIGURE",
    doing: "Choose analysis profile and analytical detection modules",
    next: "Next: Execute the 17-stage analytical pipeline offline",
  },
  execution: {
    num: "05",
    title: "ANALYZE",
    doing: "Executing the 17-stage forensic analytical pipeline",
    next: "Next: Open case workspace to inspect priority alerts and evidence",
  },
};

export function NewInvestigation() {
  const navigate = useNavigate();
  const inputPrimary = useRef<HTMLInputElement>(null);
  const inputNetwork = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<Step>("create");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [investigation, setInvestigation] = useState<Investigation | null>(null);

  const [busy, setBusy] = useState(false);
  const [draggingPrimary, setDraggingPrimary] = useState(false);
  const [draggingNetwork, setDraggingNetwork] = useState(false);
  const [dataset, setDataset] = useState<UploadedDataset | null>(null);
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [error, setError] = useState<Error | null>(null);

  // Configuration state
  const [profile, setProfile] = useState<string>("standard");
  const [modBlockchain, setModBlockchain] = useState(true);
  const [modNetwork, setModNetwork] = useState(true);
  const [modMLRisk, setModMLRisk] = useState(true);
  const [modAnomaly, setModAnomaly] = useState(true);
  const [modPatterns, setModPatterns] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [sensitivity, setSensitivity] = useState("0.75");
  const [modelFamily, setModelFamily] = useState("PS_NATIVE_LIGHTGBM");

  // Execution & Progress state
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState<RunProgressResponse | null>(null);

  const validation = dataset?.validation as IngestResult | undefined;

  const handleCreate = async () => {
    setBusy(true);
    setError(null);
    try {
      const inv = await api.createInvestigation(name, description);
      setInvestigation(inv);
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

  const handleStartAnalysis = async () => {
    if (!investigation || !dataset) return;
    setBusy(true);
    setError(null);
    setStep("execution");

    try {
      const resp = await api.runDatasetAnalysis(investigation.id, dataset.id, {
        profile,
        modules: {
          blockchain: modBlockchain,
          network: modNetwork,
          ml_risk: modMLRisk,
          anomaly: modAnomaly,
          patterns: modPatterns,
        },
        advanced: {
          sensitivity: parseFloat(sensitivity),
          model_family: modelFamily,
        },
      });

      setActiveRunId(resp.run_id);
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error(String(cause)));
      setBusy(false);
    }
  };

  // Poll backend run progress every 600ms
  useEffect(() => {
    if (!investigation || !activeRunId || step !== "execution") return;

    let stopped = false;
    const interval = setInterval(async () => {
      try {
        const data = await api.getRunProgress(investigation.id, activeRunId);
        if (stopped) return;
        setProgress(data);

        if (data.status === "COMPLETE" || data.status === "FAILED") {
          clearInterval(interval);
          setBusy(false);
        }
      } catch {
        // Continue polling
      }
    }, 600);

    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, [investigation, activeRunId, step]);

  const openInvestigation = () => {
    if (investigation) navigate(`/inv/${investigation.id}`);
  };

  return (
    <>
      <div className="page-header" style={{ marginBottom: 16 }}>
        <div>
          <h1>New Investigation</h1>
          <p className="muted" style={{ margin: 0 }}>
            {STEP_META[step].num} {STEP_META[step].title} — {STEP_META[step].doing}
          </p>
          <span className="small faint">{STEP_META[step].next}</span>
        </div>
      </div>

      <div className="step-progress" style={{ marginBottom: 24 }}>
        {STEPS.map((s, i) => {
          const isDone = STEPS.indexOf(step) > i;
          const isActive = step === s;
          const meta = STEP_META[s];
          return (
            <div
              key={s}
              className={`step-dot${isActive ? " active" : ""}${isDone ? " done" : ""}`}
            >
              <span className="step-dot-n">{meta.num}</span>
              <span className="step-dot-label">{meta.title}</span>
            </div>
          );
        })}
      </div>

      {/* STEP 1: CREATE SCOPE */}
      {step === "create" && (
        <section className="panel">
          <div className="panel-head"><h2>Investigation scope & case metadata</h2></div>
          <div className="panel-body">
            <div className="form-group">
              <label htmlFor="inv-name">Investigation name</label>
              <input
                id="inv-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Operation Silk Cascade — Bitcoin Financial Flow Analysis"
              />
            </div>
            <div className="form-group">
              <label htmlFor="inv-desc">Description / Investigative Hypothesis</label>
              <textarea
                id="inv-desc"
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Detailed rationale, initial suspect addresses, observation timeframes, or related case references…"
              />
            </div>
            {error ? <ErrorState error={error} /> : null}
            <div className="form-actions">
              <button
                className="btn btn-primary"
                onClick={handleCreate}
                disabled={!name.trim() || busy}
              >
                {busy ? "Initializing Case Workspace…" : "Create investigation"}
              </button>
            </div>
          </div>
        </section>
      )}

      {/* STEP 2: DUAL DROPZONES */}
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
              Upload Bitcoin forensic captures conforming to the canonical PS-format.
              You may upload a combined archive or separate blockchain ledger records and P2P network observations.
              All files are validated offline, stamped with immutable SHA-256 hashes, and stored in the case repository.
            </p>

            <div style={{ padding: "12px 16px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6, marginTop: 14, fontSize: 13, lineHeight: 1.5 }}>
              <strong style={{ color: "var(--cyan)" }}>Dataset Format Guidance:</strong> If using a 14-column <strong>Unified Dataset</strong> (like <code>canonical_acceptance_capture.csv</code>), it already bundles blockchain ledger transactions and P2P network propagation telemetry together. Upload it into <strong>Box 1</strong> — Box 2 is optional and only needed if you have split, separate log files.
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
              {/* Primary / Combined Dropzone */}
              <div>
                <span className="small muted" style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>
                  1. Blockchain Ledger Data (or Unified Dataset)
                </span>
                <div
                  className={`dropzone${draggingPrimary ? " active" : ""}`}
                  onDragOver={(e) => { e.preventDefault(); setDraggingPrimary(true); }}
                  onDragLeave={() => setDraggingPrimary(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDraggingPrimary(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) void handleFile(f);
                  }}
                  onClick={() => inputPrimary.current?.click()}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => { if (e.key === "Enter") inputPrimary.current?.click(); }}
                >
                  <strong>{busy ? "Parsing & Validating…" : "Drop Ledger CSV/JSON/XML"}</strong>
                  <span className="small faint">CSV · JSON · XML</span>
                  <input
                    ref={inputPrimary}
                    type="file"
                    accept={ACCEPTED}
                    hidden
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void handleFile(f);
                    }}
                  />
                </div>
              </div>

              {/* Network Observations Dropzone */}
              <div>
                <span className="small muted" style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>
                  2. P2P Network Propagation Data (Optional)
                </span>
                <div
                  className={`dropzone${draggingNetwork ? " active" : ""}`}
                  onDragOver={(e) => { e.preventDefault(); setDraggingNetwork(true); }}
                  onDragLeave={() => setDraggingNetwork(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDraggingNetwork(false);
                    const f = e.dataTransfer.files?.[0];
                    if (f) void handleFile(f);
                  }}
                  onClick={() => inputNetwork.current?.click()}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => { if (e.key === "Enter") inputNetwork.current?.click(); }}
                >
                  <strong>{busy ? "Parsing & Validating…" : "Drop P2P Network Logs"}</strong>
                  <span className="small faint">Source IPs, Timestamps, Peer Messages, ASNs</span>
                  <input
                    ref={inputNetwork}
                    type="file"
                    accept={ACCEPTED}
                    hidden
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void handleFile(f);
                    }}
                  />
                </div>
              </div>
            </div>

            {error ? <ErrorState error={error} /> : null}
          </div>
        </section>
      )}

      {/* STEP 3: FORENSIC VALIDATION */}
      {step === "validation" && dataset && (
        <>
          <section className="panel">
            <div className="panel-head">
              <h2>Data validation</h2>
              <span className={`status-badge status-${
                dataset.status === "VALIDATED" ? "active" : "closed"}`}>
                DATASET STATUS: {dataset.status === "VALIDATED" ? (validation?.validation?.warnings?.length ? "WARNING" : "VALID") : "INVALID"}
              </span>
              <span className="small muted">
                {dataset.filename} · {dataset.format.toUpperCase()}
              </span>
            </div>
            <div className="panel-body">
              <dl className="kv" style={{ marginBottom: 16 }}>
                <dt>Stored SHA-256</dt><dd className="mono">{dataset.sha256}</dd>
                <dt>Archive Footprint</dt><dd>{dataset.size_bytes.toLocaleString()} bytes</dd>
              </dl>

              {validation?.validation ? (
                <div className="card-grid-4" style={{ marginBottom: 16 }}>
                  <StatMini k="Records" v={validation.validation.rows_read} />
                  <StatMini k="Valid" v={validation.validation.rows_valid} />
                  <StatMini k="Transactions" v={validation.correlation?.transactions ?? 0} />
                  <StatMini k="Correlatable" v={validation.correlation?.correlatable_records ?? 0} />
                </div>
              ) : null}

              {/* 4 Clean Metric Cards */}
              <div className="card-grid-4" style={{ marginBottom: 20 }}>
                <div className="stat-card">
                  <span className="stat-card-value" style={{ color: "var(--model)" }}>
                    {validation?.validation ? `${validation.validation.columns_present.length} / 14` : "14 / 14"}
                  </span>
                  <span className="stat-card-label">FIELD COVERAGE</span>
                  <span className="small faint">
                    {validation?.validation ? `${Math.round((validation.validation.columns_present.length / 14) * 100)}% canonical fields` : "Canonical PS schema"}
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-card-value" style={{ color: "var(--cyan)" }}>
                    {(validation?.correlation?.transactions ?? 0) > 0 ? "READY" : "STANDALONE"}
                  </span>
                  <span className="stat-card-label">CORRELATION READINESS</span>
                  <span className="small faint">
                    {validation?.correlation?.transactions ?? 0} candidate transactions
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-card-value" style={{ color: (validation?.correlation?.source_ips ?? 0) > 0 ? "var(--network)" : "var(--text-dim)" }}>
                    {(validation?.correlation?.source_ips ?? 0) > 0 ? `${validation?.correlation?.source_ips} PEERS` : "NONE"}
                  </span>
                  <span className="stat-card-label">NETWORK COVERAGE</span>
                  <span className="small faint">
                    {validation?.correlation?.asns ?? 0} ASNs / passive monitors
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-card-value" style={{ color: validation?.validation?.rows_rejected === 0 ? "var(--model)" : "var(--high)" }}>
                    {validation?.validation?.rows_read ? `${Math.round((validation.validation.rows_valid / validation.validation.rows_read) * 100)}%` : "100%"}
                  </span>
                  <span className="stat-card-label">DATA QUALITY</span>
                  <span className="small faint">
                    {validation?.validation?.rows_valid ?? 0} valid records
                  </span>
                </div>
              </div>

              {/* Detected Fields Compact Summary */}
              <div style={{ padding: "14px 16px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6, marginBottom: 16 }}>
                <strong style={{ display: "block", marginBottom: 8, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Detected Forensic Fields
                </strong>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {["txid", "input_addresses", "output_addresses", "input_amounts", "output_amounts", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "fee", "script_type", "geo_country", "asn"].map((col) => {
                    const present = !validation?.validation || validation.validation.columns_present.includes(col);
                    return (
                      <span
                        key={col}
                        className="mono small"
                        style={{
                          padding: "3px 8px",
                          borderRadius: 4,
                          background: present ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.08)",
                          color: present ? "var(--model)" : "var(--text-dim)",
                          border: `1px solid ${present ? "rgba(16, 185, 129, 0.3)" : "var(--hairline)"}`,
                        }}
                      >
                        {present ? "✓" : "○"} {col}
                      </span>
                    );
                  })}
                </div>
              </div>

              {/* Collapsible Raw Technical Details */}
              <details style={{ marginTop: 12, padding: "10px 14px", background: "var(--bg-raised)", borderRadius: 6, border: "1px solid var(--hairline)" }}>
                <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: 13 }}>
                  View technical details (Raw schema & cryptographic fingerprint)
                </summary>
                <div style={{ marginTop: 12 }}>
                  <dl className="kv" style={{ marginBottom: 12 }}>
                    <dt>Rows Read</dt><dd>{validation?.validation?.rows_read ?? 0}</dd>
                    <dt>Duplicates Filtered</dt><dd>{validation?.validation?.exact_duplicates_rejected ?? 0}</dd>
                  </dl>
                  {validation?.validation?.warnings?.length ? (
                    <div className="banner banner-synthetic" style={{ marginTop: 10 }}>
                      <h4 style={{ margin: "0 0 4px" }}>Validation Warnings</h4>
                      <ul style={{ margin: 0, paddingLeft: 18 }}>
                        {validation.validation.warnings.map((w, i) => <li key={i}>{w}</li>)}
                      </ul>
                    </div>
                  ) : null}
                  {validation?.validation?.errors?.length ? (
                    <div className="banner banner-error" style={{ marginTop: 10 }}>
                      <h4 style={{ margin: "0 0 4px" }}>Validation Errors</h4>
                      <ul style={{ margin: 0, paddingLeft: 18 }}>
                        {validation.validation.errors.map((e, i) => <li key={i}>{e}</li>)}
                      </ul>
                    </div>
                  ) : null}
                </div>
              </details>
            </div>
          </section>

          <div className="form-actions" style={{ marginTop: 18 }}>
            <button className="btn" onClick={() => setStep("upload")}>← Back to Upload</button>
            <button
              className="btn btn-primary"
              onClick={() => setStep("config")}
              disabled={dataset.status === "REJECTED" || (validation?.validation?.errors && validation.validation.errors.length > 0)}
            >
              Continue
            </button>
          </div>
        </>
      )}

      {/* STEP 4: CHOOSE ANALYSIS PROFILE */}
      {step === "config" && (
        <section className="panel">
          <div className="panel-head">
            <h2>Choose Analysis Profile</h2>
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

            <div className="form-group" style={{ maxWidth: 500, margin: "18px 0" }}>
              <label htmlFor="analysis-profile" style={{ fontSize: 13, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Analysis Profile
              </label>
              <select
                id="analysis-profile"
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                style={{ fontSize: 14, padding: "8px 12px" }}
              >
                <option value="standard">STANDARD INVESTIGATION (Recommended)</option>
                <option value="deep_forensic">DEEP FORENSIC & PATTERN TRACING</option>
                <option value="triage_rapid">RAPID TRIAGE & PRIORITY ALERTS</option>
              </select>
            </div>

            <div style={{ padding: "14px 16px", background: "var(--bg)", border: "1px solid var(--hairline)", borderRadius: 6, marginBottom: 20 }}>
              <strong style={{ display: "block", color: "var(--cyan)", marginBottom: 4 }}>
                STANDARD INVESTIGATION
              </strong>
              <p className="small muted" style={{ margin: 0, lineHeight: 1.5 }}>
                Runs the standard ObsidianChain detection, anomaly, pattern and evidence pipeline. Executes all 17 offline stages to derive entity clusters, feature vectors, supervised LightGBM risk scores, and multi-layer fused alerts.
              </p>
            </div>

            <div>
              <h3 className="small muted" style={{ marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Analytical Detection Modules
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={modBlockchain}
                    onChange={(e) => setModBlockchain(e.target.checked)}
                  />
                  <span>
                    <strong>Blockchain Graph & Clustering:</strong> Address-Tx bipartite graph, Union-Find common-input clustering, change address heuristics.
                  </span>
                </label>

                <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={modNetwork}
                    onChange={(e) => setModNetwork(e.target.checked)}
                  />
                  <span>
                    <strong>Network Correlation:</strong> P2P propagation mapping, multi-observer diversity, GeoIP and ASN attribution.
                  </span>
                </label>

                <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={modMLRisk}
                    onChange={(e) => setModMLRisk(e.target.checked)}
                  />
                  <span>
                    <strong>PS-Native ML Risk Scoring:</strong> Calibrated LightGBM classifier derived from canonical PS feature space.
                  </span>
                </label>

                <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={modAnomaly}
                    onChange={(e) => setModAnomaly(e.target.checked)}
                  />
                  <span>
                    <strong>Unsupervised Anomaly Detection:</strong> Isolation Forest structural anomaly scoring for out-of-distribution entities.
                  </span>
                </label>

                <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={modPatterns}
                    onChange={(e) => setModPatterns(e.target.checked)}
                  />
                  <span>
                    <strong>Pattern Detection:</strong> Peeling chain heuristics, rapid fan-out, and coin mixing/CoinJoin fingerprinting.
                  </span>
                </label>
              </div>
            </div>

            {/* Advanced Collapsible Accordion (Senior Analysts) */}
            <div style={{ marginTop: 24, borderTop: "1px solid var(--hairline)", paddingTop: 16 }}>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{ background: "transparent", border: "none", color: "var(--text-dim)", padding: 0 }}
              >
                {showAdvanced ? "▼ Hide Advanced Configuration" : "▶ Advanced Configuration (Senior Analysts)"}
              </button>

              {showAdvanced && (
                <div style={{ marginTop: 14, padding: "14px 18px", background: "var(--bg-raised)", borderRadius: 6, border: "1px solid var(--hairline)" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <div className="form-group">
                      <label htmlFor="model-family">Supervised Risk Model Family</label>
                      <select
                        id="model-family"
                        value={modelFamily}
                        onChange={(e) => setModelFamily(e.target.value)}
                      >
                        <option value="PS_NATIVE_LIGHTGBM">PS-Native LightGBM (Canonical Schema)</option>
                        <option value="ENSEMBLE_HYBRID">Ensemble (LightGBM + Structural Heuristics)</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label htmlFor="sensitivity">Anomaly Prioritization Threshold</label>
                      <select
                        id="sensitivity"
                        value={sensitivity}
                        onChange={(e) => setSensitivity(e.target.value)}
                      >
                        <option value="0.90">Strict (Top 10% High Risk)</option>
                        <option value="0.75">Standard (Top 25% Elevated Risk)</option>
                        <option value="0.50">Broad (Top 50% Explanatory Scope)</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {error ? <ErrorState error={error} /> : null}

            <div className="form-actions" style={{ marginTop: 24 }}>
              <button className="btn" onClick={() => setStep("validation")}>← Back</button>
              <button
                className="btn btn-primary"
                onClick={handleStartAnalysis}
                disabled={busy}
              >
                {busy ? "Starting Analytical Engine…" : "Start Analysis →"}
              </button>
            </div>
          </div>
        </section>
      )}

      {/* STEP 5: LIVE BACKEND-DRIVEN 17-STAGE EXECUTION */}
      {step === "execution" && (
        <section className="panel">
          <div className="panel-head">
            <h2>Live 17-Stage Analytical Pipeline</h2>
            <span className={`status-badge status-${
              progress?.status === "COMPLETE" ? "active" :
              progress?.status === "FAILED" ? "closed" : "draft"}`}>
              {progress?.status ?? "INITIALIZING"}
            </span>
          </div>
          <div className="panel-body">
            {/* Progress Bar Header */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <div>
                  <strong style={{ fontSize: "1.1rem" }}>
                    Stage {progress?.current_stage ?? 1} of {progress?.total_stages ?? 17}:
                  </strong>
                  <span className="mono" style={{ marginLeft: 8, color: "var(--cyan)" }}>
                    {progress?.stage_name || CANONICAL_STAGES[0]}
                  </span>
                </div>
                <div className="mono" style={{ fontWeight: 600, fontSize: "1.2rem" }}>
                  {progress?.progress_pct ?? 5}%
                </div>
              </div>

              <div style={{ width: "100%", height: 8, background: "var(--bg-raised)", borderRadius: 4, overflow: "hidden", border: "1px solid var(--hairline)" }}>
                <div
                  style={{
                    width: `${progress?.progress_pct ?? 5}%`,
                    height: "100%",
                    background: progress?.status === "FAILED" ? "var(--critical)" : "var(--cyan)",
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
            </div>

            {/* Live Stages Checklist */}
            <div className="analysis-steps" style={{ maxHeight: 440, overflowY: "auto", paddingRight: 8 }}>
              {CANONICAL_STAGES.map((stageName, idx) => {
                const stageNum = idx + 1;
                const stageNumStr = String(stageNum).padStart(2, "0");
                const stagesList = progress?.stages || [];
                const completed = stagesList.find((s) => s.stage_number === stageNum && (s.status === "COMPLETED" || (s.status as string) === "COMPLETE" || (s.status as string) === "SUCCESS"));
                const running = (progress?.current_stage === stageNum && progress?.status === "RUNNING");
                const failed = (progress?.current_stage === stageNum && progress?.status === "FAILED");
                const isDone = Boolean(completed) || ((progress?.current_stage ?? 0) > stageNum) || progress?.status === "COMPLETE";

                return (
                  <div
                    key={stageName}
                    className={`analysis-step ${isDone ? "done" : running ? "active" : failed ? "failed" : "pending"}`}
                    style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 12px" }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span className="mono small" style={{ width: 22, color: "var(--muted)" }}>
                        {stageNumStr}
                      </span>
                      <span style={{ fontWeight: running || isDone ? 600 : 400 }}>
                        {stageName}
                      </span>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      {completed?.summary && (
                        <span className="small muted mono" style={{ fontSize: 11 }}>
                          {typeof completed.summary === "object"
                            ? Object.entries(completed.summary).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(" · ")
                            : String(completed.summary)}
                        </span>
                      )}
                      <span className="mono small" style={{
                        fontWeight: 600,
                        color: isDone ? "var(--model)" : running ? "var(--cyan)" : failed ? "var(--critical)" : "var(--muted)",
                      }}>
                        {isDone ? "✓ COMPLETED" : running ? "● RUNNING" : failed ? "✗ FAILED" : "○ PENDING"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Run Complete Summary Card */}
            {progress?.status === "COMPLETE" && (
              <div className="panel" style={{ marginTop: 24, background: "var(--bg-raised)", borderColor: "var(--border-strong)" }}>
                <div className="panel-body">
                  <h3 style={{ margin: "0 0 12px", color: "var(--model)" }}>ANALYSIS COMPLETE</h3>
                  <div className="card-grid-4">
                    <StatMini k="Ranked Alerts" v={Number(progress.summary?.alerts_generated ?? progress?.alerts_count ?? 0)} />
                    <StatMini k="Entities Clustered" v={Number(progress.summary?.entities_clustered ?? 0)} />
                    <StatMini k="P2P Correlations" v={Number(progress.summary?.correlations_identified ?? 0)} />
                    <div className="stat-mini">
                      <span className="stat-mini-v mono" style={{ fontSize: "0.95rem" }}>
                        {String(progress.summary?.run_fingerprint || activeRunId || "run_fused").slice(0, 12)}…
                      </span>
                      <span className="stat-mini-k">Run Fingerprint</span>
                    </div>
                  </div>

                  <p className="muted small" style={{ marginTop: 14, marginBottom: 0 }}>
                    All analytical artifacts have been cryptographically signed and bound to this investigation.
                    Casework decisions, graph exploration, and priority alert triage are now available.
                  </p>
                </div>
              </div>
            )}

            {progress?.status === "FAILED" && (
              <div className="banner banner-error" style={{ marginTop: 20 }}>
                <h4>PIPELINE EXECUTION FAILED</h4>
                <p>{progress.error || "An analytical stage encountered an error during offline processing."}</p>
              </div>
            )}

            <div className="form-actions" style={{ marginTop: 24 }}>
              {progress?.status === "COMPLETE" ? (
                <button className="btn btn-primary" onClick={openInvestigation} style={{ padding: "10px 24px", fontWeight: 700 }}>
                  OPEN INVESTIGATION →
                </button>
              ) : (
                <button className="btn" disabled>
                  {progress?.status === "FAILED" ? "Execution Halted" : "Processing Offline Stages…"}
                </button>
              )}
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
