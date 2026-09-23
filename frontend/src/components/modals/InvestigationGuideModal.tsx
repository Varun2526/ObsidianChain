import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Stepper, { Step } from "../ui/Stepper";

interface InvestigationGuideModalProps {
  open: boolean;
  onClose: () => void;
}

export function InvestigationGuideModal({ open, onClose }: InvestigationGuideModalProps) {
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (open) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const handleStartNew = () => {
    onClose();
    navigate("/investigations/new");
  };

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1100,
        padding: "16px",
      }}
    >
      <div
        className="modal-content"
        role="dialog" aria-modal="true" aria-label="Investigation workflow guide" onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--bg-canvas, #0c0e14)",
          border: "1px solid var(--border-strong, #2a2e3d)",
          borderRadius: "16px",
          width: "100%",
          maxWidth: "680px",
          maxHeight: "90vh",
          overflowY: "auto",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.8)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Modal Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "20px 24px",
            borderBottom: "1px solid var(--hairline, #1f2330)",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  padding: "2px 8px",
                  borderRadius: "4px",
                  background: "rgba(0, 240, 170, 0.12)",
                  color: "var(--cyan, #00f0aa)",
                  border: "1px solid rgba(0, 240, 170, 0.3)",
                }}
              >
                SIH-26146 STANDARD OPERATING PROCEDURE
              </span>
            </div>
            <h2 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 600 }}>
              Forensic Investigation Lifecycle
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--muted, #8b949e)",
              cursor: "pointer",
              fontSize: "13px",
              lineHeight: 1,
              padding: "4px 8px",
              borderRadius: "4px",
            }}
            title="Close (Esc)"
            aria-label="Close"
          >
            Close
          </button>
        </div>

        {/* Stepper Body */}
        <div style={{ padding: "16px 20px 24px" }}>
          <Stepper
            className="no-aspect"
            stepCircleContainerClassName="wide"
            initialStep={1}
            backButtonText="← Previous Step"
            nextButtonText="Next Step →"
            onFinalStepCompleted={handleStartNew}
          >
            {/* STEP 1 */}
            <Step>
              <div style={{ textAlign: "left" }}>
                <span className="mono small" style={{ color: "var(--blockchain, #38bdf8)" }}>
                  PHASE 01 · CASE PARAMETERS
                </span>
                <h3 style={{ margin: "4px 0 10px", fontSize: "1.15rem", fontWeight: 600 }}>
                  Case Scoping & Investigative Hypothesis
                </h3>
                <p className="muted small" style={{ lineHeight: 1.5, marginBottom: "16px" }}>
                  Every investigation in ObsidianChain establishes an isolated forensic workspace. Define your case parameters, hypothesis, and priority level before ingesting evidence.
                </p>
                <div
                  style={{
                    background: "var(--bg-raised, #161922)",
                    border: "1px solid var(--border-subtle, #232736)",
                    borderRadius: "8px",
                    padding: "12px 16px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Immutable Attribution:</strong> Bound directly to the authenticated station session</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Scoped Evidence Boundary:</strong> Zero leakage between independent case dockets</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Canonical Identifier:</strong> System assigns deterministic docket IDs (e.g. INV-2026-001)</span>
                  </div>
                </div>
              </div>
            </Step>

            {/* STEP 2 */}
            <Step>
              <div style={{ textAlign: "left" }}>
                <span className="mono small" style={{ color: "var(--blockchain, #38bdf8)" }}>
                  PHASE 02 · MULTI-LAYER INGESTION
                </span>
                <h3 style={{ margin: "4px 0 10px", fontSize: "1.15rem", fontWeight: 600 }}>
                  Dual Blockchain Ledger & P2P Telemetry Ingestion
                </h3>
                <p className="muted small" style={{ lineHeight: 1.5, marginBottom: "16px" }}>
                  Upload raw on-chain transaction records and optional P2P node network observations via unified or dual dropzones as mandated by SIH Problem Statement 26146.
                </p>
                <div
                  style={{
                    background: "var(--bg-raised, #161922)",
                    border: "1px solid var(--border-subtle, #232736)",
                    borderRadius: "8px",
                    padding: "12px 16px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Triple Format Support:</strong> First-class ingestion for CSV, JSON, and XML</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Dual Dropzones:</strong> Ingest ledger transactions alongside P2P network telemetry</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Content-Addressed Storage:</strong> Auto-computes cryptographic SHA-256 digest</span>
                  </div>
                </div>
              </div>
            </Step>

            {/* STEP 3 */}
            <Step>
              <div style={{ textAlign: "left" }}>
                <span className="mono small" style={{ color: "var(--blockchain, #38bdf8)" }}>
                  PHASE 03 · INTEGRITY VERIFICATION
                </span>
                <h3 style={{ margin: "4px 0 10px", fontSize: "1.15rem", fontWeight: 600 }}>
                  Forensic Validation & Deduplication
                </h3>
                <p className="muted small" style={{ lineHeight: 1.5, marginBottom: "16px" }}>
                  Every raw file is parsed through schema validation, record deduplication, and cross-layer correlation checks before entering the analytical pipeline.
                </p>
                <div
                  style={{
                    background: "var(--bg-raised, #161922)",
                    border: "1px solid var(--border-subtle, #232736)",
                    borderRadius: "8px",
                    padding: "12px 16px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Schema Validation:</strong> Checks required transaction, address, and timestamp fields</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Record Deduplication:</strong> Identifies and rejects duplicate transactions honestly</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Correlation Readiness:</strong> Verifies addresses vs IP/ASN correlation overlap</span>
                  </div>
                </div>
              </div>
            </Step>

            {/* STEP 4 */}
            <Step>
              <div style={{ textAlign: "left" }}>
                <span className="mono small" style={{ color: "var(--blockchain, #38bdf8)" }}>
                  PHASE 04 · OFFLINE ANALYTICAL ENGINE
                </span>
                <h3 style={{ margin: "4px 0 10px", fontSize: "1.15rem", fontWeight: 600 }}>
                  17-Stage Forensic Analytical Execution
                </h3>
                <p className="muted small" style={{ lineHeight: 1.5, marginBottom: "16px" }}>
                  Execute the offline 17-stage pipeline. Real-time progress updates trace graph traversal, PS-native ML detection, and multi-layer evidence fusion.
                </p>
                <div
                  style={{
                    background: "var(--bg-raised, #161922)",
                    border: "1px solid var(--border-subtle, #232736)",
                    borderRadius: "8px",
                    padding: "12px 16px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Graph Traversal & Clustering:</strong> UTXO transaction graphs & entity clustering</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>PS-Native ML + Anomaly:</strong> Supervised classification with Elliptic++ baseline</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Multi-Layer Fusion:</strong> Fuses on-chain, P2P network, and temporal heuristics</span>
                  </div>
                </div>
              </div>
            </Step>

            {/* STEP 5 */}
            <Step>
              <div style={{ textAlign: "left" }}>
                <span className="mono small" style={{ color: "var(--blockchain, #38bdf8)" }}>
                  PHASE 05 · TRIAGE & GOVERNANCE
                </span>
                <h3 style={{ margin: "4px 0 10px", fontSize: "1.15rem", fontWeight: 600 }}>
                  Ranked Alerts, SHAP Explainability & Review
                </h3>
                <p className="muted small" style={{ lineHeight: 1.5, marginBottom: "16px" }}>
                  Inspect prioritized risk alerts with full SHAP attribution, explore multi-layer network graphs, and conduct formal QA sign-off into the append-only audit trail.
                </p>
                <div
                  style={{
                    background: "var(--bg-raised, #161922)",
                    border: "1px solid var(--border-subtle, #232736)",
                    borderRadius: "8px",
                    padding: "12px 16px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>SHAP Feature Explainability:</strong> Clear rationale for every flagged entity</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Multi-Layer Graph Inspection:</strong> Seamlessly toggle Chain, Network, and Fused layers</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ color: "var(--cyan, #00f0aa)", fontSize: "14px" }}>✓</span>
                    <span className="small"><strong>Formal Review Decisions:</strong> Record signed dispositions into append-only audit trail</span>
                  </div>
                </div>
              </div>
            </Step>
          </Stepper>
        </div>
      </div>
    </div>
  );
}
