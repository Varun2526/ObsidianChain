/**
 * The uploaded-run results panel keeps three kinds of claim apart: the
 * model's holdout result, this run's unverified alerts, and evidence classes.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as api from "../api/console";
import type { RunResults } from "../api/types";
import { RunResultsPanel } from "../components/forensics/RunResultsPanel";

const RESULTS: RunResults = {
  run_id: "run_1", run_fingerprint: "abc", created_at: null, input_sha256: "f".repeat(64),
  ml_status: "SCORED",
  model: { version: "ps_native_v5", feature_schema_version: "ps_native_features/5",
           holdout_result: { nap: 0.54752, "P@100": 1.0, ece: 0.0096, sha256: "x" },
           holdout_result_type: "HOLDOUT (model-level, t42-49 of Elliptic++; not this run)" },
  run_result_type: "PRODUCTION RUN - performance unverified until labels arrive",
  monitoring_alerts: [{ severity: "INFO", code: "PERFORMANCE_UNVERIFIED_UNTIL_LABELS", detail: "run model health later" }],
  drift_relative_to_development: "WITHIN_BASELINE",
  total_alerts: 1,
  stages: [],
  alerts: [{
    alert_id: "alert_c1", cluster_id: "c1", primary_address: "1LeadAddressExample0000", member_count: 2,
    fused_risk_score: 0.91, severity: "CRITICAL", rank: 1, summary: { corroborating_evidence_lines: 2 },
    explanation_statement: "associations learned from labelled history, not causes",
    evidence: [
      { category: "MODEL_SIGNAL", evidence_class: "MODEL", signal_name: "m", status: "PRESENT", score: 0.9, explanation: "model said so" },
      { category: "PATTERN_CONTEXT", evidence_class: "RULE", signal_name: "p", status: "PRESENT", score: 0.85, explanation: "peeling chain depth 4" },
      { category: "NETWORK_CONTEXT", evidence_class: "NETWORK", signal_name: "n", status: "NO_EVIDENCE", score: 0, explanation: "hidden" },
    ],
  }],
};

afterEach(() => vi.restoreAllMocks());

describe("RunResultsPanel", () => {
  it("labels the holdout result as the model's, not this run's", async () => {
    vi.spyOn(api, "getRunResults").mockResolvedValue(RESULTS);
    render(<RunResultsPanel investigationId="inv" runId="run_1" />);
    await waitFor(() => expect(screen.getByText("ps_native_v5")).toBeInTheDocument());
    expect(screen.getByText(/not this run/)).toBeInTheDocument();
    expect(screen.getByText(/unverified until labels arrive/)).toBeInTheDocument();
    expect(screen.getByText("0.548")).toBeInTheDocument();
  });

  it("shows evidence by class and hides absent evidence", async () => {
    vi.spyOn(api, "getRunResults").mockResolvedValue(RESULTS);
    render(<RunResultsPanel investigationId="inv" runId="run_1" />);
    await userEvent.click(await screen.findByRole("button", { name: "Why" }));
    expect(screen.getByText(/Model \(learned association\):/)).toBeInTheDocument();
    expect(screen.getByText(/Rule:/)).toBeInTheDocument();
    expect(screen.queryByText("hidden")).not.toBeInTheDocument();
    expect(screen.getByText(/not causes/)).toBeInTheDocument();
  });
});
