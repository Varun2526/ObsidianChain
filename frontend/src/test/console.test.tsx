/**
 * The application layer, in the browser.
 *
 * What these cover that the analytical suite does not: that identity comes
 * from the server and not from storage, that a case-scoped route refuses to
 * render a workspace it has not loaded, that the model's assessment and the
 * investigator's are drawn as separate things, and that an upload is never
 * presented as having been scored.
 *
 * Fetch is stubbed per-path rather than globally, because these components
 * make several calls and a single blanket response is how a test ends up
 * asserting against a shape the backend never returns.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchAlerts } from "../api/client";
import * as api from "../api/console";
import { AuthProvider, useAuth } from "../store/auth";
import { CaseGate } from "../store/investigation";
import { LoginPage } from "../pages/LoginPage";
import { AlertDetailPage } from "../pages/investigation/AlertDetail";
import { NewInvestigation } from "../pages/investigation/NewInvestigation";
import { InvestigationOverview } from "../pages/investigation/InvestigationOverview";
import { ReportPage } from "../pages/investigation/ReportPage";
import { SeparationEvidencePanel } from "../components/forensics/SeparationEvidencePanel";
import { DispositionBadge, StaleRunBanner } from "../components/layout/CaseChrome";
import detailFixture from "../../fixtures/alert_detail.json";
import type { AlertDetail } from "../api/types";

const detail = detailFixture as unknown as AlertDetail;

const IDENTITY = {
  user: {
    id: "usr_1", username: "alice", display_name: "Alice A",
    role: "INVESTIGATOR", active: true, created_at: "2026-09-01T00:00:00Z",
  },
  capabilities: [
    "create_investigation", "set_disposition", "write_note",
    "reference_alert", "assign_alert", "create_report",
    "change_investigation_status", "upload_dataset",
  ],
};

const CASE = {
  id: "inv_abc", case_number: 1, case_label: "OC-0001",
  name: "Case A", description: "A description", owner_id: "usr_1",
  owner: { id: "usr_1", username: "alice", display_name: "Alice A" },
  status: "ACTIVE", bound_run_fingerprint: "043ea584e99daf99",
  created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-02T00:00:00Z",
  closed_at: null,
  analytical_run: {
    bound_run_fingerprint: "043ea584e99daf99",
    current_artifact_run: "043ea584e99daf99",
    status: "CURRENT",
    meaning: "The run this case's referenced alerts were taken from.",
  },
  datasets: [],
  summary: {
    alerts_referenced: 1, outstanding: 0, notes: 1,
    dispositions_by_state: {
      NEW: 0, TRIAGED: 0, IN_REVIEW: 1, CONFIRMED: 0, DISMISSED: 0, ESCALATED: 0,
    },
  },
};

const DISPOSITION = {
  id: "dsp_1", state: "IN_REVIEW", rationale: "checking peel depth",
  decided_by: "usr_1", decided_by_username: "alice",
  decided_by_display_name: "Alice A", decided_at: "2026-09-02T10:00:00Z",
  superseded_by: null, active: true,
};

/**
 * Route fetch by path. An unmatched path is a test bug, so it throws rather
 * than returning an empty object that would silently become "no data".
 */
function routeFetch(routes: Array<[RegExp, unknown, number?]>) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    void init;
    const url = String(input);
    for (const [pattern, body, status = 200] of routes) {
      if (pattern.test(url)) {
        return {
          ok: status >= 200 && status < 300,
          status,
          json: async () => body,
        } as Response;
      }
    }
    throw new Error(`unstubbed request: ${url}`);
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ---- identity comes from the server ------------------------------------

describe("authentication", () => {
  it("asks the server who it is on startup", async () => {
    const fetchMock = routeFetch([[/\/api\/auth\/me/, IDENTITY]]);
    vi.stubGlobal("fetch", fetchMock);
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("alice"));
    expect(fetchMock).toHaveBeenCalled();
  });

  it("sends the session cookie with every call", async () => {
    const fetchMock = routeFetch([[/\/api\/auth\/me/, IDENTITY]]);
    vi.stubGlobal("fetch", fetchMock);
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const init = fetchMock.mock.calls[0]![1];
    expect(init?.credentials).toBe("same-origin");
  });

  // "No identity in browser storage" is asserted at SOURCE level in
  // tests/test_frontend_offline.py rather than here: jsdom is configured
  // without a functional Storage, so a runtime assertion on localStorage
  // would pass whatever the code did, which is worse than no test at all.

  it("treats a 401 as logged out rather than as an error", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/api\/auth\/me/, { error: "authentication_required" }, 401],
    ]));
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() =>
      expect(screen.getByTestId("who")).toHaveTextContent("anonymous"));
  });

  it("surfaces the backend's own refusal on a bad password", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/api\/auth\/me/, { error: "authentication_required" }, 401],
      [/\/api\/auth\/login/,
       { error: "invalid_credentials", detail: "invalid username or password" },
       401],
    ]));
    render(<AuthProvider><LoginPage /></AuthProvider>);
    await userEvent.type(screen.getByLabelText("Username"), "alice");
    await userEvent.type(screen.getByLabelText("Password"), "wrong");
    await userEvent.click(screen.getByRole("button", { name: /sign in/i }));
    await waitFor(() =>
      expect(screen.getByText(/invalid username or password/i)).toBeInTheDocument());
  });

  it("sends the password to the backend rather than checking it locally", async () => {
    const fetchMock = routeFetch([
      [/\/api\/auth\/me/, { error: "authentication_required" }, 401],
      [/\/api\/auth\/login/, IDENTITY],
    ]);
    vi.stubGlobal("fetch", fetchMock);
    render(<AuthProvider><LoginPage /></AuthProvider>);
    await userEvent.type(screen.getByLabelText("Username"), "alice");
    await userEvent.type(screen.getByLabelText("Password"), "alice-password");
    await userEvent.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find((c) =>
        String(c[0]).includes("/auth/login"));
      expect(call).toBeTruthy();
      expect(JSON.parse(call![1]!.body as string)).toEqual({
        username: "alice", password: "alice-password",
      });
    });
  });
});

// ---- the analytical routes now need the session too --------------------

describe("analytical client", () => {
  it("sends the session cookie with alert requests", async () => {
    const fetchMock = routeFetch([[/\/api\/alerts/, { alerts: [] }]]);
    vi.stubGlobal("fetch", fetchMock);
    await fetchAlerts({ limit: 1 });
    const init = fetchMock.mock.calls[0]![1];
    expect(init?.credentials).toBe("same-origin");
  });

  it("signs the application out when the session has lapsed", async () => {
    // A lapsed session can be discovered by ANY call, including a
    // background one. Leaving the shell drawn with every panel failing
    // separately reads as a broken application rather than as being signed
    // out, so the discovery is broadcast once.
    vi.stubGlobal("fetch", routeFetch([
      [/\/api\/auth\/me/, IDENTITY],
      [/\/api\/alerts/, { error: "session_expired" }, 401],
    ]));
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("alice"));

    await fetchAlerts({ limit: 1 }).catch(() => {});
    await waitFor(() =>
      expect(screen.getByTestId("who")).toHaveTextContent("anonymous"));
  });

  it("does not sign the application out on a 403", async () => {
    // 403 means the session is perfectly valid and this user may not have
    // that particular thing. Logging them out over it would be wrong.
    vi.stubGlobal("fetch", routeFetch([
      [/\/api\/auth\/me/, IDENTITY],
      [/\/api\/investigations\//, { error: "access_denied" }, 403],
    ]));
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("alice"));

    await api.getInvestigation("inv_someone_else").catch(() => {});
    expect(screen.getByTestId("who")).toHaveTextContent("alice");
  });
});

function Probe() {
  const { identity, loading } = useAuth();
  return (
    <span data-testid="who">
      {loading ? "loading" : identity ? identity.user.username : "anonymous"}
    </span>
  );
}

// ---- a case-scoped route loads the case first --------------------------

describe("case-scoped routes", () => {
  function renderGate(status: number, body: unknown) {
    vi.stubGlobal("fetch", routeFetch([
      [/\/api\/investigations\/inv_/, body, status],
    ]));
    return render(
      <MemoryRouter>
        <CaseGate invId="inv_abc"><div>WORKSPACE</div></CaseGate>
      </MemoryRouter>,
    );
  }

  it("renders the page once the case loads", async () => {
    renderGate(200, CASE);
    await waitFor(() => expect(screen.getByText("WORKSPACE")).toBeInTheDocument());
  });

  it("shows a not-found rather than a workspace for an unknown case", async () => {
    renderGate(404, { error: "investigation_not_found", detail: "no such case" });
    await waitFor(() =>
      expect(screen.getByText("Investigation not found")).toBeInTheDocument());
    // The defect: /inv/ANYTHING/alerts used to render the global queue.
    expect(screen.queryByText("WORKSPACE")).not.toBeInTheDocument();
  });

  it("shows access denied rather than a workspace for someone else's case", async () => {
    renderGate(403, { error: "access_denied", detail: "belongs to another" });
    await waitFor(() =>
      expect(screen.getByText("Access denied")).toBeInTheDocument());
    expect(screen.queryByText("WORKSPACE")).not.toBeInTheDocument();
  });
});

// ---- the two assessments are never merged ------------------------------

describe("AlertDetailPage in a case", () => {
  const caseAlert = {
    alert_id: detail.alert_id,
    investigation_id: "inv_abc",
    analytical: { available: true, alert: detail },
    investigator: {
      reference: {
        alert_id: detail.alert_id, run_fingerprint: "043ea584e99daf99",
        added_by: "usr_1", added_at: "2026-09-02T09:00:00Z",
      },
      assignment: { assigned_to: null, assigned_at: null, assignee: null },
      disposition: DISPOSITION,
      disposition_history: [DISPOSITION],
      notes: [],
      states: ["NEW", "TRIAGED", "IN_REVIEW", "CONFIRMED", "DISMISSED", "ESCALATED"],
      meaning: "A disposition is an INVESTIGATOR'S decision about this alert.",
    },
  };

  beforeEach(() => {
    vi.mock("cytoscape", () => ({
      default: () => ({
        on: () => {}, ready: (fn: () => void) => fn(), fit: () => {},
        destroy: () => {},
      }),
    }));
  });

  async function renderCaseAlert() {
    vi.stubGlobal("fetch", routeFetch([
      [/\/auth\/me/, IDENTITY],
      [/\/users\/assignable/, { users: [] }],
      [/separation-evidence/, { error: "not_found" }, 404],
      [/\/api\/investigations\/inv_abc\/alerts\//, caseAlert],
    ]));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={[`/inv/inv_abc/alerts/${detail.alert_id}`]}>
          <Routes>
            <Route path="/inv/:invId/alerts/:alertId" element={<AlertDetailPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByText("Analytical assessment")).toBeInTheDocument());
  }

  it("labels the model's output and the investigator's separately", async () => {
    await renderCaseAlert();
    expect(screen.getByText("Analytical assessment")).toBeInTheDocument();
    expect(screen.getByText("Investigator assessment")).toBeInTheDocument();
  });

  it("says in the page that the two are different assessments", async () => {
    await renderCaseAlert();
    expect(
      screen.getByText(/Model output, read from the immutable artifact/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Separate from the model's output above/),
    ).toBeInTheDocument();
  });

  it("shows both the model's severity and the investigator's decision", async () => {
    await renderCaseAlert();
    expect(screen.getAllByText(detail.summary.severity).length).toBeGreaterThan(0);
    // Appears in the current badge AND in the decision history, which is
    // correct - the history shows where the decision stands and how it got
    // there.
    expect(screen.getAllByText("IN REVIEW").length).toBeGreaterThan(0);
  });

  it("shows the decision history rather than only the current state", async () => {
    await renderCaseAlert();
    expect(screen.getByText("Decision history")).toBeInTheDocument();
    expect(screen.getByText(/checking peel depth/)).toBeInTheDocument();
  });

  it("explains why the analytical half is missing instead of hiding it", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/auth\/me/, IDENTITY],
      [/\/users\/assignable/, { users: [] }],
      [/\/api\/investigations\/inv_abc\/alerts\//, {
        ...caseAlert,
        analytical: {
          available: false,
          reason: "STALE_REFERENCE",
          referenced_run: "043ea584e99daf99",
          current_artifact_run: "ffffffffffffffff",
          detail: "A determining input changed; the reference is kept and marked STALE.",
        },
      }],
    ]));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={[`/inv/inv_abc/alerts/${detail.alert_id}`]}>
          <Routes>
            <Route path="/inv/:invId/alerts/:alertId" element={<AlertDetailPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByText(/Analytical results unavailable/)).toBeInTheDocument());
    expect(screen.getByText(/STALE REFERENCE/)).toBeInTheDocument();
    // The decision survives the analytical half being unresolvable.
    expect(screen.getAllByText("IN REVIEW").length).toBeGreaterThan(0);
  });
});

// ---- upload is never presented as scored -------------------------------

describe("NewInvestigation", () => {
  const dataset = {
    id: "ds_1", investigation_id: "inv_abc", filename: "capture.csv",
    sha256: "a".repeat(64), size_bytes: 64, format: "csv",
    uploaded_by: "usr_1", uploaded_at: "2026-09-02T00:00:00Z",
    status: "VALIDATED",
    validation: {
      filename: "capture.csv", bytes: 64,
      validation: {
        ok: true, rows_read: 12, rows_valid: 12, rows_rejected: 0,
        source_format: "csv", columns_present: ["txid"], columns_missing: [],
        required_missing: [], errors: [], warnings: [],
      },
      correlation: {
        transactions: 12, addresses: 5, source_ips: 3, asns: 2,
        correlatable_records: 12,
      },
      preview: [],
      canonical_columns: ["txid", "src_ip"],
    },
  };
  const notRun = {
    id: "run_1", dataset_id: "ds_1", status: "NOT_RUN",
    run_fingerprint: null, produced_alerts: false,
    created_at: "2026-09-02T00:00:00Z", started_at: null,
    completed_at: null, error: null,
    meaning: "This dataset was received, parsed and validated. It has NOT been scored.",
    command: 'make run ARGS="phase6-dataset" && make run ARGS="phase7-alerts"',
  };

  async function uploadThrough() {
    vi.stubGlobal("fetch", routeFetch([
      [/\/auth\/me/, IDENTITY],
      [/\/api\/investigations\/inv_abc\/datasets/, { dataset, analysis_run: notRun }],
      [/\/api\/investigations$/, CASE],
    ]));
    render(
      <AuthProvider>
        <MemoryRouter><NewInvestigation /></MemoryRouter>
      </AuthProvider>,
    );
    await userEvent.type(screen.getByLabelText(/Investigation name/i), "Case A");
    await userEvent.click(screen.getByRole("button", { name: /create investigation/i }));
    await waitFor(() => expect(screen.getByText("Data source")).toBeInTheDocument());
    const file = new File(["txid\n1\n"], "capture.csv", { type: "text/csv" });
    await userEvent.upload(
      document.querySelector('input[type="file"]') as HTMLInputElement, file);
    await waitFor(() =>
      expect(screen.getByText("Data validation")).toBeInTheDocument());
  }

  it("reports what parsed, from the endpoint's own response", async () => {
    await uploadThrough();
    expect(screen.getByText("Records")).toBeInTheDocument();
    expect(screen.getByText("Valid")).toBeInTheDocument();
  });

  it("offers all three formats the PS names", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/auth\/me/, IDENTITY],
      [/\/api\/investigations$/, CASE],
    ]));
    render(<AuthProvider><MemoryRouter><NewInvestigation /></MemoryRouter></AuthProvider>);
    await userEvent.type(screen.getByLabelText(/Investigation name/i), "Case A");
    await userEvent.click(screen.getByRole("button", { name: /create investigation/i }));
    await waitFor(() =>
      expect(screen.getByText(/CSV · JSON · XML/)).toBeInTheDocument());
  });

  it("records the dataset's own hash rather than trusting the filename", async () => {
    await uploadThrough();
    expect(screen.getByText("Stored SHA-256")).toBeInTheDocument();
    expect(screen.getByText("a".repeat(64))).toBeInTheDocument();
  });

  it("states that the upload was NOT scored, and names the offline command", async () => {
    await uploadThrough();
    await userEvent.click(screen.getByRole("button", { name: /^Continue$/ }));
    await waitFor(() =>
      expect(screen.getByText("This dataset has not been scored")).toBeInTheDocument());
    expect(screen.getByText(/phase6-dataset/)).toBeInTheDocument();
  });

  it("never reports the global alert count as this case's result", async () => {
    await uploadThrough();
    await userEvent.click(screen.getByRole("button", { name: /^Continue$/ }));
    // NOT_RUN appears as the step's status badge and again in the run
    // detail list; both are the same honest statement.
    await waitFor(() =>
      expect(screen.getAllByText("NOT_RUN").length).toBeGreaterThan(0));
    // The defect: this step used to fetch /api/alerts and report "2,128
    // precomputed alerts" against a case whose data had never been scored.
    expect(screen.getByText(/Alerts from this dataset/)).toBeInTheDocument();
    expect(screen.getAllByText("none").length).toBeGreaterThan(0);
    expect(screen.queryByText(/2,128/)).not.toBeInTheDocument();
  });
});

// ---- the overview counts the case's own work ---------------------------

describe("InvestigationOverview", () => {
  it("reports referenced alerts, not the artifact's total", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/auth\/me/, IDENTITY],
      [/\/investigations\/inv_abc\/alerts/, {
        alerts: [], current_artifact_run: "043ea584e99daf99",
        stale_count: 0, summary: CASE.summary, meaning: "",
      }],
      [/\/investigations\/inv_abc/, CASE],
    ]));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/inv/inv_abc"]}>
          <Routes>
            <Route path="/inv/:invId" element={
              <CaseGate invId="inv_abc"><InvestigationOverview /></CaseGate>
            } />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByText("Alerts in this case")).toBeInTheDocument());
    expect(screen.getByText("Awaiting a decision")).toBeInTheDocument();
    expect(screen.getByText("Bound run")).toBeInTheDocument();
  });
});

// ---- stale references are shown, never dropped -------------------------

describe("stale run handling", () => {
  it("names both fingerprints and says nothing was re-pointed", () => {
    render(
      <StaleRunBanner bound="043ea584e99daf99" current="ffffffffffffffff" count={2} />,
    );
    expect(screen.getByText(/Analytical run changed/)).toBeInTheDocument();
    expect(screen.getByText("043ea584e99daf99")).toBeInTheDocument();
    expect(screen.getByText("ffffffffffffffff")).toBeInTheDocument();
    expect(
      screen.getByText(/Nothing has been removed or re-pointed/),
    ).toBeInTheDocument();
  });

  it("keeps a stale reference in the report instead of dropping it", async () => {
    const stale = {
      alert_id: "043ea584e99daf99:1", run_fingerprint: "043ea584e99daf99",
      added_by: "usr_1", added_by_username: "alice", added_at: "2026-09-02T00:00:00Z",
      assigned_to: null, assigned_to_username: null,
      assigned_to_display_name: null, assigned_at: null,
      disposition: DISPOSITION, stale: true,
      warning: "This alert was referenced against analytical run 043ea584e99daf99.",
    };
    vi.stubGlobal("fetch", routeFetch([
      [/\/auth\/me/, IDENTITY],
      [/\/investigations\/inv_abc\/report/, {
        case: CASE,
        analytical_run: {
          bound_run_fingerprint: "043ea584e99daf99",
          current_artifact_run: "ffffffffffffffff", status: "STALE",
        },
        report: {
          id: "rpt_1", investigation_id: "inv_abc", version: 1,
          title: "Report A", executive_summary: "Summary A", content: "Body A",
          run_fingerprint: "043ea584e99daf99", generated_by: "usr_1",
          generated_at: "2026-09-02T00:00:00Z", content_sha256: "b".repeat(64),
          status: "DRAFT", finalised_by: null, finalised_at: null,
        },
        versions: [], alert_references: [stale], stale_references: [stale],
        unverifiable_references: [], notes: [], summary: CASE.summary,
        disposition_meaning: "A disposition is an INVESTIGATOR'S decision.",
        separation_note: "Analytical results are read from immutable artifacts.",
        may_finalise: false,
      }],
    ]));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/inv/inv_abc/report"]}>
          <Routes>
            <Route path="/inv/:invId/report" element={<ReportPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByText("Stale alert references")).toBeInTheDocument());
    // The defect: `.catch(() => null)` removed these silently.
    expect(screen.getAllByText("043ea584e99daf99:1").length).toBeGreaterThan(0);
  });

  it("persists the executive summary it was given", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/auth\/me/, IDENTITY],
      [/\/investigations\/inv_abc\/report/, {
        case: CASE,
        analytical_run: CASE.analytical_run,
        report: {
          id: "rpt_1", investigation_id: "inv_abc", version: 2,
          title: "Report A", executive_summary: "The summary that used to vanish",
          content: "", run_fingerprint: "043ea584e99daf99",
          generated_by: "usr_1", generated_at: "2026-09-02T00:00:00Z",
          content_sha256: "c".repeat(64), status: "DRAFT",
          finalised_by: null, finalised_at: null,
        },
        versions: [], alert_references: [], stale_references: [],
        unverifiable_references: [], notes: [], summary: CASE.summary,
        disposition_meaning: "", separation_note: "", may_finalise: false,
      }],
    ]));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/inv/inv_abc/report"]}>
          <Routes>
            <Route path="/inv/:invId/report" element={<ReportPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByDisplayValue("The summary that used to vanish"))
        .toBeInTheDocument());
    expect(screen.getByText("c".repeat(64))).toBeInTheDocument();
  });
});

// ---- separation evidence semantics -------------------------------------

describe("SeparationEvidencePanel", () => {
  const payload = {
    alert_id: "043ea584e99daf99:1", cluster_id: 1,
    alert_run_fingerprint: "043ea584e99daf99",
    evidence_run_fingerprint: "cbd6248beea8f700",
    statement: "These are the network-separation records for the proposed merges.",
    join_basis: "Funnel rows are selected by address code.",
    proposed_merges_total: 2,
    verdicts: { NO_EVIDENCE: 1, NOT_SEPARATED: 1 },
    reason_codes: { INSUFFICIENT_POOLED: 1, NOT_SIGNIFICANT: 1 },
    separated_count: 0,
    cannot_link_meaning:
      "A SEPARATED verdict carries CANNOT-LINK. The network layer never emits MUST-LINK.",
    not_separated_meaning:
      "NOT_SEPARATED is not evidence that these two candidate components are the same entity.",
    verdict_scope: "This verdict describes the network evidence for THIS union.",
    verdict_definition: "Direct return values of separation_evidence().",
    frozen_run_limitation: "SEPARATED = 0 across this run is a property of this frozen dataset.",
    reason_code_catalogue: {}, unreachable_reason_codes: {},
    rows_shown: 2, rows_withheld: 0,
    rows: [
      { evidence_id: "cbd6248beea8f700:0", edge_index: 0, node_a: 1, node_b: 2,
        verdict: "NO_EVIDENCE", reason_code: "INSUFFICIENT_POOLED",
        reason: "pooled observations below minimum 25", min_pooled: 0,
        size_a: 1, size_b: 1, chi2: null, p_value: null, effect: null },
      { evidence_id: "cbd6248beea8f700:1", edge_index: 1, node_a: 1, node_b: 3,
        verdict: "NOT_SEPARATED", reason_code: "NOT_SIGNIFICANT",
        reason: "not significant", min_pooled: 30,
        size_a: 4, size_b: 2, chi2: 1.2, p_value: 0.31, effect: 0.02 },
    ],
  };

  it("states that NOT_SEPARATED is not evidence of a common entity", async () => {
    vi.stubGlobal("fetch", routeFetch([[/separation-evidence/, payload]]));
    render(<SeparationEvidencePanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText(/is not evidence that these two candidate components are the same entity/))
        .toBeInTheDocument());
  });

  it("names CANNOT-LINK as the only constraint and denies MUST-LINK", async () => {
    vi.stubGlobal("fetch", routeFetch([[/separation-evidence/, payload]]));
    render(<SeparationEvidencePanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText(/never emits MUST-LINK/)).toBeInTheDocument());
  });

  it("reports both run fingerprints rather than implying one", async () => {
    vi.stubGlobal("fetch", routeFetch([[/separation-evidence/, payload]]));
    render(<SeparationEvidencePanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText("043ea584e99daf99")).toBeInTheDocument());
    expect(screen.getByText("cbd6248beea8f700")).toBeInTheDocument();
  });

  it("renders an unmeasured p-value as absent, never as zero", async () => {
    vi.stubGlobal("fetch", routeFetch([[/separation-evidence/, payload]]));
    render(<SeparationEvidencePanel alertId="043ea584e99daf99:1" />);
    await waitFor(() => expect(screen.getByText("n/a")).toBeInTheDocument());
    expect(screen.queryByText("0.00e+0")).not.toBeInTheDocument();
  });

  it("claims nothing when the join is unavailable", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/separation-evidence/, { error: "separation_basis_mismatch" }, 500],
    ]));
    render(<SeparationEvidencePanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText(/nothing is claimed/)).toBeInTheDocument());
  });

  it("degrades rather than throwing on a response it does not recognise", async () => {
    // A panel that throws here would blank the whole alert page, taking the
    // analytical assessment down with it.
    vi.stubGlobal("fetch", routeFetch([[/separation-evidence/, { unexpected: true }]]));
    render(<SeparationEvidencePanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText(/nothing is claimed/)).toBeInTheDocument());
  });
});

// ---- the two badge families stay distinct ------------------------------

describe("DispositionBadge", () => {
  it("renders an investigator decision, not a severity", () => {
    const { container } = render(<DispositionBadge state="CONFIRMED" />);
    const badge = container.querySelector(".disp");
    expect(badge).toBeTruthy();
    // A severity badge uses .sev-*; a disposition must never borrow it, or
    // a model ranking and an investigator's decision would read alike.
    expect(container.querySelector('[class*="sev-"]')).toBeNull();
  });

  it("says no decision rather than inventing NEW", () => {
    render(<DispositionBadge state={null} />);
    expect(screen.getByText("no decision")).toBeInTheDocument();
  });
});

// ---- structural patterns: peeling and mixing ---------------------------

import { StructuralPatternsPanel } from "../components/forensics/StructuralPatternsPanel";
import { AnalysisLayerToggle, useAnalysisLayer } from "../components/forensics/AnalysisLayerToggle";

const PATTERNS = {
  alert_id: "043ea584e99daf99:1", cluster_id: 1,
  run_fingerprint: "043ea584e99daf99",
  category: "BLOCKCHAIN_CONTEXT",
  peeling: {
    available: true, members_scored: 148, members_in_chain: 12,
    peel_chain_members_recorded: 12, max_chain_depth: 7,
    fields: ["in_chain", "chain_depth_max"], members: [],
    meaning: "A peeling-chain-like structure is a repeated ordered sequence "
           + "of transactions. It is structural candidate generation, not "
           + "laundering classification.",
  },
  mixing: {
    available: true, scan_id: "fa4b1382cdbc9dee",
    detector: "mixing/1:min_participants=3",
    join_basis: "The scan and the alert run are separate artifacts, joined by txid.",
    transactions_measured: 40, transactions_correlated: 40,
    classes: { NO_MIXING_SIGNAL: 36, MIXING_PATTERN: 4 },
    pattern_count: 4,
    suppressed: { ORDINARY_SHAPE: 32, BATCH_SHAPE: 4 },
    suppressor_meanings: {
      ORDINARY_SHAPE: "Few inputs and few outputs: the ordinary "
                    + "payment-plus-change shape, which is the majority of all "
                    + "transactions.",
      BATCH_SHAPE: "One or two inputs paying many outputs. That is an exchange "
                 + "or merchant batch, not a collaborative spend.",
    },
    transactions: [{
      txid: "abc123def456789012", mixing_class: "MIXING_PATTERN",
      mixing_score: 0.81, output_uniformity: 1.0, input_heterogeneity: 1.0,
      participant_symmetry: 1.0, suppressor: null, n_inputs: 8, n_outputs: 8,
    }],
    meaning: "A mixing-like pattern is a STRUCTURAL observation. It is not "
           + "proof that a mixing service was used, and using one is not "
           + "itself unlawful.",
    insufficient_data_meaning: "Reported as insufficient data rather than as "
                             + "no signal.",
  },
};

describe("StructuralPatternsPanel", () => {
  it("calls the pattern a pattern, never a mixer or a crime", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/patterns/, PATTERNS]]));
    render(<StructuralPatternsPanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText("Mixing-like transaction pattern")).toBeInTheDocument());
    const text = document.body.textContent ?? "";
    for (const forbidden of ["mixer identified", "criminal", "laundering confirmed",
                             "money laundering", "fraud proven"]) {
      expect(text.toLowerCase()).not.toContain(forbidden);
    }
  });

  it("states that a mixing-like pattern is not proof and not unlawful", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/patterns/, PATTERNS]]));
    render(<StructuralPatternsPanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText(/not proof that a mixing service was used/))
        .toBeInTheDocument());
    expect(screen.getByText(/not\s+itself unlawful/)).toBeInTheDocument();
  });

  it("calls peeling structural, not laundering", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/patterns/, PATTERNS]]));
    render(<StructuralPatternsPanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText("Peeling-chain-like structure")).toBeInTheDocument());
    expect(
      screen.getByText(/not\s+laundering classification/),
    ).toBeInTheDocument();
  });

  it("shows the benign shapes it recognised, with their reasons", async () => {
    // "32 were ordinary payments" is far more useful to an investigator
    // than a bare "no mixing found".
    vi.stubGlobal("fetch", routeFetch([[/\/patterns/, PATTERNS]]));
    render(<StructuralPatternsPanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText("Recognised as benign shapes")).toBeInTheDocument());
    expect(screen.getByText("ORDINARY_SHAPE")).toBeInTheDocument();
    expect(screen.getByText(/payment-plus-change shape/)).toBeInTheDocument();
  });

  it("reports peeling depth and how many members sit in a chain", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/patterns/, PATTERNS]]));
    render(<StructuralPatternsPanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText("Members in a chain")).toBeInTheDocument());
    expect(screen.getByText("Deepest chain")).toBeInTheDocument();
  });

  it("says nothing rather than claiming clean when the scan is missing", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/patterns/, {
      ...PATTERNS,
      mixing: {
        available: false, status: "INSUFFICIENT_EVIDENCE",
        detail: "The transaction-structure scan has not been generated.",
        meaning: PATTERNS.mixing.meaning,
      },
    }]]));
    render(<StructuralPatternsPanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText(/has not been generated/)).toBeInTheDocument());
    expect(screen.getAllByText(/Insufficient evidence/).length).toBeGreaterThan(0);
  });

  it("degrades rather than throwing on an unrecognised payload", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/patterns/, { unexpected: true }]]));
    render(<StructuralPatternsPanel alertId="043ea584e99daf99:1" />);
    await waitFor(() =>
      expect(screen.getByText(/not measured/)).toBeInTheDocument());
  });
});

// ---- the analysis layer toggle -----------------------------------------

function LayerProbe() {
  const [layer, setLayer] = useAnalysisLayer();
  return (
    <>
      <span data-testid="layer">{layer}</span>
      <AnalysisLayerToggle layer={layer} onChange={setLayer} />
    </>
  );
}

function renderToggle(initial = "/inv/inv_abc/alerts/a1") {
  return render(
    <MemoryRouter initialEntries={[initial]}>
      <Routes>
        <Route path="/inv/:invId/alerts/:alertId" element={<LayerProbe />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("AnalysisLayerToggle", () => {
  it("offers chain, network and fused", () => {
    renderToggle();
    for (const label of ["Chain", "Network", "Fused"]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("defaults to fused", () => {
    renderToggle();
    expect(screen.getByTestId("layer")).toHaveTextContent("FUSED");
  });

  it("switches layer without navigating away from the alert", async () => {
    renderToggle();
    await userEvent.click(screen.getByRole("button", { name: "Chain" }));
    expect(screen.getByTestId("layer")).toHaveTextContent("CHAIN");
    await userEvent.click(screen.getByRole("button", { name: "Network" }));
    expect(screen.getByTestId("layer")).toHaveTextContent("NETWORK");
    // The toggle is still there, i.e. the route still resolved to this alert.
    expect(screen.getByRole("button", { name: "Fused" })).toBeInTheDocument();
  });

  it("reads the layer from the url so it survives a reload", () => {
    renderToggle("/inv/inv_abc/alerts/a1?layer=network");
    expect(screen.getByTestId("layer")).toHaveTextContent("NETWORK");
  });

  it("falls back to fused on an unrecognised layer", () => {
    renderToggle("/inv/inv_abc/alerts/a1?layer=quantum");
    expect(screen.getByTestId("layer")).toHaveTextContent("FUSED");
  });

  it("performs no request when the layer changes", async () => {
    // The toggle selects among evidence already fetched. A request here
    // would mean the UI was recomputing on a click.
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    renderToggle();
    await userEvent.click(screen.getByRole("button", { name: "Chain" }));
    await userEvent.click(screen.getByRole("button", { name: "Network" }));
    await userEvent.click(screen.getByRole("button", { name: "Fused" }));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("says it is a view over one run, not three models", () => {
    renderToggle();
    expect(screen.getByText(/nothing is recomputed and there are not three models/))
      .toBeInTheDocument();
  });

  it("says so honestly when network analysis is unavailable", () => {
    render(
      <MemoryRouter initialEntries={["/a?layer=network"]}>
        <Routes>
          <Route path="/a" element={
            <AnalysisLayerToggle layer="NETWORK" onChange={() => {}}
              networkAvailable={false} />
          } />
        </Routes>
      </MemoryRouter>,
    );
    expect(
      screen.getByText(/Network analysis unavailable for this alert/),
    ).toBeInTheDocument();
  });

  it("makes no ownership claim in its own description of the network layer", () => {
    renderToggle("/inv/inv_abc/alerts/a1?layer=network");
    expect(screen.getByText(/never attributes ownership/)).toBeInTheDocument();
  });
});

// ---- the synthetic evaluation is never mistaken for production ---------

import { EvaluationPage } from "../pages/EvaluationPage";

const EVALUATION = {
  available: true,
  provenance_type: "SYNTHETIC_CONTROL",
  banner: "SYNTHETIC EVALUATION. Every figure below is a property of a "
        + "generated world built to exercise the detectors under known "
        + "conditions. None of it is a measurement of Bitcoin.",
  world: {
    generator_version: "1.0.0", network_generator_version: "2.0.0",
    world_fingerprint: "20eb52881f0ff5f7aa", seed: 20260917,
    created_at: "2026-09-18T00:00:00Z",
    counts: { entities: 120, addresses: 1944, transactions: 372, observations: 2976 },
    behaviours: ["NORMAL", "MIXING_LIKE", "EXCHANGE_BATCH"],
    adversarial_behaviours: ["EXCHANGE_BATCH"],
    positive_class: ["MIXING_LIKE"],
    positive_class_meaning: "a labelling convention for a controlled "
                          + "experiment, not a claim that the behaviour is unlawful.",
    behaviour_meaning: "These are synthetic transaction SHAPES. They are not "
                     + "claims about how real actors behave.",
    notes: [],
  },
  layer_comparison: {
    meaning: "A comparison of what each EVIDENCE LAYER contributes. It is "
           + "not a comparison of three models.",
    not_measured_here: "False splits and constraint precision are properties "
                     + "of the constrained clustering run, which this "
                     + "experiment does not perform.",
    chain: { addresses: 1416, transactions_with_mixing_pattern: 24 },
    network: { available: true, addresses_with_evidence: 254, addresses_abstaining: 1162 },
    fused: { note: "The network layer never merges and never attributes ownership." },
  },
  overlap: {
    meaning: "Overlap is the fraction of transactions carrying observations.",
    abstention_meaning: "Missing evidence is NOT evidence of separation.",
    levels: [
      { requested_overlap: 1.0, observed_overlap: 1.0,
        transactions_with_observations: 372, coverage: 0.179,
        abstention: 0.821, evidence_available: true },
      { requested_overlap: 0.0, observed_overlap: 0.0,
        transactions_with_observations: 0, coverage: 0.0,
        abstention: 1.0, evidence_available: false },
    ],
    notes: [],
  },
  experiment_available: true,
};

describe("EvaluationPage", () => {
  it("leads with the synthetic banner", async () => {
    vi.stubGlobal("fetch", routeFetch([[/evaluation\/synthetic/, EVALUATION]]));
    render(<MemoryRouter><EvaluationPage /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/not measurements of Bitcoin/)).toBeInTheDocument());
    expect(screen.getByText("SYNTHETIC_CONTROL")).toBeInTheDocument();
  });

  it("says the positive class is a convention, not a claim of illegality", async () => {
    vi.stubGlobal("fetch", routeFetch([[/evaluation\/synthetic/, EVALUATION]]));
    render(<MemoryRouter><EvaluationPage /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/not a claim that the behaviour is unlawful/))
        .toBeInTheDocument());
  });

  it("shows the chain, network and fused comparison", async () => {
    vi.stubGlobal("fetch", routeFetch([[/evaluation\/synthetic/, EVALUATION]]));
    render(<MemoryRouter><EvaluationPage /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText("Chain / Network / Fused")).toBeInTheDocument());
    expect(screen.getByText(/not a comparison of three models/)).toBeInTheDocument();
  });

  it("names what the experiment did not measure", async () => {
    // A gap must never be read as a zero.
    vi.stubGlobal("fetch", routeFetch([[/evaluation\/synthetic/, EVALUATION]]));
    render(<MemoryRouter><EvaluationPage /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/Not measured here/)).toBeInTheDocument());
    expect(screen.getByText(/does not perform/)).toBeInTheDocument();
  });

  it("shows the degradation sweep with an explicit unavailable floor", async () => {
    vi.stubGlobal("fetch", routeFetch([[/evaluation\/synthetic/, EVALUATION]]));
    render(<MemoryRouter><EvaluationPage /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText("Overlap and degradation")).toBeInTheDocument());
    expect(screen.getByText("unavailable")).toBeInTheDocument();
    expect(screen.getByText(/NOT evidence of separation/)).toBeInTheDocument();
  });

  it("claims nothing when the evaluation was never generated", async () => {
    vi.stubGlobal("fetch", routeFetch([[/evaluation\/synthetic/, {
      available: false, provenance_type: "SYNTHETIC_CONTROL",
      banner: EVALUATION.banner,
      detail: "The controlled synthetic evaluation has not been generated. "
            + "Nothing is claimed in its absence.",
    }]]));
    render(<MemoryRouter><EvaluationPage /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/Nothing is claimed in its absence/))
        .toBeInTheDocument());
  });
});
