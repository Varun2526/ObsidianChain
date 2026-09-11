/**
 * Frontend tests against REAL captured API responses.
 *
 * The fixtures in `frontend/fixtures/` were dumped from the running Phase 7
 * API, not hand-written. That matters: a hand-written fixture encodes what I
 * believed the contract was, and would keep passing after the contract
 * changed underneath it.
 *
 * The properties under test are the ones that would be invisible on a
 * rendered page if they broke: that a null renders as "not available" rather
 * than as a zero, that network context is never phrased as an ownership
 * claim, that a stale run fingerprint is surfaced distinctly, and that the
 * graph does not silently draw objects the API never described.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import listFixture from "../../fixtures/alerts_list.json";
import detailFixture from "../../fixtures/alert_detail_graph.json";
import { ApiError, buildAlertQuery } from "../api/client";
import type { AlertDetail, AlertListResponse } from "../api/types";
import { AlertQueue } from "../components/AlertQueue";
import { AlertDetailPage } from "../components/AlertDetail";
import { ErrorState } from "../components/ErrorState";
import { NetworkContextPanel } from "../components/NetworkContextPanel";
import { buildGraphModel } from "../components/InvestigationGraph";
import { Value } from "../components/primitives";

const list = listFixture as unknown as AlertListResponse;
const detail = detailFixture as unknown as AlertDetail;

function mockFetch(handler: (url: string) => { status?: number; body: unknown }) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const { status = 200, body } = handler(String(input));
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    } as Response;
  });
}

beforeEach(() => {
  // jsdom has no layout engine, so Cytoscape cannot measure a container.
  // Stubbed rather than skipped: the graph's real logic is buildGraphModel,
  // which is tested directly and needs no DOM at all.
  vi.mock("cytoscape", () => ({
    default: () => ({
      on: () => {},
      // `ready` runs its callback immediately here; the real Core defers it
      // until layout settles, which jsdom cannot do without a layout engine.
      ready: (fn: () => void) => fn(),
      fit: () => {},
      destroy: () => {},
    }),
  }));
});

afterEach(() => { vi.restoreAllMocks(); });

// ---- the fixtures are real -------------------------------------------

describe("fixtures", () => {
  it("came from the real API and carry no ground-truth label", () => {
    const rendered = JSON.stringify(detail);
    expect(rendered).not.toContain('"y":');
    expect(rendered).not.toContain('"is_illicit"');
    expect(detail.provenance.provenance_type).toBe("PRODUCTION");
    expect(detail.provenance.run_fingerprint).toHaveLength(64);
  });
});

// ---- the query the client actually sends ------------------------------

describe("buildAlertQuery", () => {
  it("repeats severity so the backend receives a list", () => {
    const query = buildAlertQuery({ severity: ["CRITICAL", "HIGH"] });
    expect(query).toContain("severity=CRITICAL");
    expect(query).toContain("severity=HIGH");
  });

  it("omits filters that were never set rather than sending defaults", () => {
    const query = buildAlertQuery({});
    expect(query).not.toContain("min_risk");
    expect(query).not.toContain("first_timestep");
    expect(query).toContain("limit=25");
  });

  it("sends a zero minimum rather than dropping it", () => {
    // 0 is a real filter value and must survive the falsy check.
    expect(buildAlertQuery({ minRisk: 0 })).toContain("min_risk=0");
  });
});

// ---- nulls never become zeros -----------------------------------------

describe("Value", () => {
  it("renders a null as an explicit absence, not a zero", () => {
    render(<Value value={null} />);
    expect(screen.getByText("n/a")).toBeInTheDocument();
    expect(screen.queryByText("0")).not.toBeInTheDocument();
  });

  it("still renders a real zero", () => {
    render(<Value value={0} />);
    expect(screen.getByText("0")).toBeInTheDocument();
  });
});

// ---- the alert queue ---------------------------------------------------

describe("AlertQueue", () => {
  it("renders the ranked queue from the API", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: list })));
    render(<MemoryRouter><AlertQueue /></MemoryRouter>);

    await waitFor(() => expect(screen.getByText(/matched of/)).toBeInTheDocument());
    const first = list.alerts[0]!;
    expect(screen.getByText(String(first.cluster_id))).toBeInTheDocument();
    expect(screen.getAllByText(first.severity).length).toBeGreaterThan(0);
  });

  it("says that column sorting is page-local, not global", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: list })));
    render(<MemoryRouter><AlertQueue /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/reorders the current page only/)).toBeInTheDocument());
  });

  it("pushes severity filtering to the API", async () => {
    const fetchMock = mockFetch(() => ({ body: list }));
    vi.stubGlobal("fetch", fetchMock);
    render(<MemoryRouter><AlertQueue /></MemoryRouter>);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    await userEvent.click(screen.getByRole("button", { name: "CRITICAL" }));
    await waitFor(() => {
      const urls = fetchMock.mock.calls.map((c) => String(c[0]));
      expect(urls.some((u) => u.includes("severity=CRITICAL"))).toBe(true);
    });
  });

  it("shows an empty state rather than a blank table", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({
      body: { ...list, alerts: [], alert_count_matched: 0 },
    })));
    render(<MemoryRouter><AlertQueue /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/No alerts match these filters/)).toBeInTheDocument());
  });

  it("surfaces an unreachable API instead of an empty page", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("boom"); }));
    render(<MemoryRouter><AlertQueue /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/Cannot reach the API/)).toBeInTheDocument());
  });
});

// ---- stale fingerprints ------------------------------------------------

describe("stale run fingerprint", () => {
  it("is surfaced distinctly, not as a generic failure", () => {
    const error = new ApiError("alert_id_stale", 409, "minted against run abc");
    render(<ErrorState error={error} />);
    expect(screen.getByText(/Run fingerprint is stale/)).toBeInTheDocument();
    expect(screen.getByText(/does not necessarily refer to the same cluster/))
      .toBeInTheDocument();
  });

  it("reaches the investigator through the detail page", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({
      status: 409,
      body: { error: "alert_id_stale", detail: "minted against run abc" },
    })));
    render(
      <MemoryRouter initialEntries={["/alerts/deadbeefdeadbeef:1"]}>
        <Routes>
          <Route path="/alerts/:alertId" element={<AlertDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );
    await waitFor(() =>
      expect(screen.getByText(/Run fingerprint is stale/)).toBeInTheDocument());
  });
});

// ---- the detail page ---------------------------------------------------

describe("AlertDetailPage", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: detail })));
  });

  async function renderDetail() {
    render(
      <MemoryRouter initialEntries={[`/alerts/${detail.alert_id}`]}>
        <Routes>
          <Route path="/alerts/:alertId" element={<AlertDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );
    await waitFor(() =>
      expect(screen.getByText(/Why was this flagged/)).toBeInTheDocument());
  }

  it("shows risk, severity and every aggregation the API returned", async () => {
    await renderDetail();
    expect(screen.getByText(/Calibrated risk/)).toBeInTheDocument();
    for (const name of Object.keys(detail.risk.aggregations)) {
      expect(screen.getAllByText(new RegExp(name)).length).toBeGreaterThan(0);
    }
  });

  it("labels contributions as log-odds and tags each value's category", async () => {
    await renderDetail();
    // Appears in the panel header AND in the frozen meaning text; both
    // are intentional, so multiplicity is the expected state.
    expect(screen.getAllByText(/log-odds/).length).toBeGreaterThan(0);
    const top = detail.why_flagged.per_member[0]!.contributions[0]!;
    expect(screen.getAllByText(top.feature).length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(top.value_category.replace(/_/g, " ")).length,
    ).toBeGreaterThan(0);
  });

  it("renders every M0-M3 evidence group", async () => {
    await renderDetail();
    for (const group of ["M0", "M1", "M2", "M3"]) {
      expect(screen.getByRole("button", { name: group })).toBeInTheDocument();
    }
  });

  it("makes the as-of timestep explicit on the timeline", async () => {
    await renderDetail();
    expect(
      screen.getByText(new RegExp(`as of timestep ${detail.summary.last_timestep}`)),
    ).toBeInTheDocument();
  });

  it("shows the provenance chain an investigator needs", async () => {
    await renderDetail();
    const panel = screen.getByText("Provenance").closest("section")!;
    // The run fingerprint and the Phase 6 dataset fingerprint are the same
    // digest in this run, so it legitimately renders twice.
    expect(
      within(panel).getAllByText(detail.provenance.run_fingerprint!).length,
    ).toBeGreaterThan(0);
    expect(within(panel).getByText(detail.provenance.model!.family)).toBeInTheDocument();
    expect(within(panel).getAllByText(/as-of/).length).toBeGreaterThan(0);
  });

  it("states how many members are not listed", async () => {
    await renderDetail();
    if (detail.members.withheld > 0) {
      expect(
        screen.getByText(new RegExp(`${detail.members.withheld.toLocaleString()} further`)),
      ).toBeInTheDocument();
    }
  });
});

// ---- network context never claims ownership ---------------------------

describe("NetworkContextPanel", () => {
  it("carries the synthetic warning and the backend's own caveat", () => {
    render(<NetworkContextPanel context={detail.network_context} />);
    expect(screen.getByText(/Synthetic network data/)).toBeInTheDocument();
    expect(screen.getByText(detail.network_context.meaning)).toBeInTheDocument();
  });

  it("makes no ownership, sender or common-ownership claim", () => {
    const { container } = render(
      <NetworkContextPanel context={detail.network_context} />,
    );
    const text = container.textContent!.toLowerCase();
    // Any ownership language present must be inside a negation - the backend
    // ships a disclaimer that legitimately contains these words.
    for (const phrase of ["owns", "sender", "same person"]) {
      let from = 0;
      while ((from = text.indexOf(phrase, from)) !== -1) {
        const before = text.slice(Math.max(0, from - 90), from);
        expect(
          /\bnot\b|\bnever\b|cannot|\bno\b/.test(before),
          `"${phrase}" appears without a negation: ...${before.slice(-70)}`,
        ).toBe(true);
        from += phrase.length;
      }
    }
  });

  it("says insufficient evidence rather than showing zeros", () => {
    render(
      <NetworkContextPanel
        context={{
          ...detail.network_context,
          available: false,
          status: "INSUFFICIENT_EVIDENCE",
          members_with_observations: 0,
          insufficient_evidence_meaning: "Not computable for these addresses.",
        }}
      />,
    );
    expect(screen.getByText(/Insufficient evidence/)).toBeInTheDocument();
    expect(screen.queryByText("Members with observations")).not.toBeInTheDocument();
  });
});

// ---- the graph ---------------------------------------------------------

describe("buildGraphModel", () => {
  it("draws a cluster node and the capped address nodes", () => {
    const model = buildGraphModel(detail, 10);
    const nodes = model.elements.filter((e) => !("source" in (e.data as object)));
    const kinds = new Set(nodes.map((n) => (n.data as any).kind));
    expect(kinds).toEqual(new Set(["cluster", "address"]));
    expect(model.renderedAddresses).toBe(
      Math.min(10, detail.members.rows.length),
    );
  });

  it("never invents a transaction node, because the API exposes none", () => {
    const model = buildGraphModel(detail, 250);
    const kinds = model.elements.map((e) => (e.data as any).kind);
    expect(kinds).not.toContain("transaction");
  });

  it("actually renders relationships, not an empty graph", () => {
    // The regression this guards: the API caps members.rows at 25 while
    // relationship edges span the whole cluster, so building nodes from
    // members alone dropped every edge and the panel reported
    // "0 relationships" on an alert that has 147.
    expect(detail.relationships.count).toBeGreaterThan(0);
    const model = buildGraphModel(detail, 60);
    expect(model.renderedEdges).toBeGreaterThan(0);
  });

  it("marks an endpoint outside the scored member list as unscored", () => {
    const model = buildGraphModel(detail, 60);
    const nodes = model.elements.filter((e) => !("source" in (e.data as object)));
    const scoredAddresses = new Set(detail.members.rows.map((m) => m.address));
    for (const node of nodes) {
      const data = node.data as any;
      if (data.kind !== "address" || scoredAddresses.has(data.address)) continue;
      // Never a fabricated risk: unknown is rendered as unknown.
      expect(data.risk).toBeNull();
      expect(data.severity).toBe("UNSCORED");
    }
  });

  it("never exceeds the node cap", () => {
    for (const cap of [10, 30, 60]) {
      expect(buildGraphModel(detail, cap).renderedAddresses).toBeLessThanOrEqual(cap);
    }
  });

  it("keeps the graph readable by capping, and reports what it withheld", () => {
    const small = buildGraphModel(detail, 5);
    expect(small.renderedAddresses).toBeLessThanOrEqual(5);
    expect(small.withheldAddresses).toBeGreaterThan(0);
  });

  it("drops an edge whose endpoint was capped out, and counts it", () => {
    const full = buildGraphModel(detail, 250);
    const small = buildGraphModel(detail, 3);
    expect(small.renderedEdges).toBeLessThanOrEqual(full.renderedEdges);
    expect(small.withheldEdges).toBeGreaterThan(0);
  });

  it("emits only relationship kinds the API supports", () => {
    const model = buildGraphModel(detail, 250);
    const edgeKinds = new Set(
      model.elements
        .filter((e) => "source" in (e.data as object))
        .map((e) => (e.data as any).kind),
    );
    for (const kind of edgeKinds) {
      expect(["MEMBER_OF", "CO_SPEND_COMPONENT", "FUNDED_VIA_TRANSACTION"])
        .toContain(kind);
    }
  });

  it("deduplicates a relationship reported in both directions", () => {
    const doubled: AlertDetail = {
      ...detail,
      relationships: {
        ...detail.relationships,
        edges: [
          ...detail.relationships.edges,
          ...detail.relationships.edges.map((e) => ({
            ...e, address_a: e.address_b, address_b: e.address_a,
          })),
        ],
      },
    };
    const once = buildGraphModel(detail, 250);
    const twice = buildGraphModel(doubled, 250);
    expect(twice.renderedEdges).toBe(once.renderedEdges);
  });
});
