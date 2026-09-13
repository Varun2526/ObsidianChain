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
  it("draws only node kinds the API actually backs", () => {
    // Phase 9 added transaction and ip nodes, because the alert contract now
    // returns txids and announcing peers. Before that it returned neither
    // and this test asserted their ABSENCE - the contract changed for a
    // reason, so the assertion follows it rather than being relaxed.
    const model = buildGraphModel(detail, 10);
    const nodes = model.elements.filter((e) => !("source" in (e.data as object)));
    const kinds = new Set(nodes.map((n) => (n.data as any).kind));
    for (const kind of kinds) {
      expect(["cluster", "address", "transaction", "ip"]).toContain(kind);
    }
    expect(kinds.has("cluster")).toBe(true);
    expect(kinds.has("address")).toBe(true);
  });

  it("invents no node kind the API never described", () => {
    const model = buildGraphModel(detail, 250);
    const kinds = new Set(
      model.elements
        .filter((e) => !("source" in (e.data as object)))
        .map((e) => (e.data as any).kind),
    );
    // No "person", "entity", "owner" or any other node standing for an
    // identity the backend does not assert.
    for (const invented of ["person", "owner", "entity", "wallet_owner"]) {
      expect(kinds.has(invented)).toBe(false);
    }
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
      expect(["MEMBER_OF", "CO_SPEND_COMPONENT", "FUNDED_VIA_TRANSACTION",
              "INVOLVES", "ANNOUNCED_BY"]).toContain(kind);
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

// ---- Phase 9: onboarding, ingestion, correlation -----------------------

import ingestFixture from "../../fixtures/ingest_csv.json";
import { Home } from "../components/Home";
import { IngestPage } from "../components/IngestPage";
import { CorrelationPanel } from "../components/CorrelationPanel";
import type { IngestResult } from "../api/types";

const ingested = ingestFixture as unknown as IngestResult;

describe("Home (onboarding)", () => {
  it("shows the three steps in order so a new user knows what to do", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: list })));
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(screen.getByText("Load data")).toBeInTheDocument();
    expect(screen.getByText("Analyse")).toBeInTheDocument();
    expect(screen.getByText("Investigate")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText(/A scored dataset is loaded/)).toBeInTheDocument());
  });

  it("never implies that opening the dashboard scored anything", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: list })));
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(
      screen.getByText(/Scoring is an offline pipeline run, not a button/),
    ).toBeInTheDocument();
  });

  it("says the network data is synthetic on the landing page", () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: list })));
    render(<MemoryRouter><Home /></MemoryRouter>);
    expect(screen.getByText(/synthetic/i)).toBeInTheDocument();
  });

  it("tells the user how to start the API when it is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("down"); }));
    render(<MemoryRouter><Home /></MemoryRouter>);
    await waitFor(() =>
      expect(screen.getByText(/make serve/)).toBeInTheDocument());
  });
});

describe("IngestPage", () => {
  it("reports what parsed, using a real endpoint response", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: ingested })));
    render(<MemoryRouter><IngestPage /></MemoryRouter>);
    const file = new File(["txid\n1\n"], "sample.csv", { type: "text/csv" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, file);
    await waitFor(() => expect(screen.getByText("Validation")).toBeInTheDocument());
    expect(screen.getByText("Rows valid")).toBeInTheDocument();
  });

  it("states prominently that the upload was NOT scored", async () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: ingested })));
    render(<MemoryRouter><IngestPage /></MemoryRouter>);
    const file = new File(["txid\n1\n"], "sample.csv", { type: "text/csv" });
    await userEvent.upload(
      document.querySelector('input[type="file"]') as HTMLInputElement, file,
    );
    await waitFor(() =>
      expect(screen.getByText(/validated, not scored/)).toBeInTheDocument());
    expect(screen.getByText(/phase6-dataset/)).toBeInTheDocument();
  });

  it("offers all three formats the PS names", () => {
    vi.stubGlobal("fetch", mockFetch(() => ({ body: ingested })));
    render(<MemoryRouter><IngestPage /></MemoryRouter>);
    expect(screen.getByText(/CSV · JSON · XML/)).toBeInTheDocument();
  });
});

describe("CorrelationPanel", () => {
  it("shows the IP -> transaction -> wallet chain from the API", () => {
    render(<CorrelationPanel correlation={detail.correlation} />);
    expect(screen.getByText(/IP → transaction → wallet/)).toBeInTheDocument();
    const tx = detail.correlation.transactions[0]!;
    expect(screen.getByText(tx.txid)).toBeInTheDocument();
  });

  it("leads with the caveat that a peer is not a sender", () => {
    render(<CorrelationPanel correlation={detail.correlation} />);
    expect(screen.getByText(/An announcing peer is not a sender/)).toBeInTheDocument();
  });

  it("reports no country rather than an empty one for reserved ranges", () => {
    render(<CorrelationPanel correlation={detail.correlation} />);
    expect(screen.getByText(/not globally routable/)).toBeInTheDocument();
    expect(
      screen.getByText(/reported as unavailable rather than guessed/),
    ).toBeInTheDocument();
  });

  it("says insufficient evidence when nothing correlated", () => {
    render(
      <CorrelationPanel
        correlation={{ ...detail.correlation, available: false,
                       status: "INSUFFICIENT_EVIDENCE", transactions: [],
                       summary: null }}
      />,
    );
    expect(screen.getByText(/Insufficient evidence/)).toBeInTheDocument();
  });
});

describe("graph with network correlation", () => {
  it("draws transaction and peer nodes from the correlation", () => {
    const model = buildGraphModel(detail, 60);
    const kinds = new Set(
      model.elements
        .filter((e) => !("source" in (e.data as object)))
        .map((e) => (e.data as any).kind),
    );
    expect(kinds.has("transaction")).toBe(true);
    expect(kinds.has("ip")).toBe(true);
    expect(model.renderedTransactions).toBeGreaterThan(0);
    expect(model.renderedIps).toBeGreaterThan(0);
  });

  it("uses ANNOUNCED_BY, never a 'sent' edge", () => {
    const model = buildGraphModel(detail, 60);
    const edgeKinds = new Set(
      model.elements
        .filter((e) => "source" in (e.data as object))
        .map((e) => (e.data as any).kind),
    );
    for (const kind of edgeKinds) {
      expect(String(kind).toLowerCase()).not.toContain("sent");
      expect(["MEMBER_OF", "CO_SPEND_COMPONENT", "FUNDED_VIA_TRANSACTION",
              "INVOLVES", "ANNOUNCED_BY"]).toContain(kind);
    }
  });

  it("draws no transaction node when nothing correlated", () => {
    const model = buildGraphModel(
      { ...detail, correlation: { ...detail.correlation, available: false,
                                  transactions: [] } },
      60,
    );
    expect(model.renderedTransactions).toBe(0);
    expect(model.renderedIps).toBe(0);
  });

  it("only draws a transaction whose wallets are on the graph", () => {
    // With a tiny cap the member set is small, so most correlated
    // transactions have no rendered endpoint and must be skipped rather
    // than drawn floating.
    const small = buildGraphModel(detail, 3);
    const txNodes = small.elements.filter(
      (e) => (e.data as any).kind === "transaction",
    );
    const addressIds = new Set(
      small.elements
        .filter((e) => (e.data as any).kind === "address")
        .map((e) => (e.data as any).id),
    );
    for (const node of txNodes) {
      const involved = small.elements.filter(
        (e) => (e.data as any).kind === "INVOLVES" &&
               (e.data as any).source === (node.data as any).id,
      );
      expect(involved.length).toBeGreaterThan(0);
      for (const edge of involved) {
        expect(addressIds.has((edge.data as any).target)).toBe(true);
      }
    }
  });
});
