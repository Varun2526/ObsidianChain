/**
 * The investigation read path in the UI: address profile, model
 * intelligence, search and the graph helpers.
 *
 * What these tests pin is the separation the redesign exists for: observed
 * chain facts, external attribution and model output render in separate,
 * labelled places; research numbers carry their result type; search shows
 * only what the API returned.
 */
import { StrictMode } from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import type { AddressProfile, GraphEdge, GraphNode, ModelDetail, ModelList, TraceResponse } from "../api/intel";
import { traceQuery } from "../api/intel";
import { FlowGraph } from "../components/graph/FlowGraph";
import { mergeGraph, matchNodes, timestepRange } from "../components/graph/graphModel";
import { CommandPalette } from "../components/modals/CommandPalette";
import { EntityPage } from "../pages/intel/EntityPage";
import { ModelsPage } from "../pages/intel/ModelsPage";
import { AuthProvider, useAuth } from "../store/auth";

function routeFetch(routes: Array<[RegExp, unknown, number?]>) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    for (const [pattern, body, status = 200] of routes) {
      if (pattern.test(url)) {
        return { ok: status >= 200 && status < 300, status, json: async () => body } as Response;
      }
    }
    throw new Error(`unstubbed request: ${url}`);
  });
}

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

const ADDR = "1EYitrwBYNWuTBcjZFbEUdqHppe2raLpaF";
const PROV = {
  chain_index_fingerprint: "e2f29c90571902cf", chain_scope: "Observed Elliptic++ structure.",
  model_scope: "Phase 7 reference run.", watchlist_scope: "External attribution.", labels_served: false as const,
};

const node = (id: string, kind: GraphNode["kind"], data: GraphNode["data"] = {}): GraphNode => ({ id, kind, label: id, data });
const edge = (s: string, t: string, kind = "PAYS"): GraphEdge => ({ id: `${kind}:${s}->${t}`, source: s, target: t, kind, data: {} });

const TRACE: TraceResponse = {
  seeds: { addresses: [ADDR], txids: [] }, direction: "both", hops: 1, max_nodes: 160,
  graph: { node_count: 2, edge_count: 1, nodes: [node(`addr:${ADDR}`, "address", { address: ADDR, seed: true }), node("tx:1", "transaction", { txid: 1, timestep: 4 })],
           edges: [edge(`addr:${ADDR}`, "tx:1", "SPENDS")] },
  truncated: false, hub_transactions_skipped: [], hub_threshold: 200, meaning: "Edges follow value.", provenance: PROV,
};

function profile(over: Partial<AddressProfile> = {}): AddressProfile {
  return {
    address: ADDR,
    observed: { transactions: 24, as_input: 10, as_output: 14, first_timestep: 2, last_timestep: 42, active_timesteps: 11,
                timeline: [{ timestep: 2, as_input: 1, as_output: 0 }],
                transactions_list: [{ txid: 1, timestep: 4, fee_btc: 0.001, in_btc: 1, out_btc: 0.999, n_inputs: 1, n_outputs: 2, role: "input" }],
                transactions_shown: 1 },
    counterparties: { paid_to: [], paid_to_total: 0, paid_by: [], paid_by_total: 0, hub_transactions_skipped: 0, hub_threshold: 200 },
    cluster: { cluster_id: 525348, size: 2971 },
    watchlist: [{ source: "OFAC_SDN", label: "SECONDEYE SOLUTION" }],
    model: { alerts: [], explanations: [], scope: "PHASE7_REFERENCE_RUN", meaning: "A learned association." },
    network: { observations: [], meaning: "Synthetic; an announcing peer is not a sender." },
    provenance: PROV,
    ...over,
  };
}

function renderAt(path: string, route: string, el: React.ReactNode) {
  render(<MemoryRouter initialEntries={[path]}><Routes><Route path={route} element={el} /></Routes></MemoryRouter>);
}

// ---- pure helpers ---------------------------------------------------------

describe("graph helpers", () => {
  it("builds the trace query the backend expects", () => {
    const q = new URLSearchParams(traceQuery({ addresses: ["a", "b"], txids: [7], direction: "upstream", hops: 3, maxNodes: 50, minTimestep: 5 }));
    expect(q.getAll("address")).toEqual(["a", "b"]);
    expect(q.getAll("txid")).toEqual(["7"]);
    expect(q.get("direction")).toBe("upstream");
    expect(q.get("hops")).toBe("3");
    expect(q.get("max_nodes")).toBe("50");
    expect(q.get("min_timestep")).toBe("5");
    expect(q.has("max_timestep")).toBe(false);
  });

  it("merges an expansion without dangling edges and without losing a seed", () => {
    const base = { nodes: [node("addr:a", "address", { seed: true }), node("tx:1", "transaction")], edges: [edge("addr:a", "tx:1", "SPENDS")] };
    const add = { nodes: [node("addr:a", "address", { seed: false, hop: 1 }), node("addr:b", "address")],
                  edges: [edge("tx:1", "addr:b"), edge("tx:9", "addr:b")] };
    const merged = mergeGraph(base, add);
    const ids = new Set(merged.nodes.map((n) => n.id));
    expect(merged.nodes).toHaveLength(3);
    expect(merged.nodes.find((n) => n.id === "addr:a")!.data.seed).toBe(true);
    expect(merged.edges.every((e) => ids.has(e.source) && ids.has(e.target))).toBe(true);
    expect(merged.edges.map((e) => e.id)).not.toContain("PAYS:tx:9->addr:b");
  });

  it("finds nodes and the timestep range from the loaded graph only", () => {
    const nodes = [node("tx:1", "transaction", { timestep: 9 }), node("tx:2", "transaction", { timestep: 3 }), node("addr:xyz", "address")];
    expect(timestepRange(nodes)).toEqual([3, 9]);
    expect(matchNodes(nodes, "xy").map((n) => n.id)).toEqual(["addr:xyz"]);
    expect(matchNodes(nodes, "")).toEqual([]);
  });

  it("gives the canvas a text alternative where it cannot draw", () => {
    render(<FlowGraph nodes={TRACE.graph.nodes} edges={TRACE.graph.edges} layout="flow" />);
    expect(screen.getByRole("img", { name: /2 nodes, 1 edges/ })).toBeInTheDocument();
  });
});

// ---- the address page ---------------------------------------------------

describe("EntityPage", () => {
  it("keeps observed activity, attribution and model output in separate, labelled sections", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/api\/addresses\//, profile()], [/\/api\/graph\/trace/, TRACE]]));
    renderAt(`/entity/${ADDR}`, "/entity/:address", <EntityPage />);
    await waitFor(() => expect(screen.getByText("External attribution")).toBeInTheDocument());
    expect(screen.getByText(/Asserted by the named source, not by this system/)).toBeInTheDocument();
    expect(screen.getByText("Observed activity")).toBeInTheDocument();
    expect(screen.getByText("Model association")).toBeInTheDocument();
    // no alert membership is stated as exactly that - not as a clearance
    expect(screen.getByText(/That is not a clearance/)).toBeInTheDocument();
    expect(screen.getByText(/Class labels are never served/)).toBeInTheDocument();
  });

  it("labels a model score as an association, not proof", async () => {
    const withAlert = profile({ model: {
      alerts: [{ alert_id: "043ea584e99daf99:1", risk_score: 0.97, severity: "CRITICAL", observed_at_timestep: 40 }],
      explanations: [{ alert_id: "043ea584e99daf99:1", address: ADDR, feature: "fees_total_asof_t", contribution: 0.8,
                       feature_value: 0.03, feature_group: "M0", signal_category: "MODEL_SIGNAL" }],
      scope: "PHASE7_REFERENCE_RUN", meaning: "A learned association." } });
    vi.stubGlobal("fetch", routeFetch([[/\/api\/addresses\//, withAlert], [/\/api\/graph\/trace/, TRACE]]));
    renderAt(`/entity/${ADDR}`, "/entity/:address", <EntityPage />);
    await waitFor(() => expect(screen.getByText("A learned association, not proof")).toBeInTheDocument());
    expect(screen.getByText("fees_total_asof_t")).toBeInTheDocument();
    expect(screen.getByText(/A contribution describes the model, not the address/)).toBeInTheDocument();
  });

  it("shows the backend's not-found answer", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/api\/addresses\//, { error: "address_not_found", detail: "address 'x' does not appear" }, 404],
      [/\/api\/graph\/trace/, { error: "address_not_found", detail: "none" }, 404],
    ]));
    renderAt("/entity/x", "/entity/:address", <EntityPage />);
    await waitFor(() => expect(screen.getByText("Address not in the chain index")).toBeInTheDocument());
  });
});

// ---- model intelligence -------------------------------------------------

const LIST: ModelList = {
  roles: { champion: "v5", candidate: null, fallback: "v5_fb" },
  models: [{ version: "v5", role: "champion", feature_schema_version: "s/5", registered_at: null, attested_source_commit: "5431241a4d",
             holdout: { nap: 0.548, "P@100": 1 }, notes: "n", withdrawn: false },
           { version: "v4", role: null, feature_schema_version: "s/5", registered_at: null, attested_source_commit: null,
             holdout: null, notes: "WITHDRAWN: leakage", withdrawn: true }],
  history: [], meaning: "The registry is the only authority.",
};

const DETAIL = {
  version: "v5", roles: ["champion"], feature_schema_version: "s/5", attested_source_commit: "5431241a4d", registered_at: null, notes: null,
  manifest: { model_type: "LightGBM", features: ["a", "b"] },
  result_types: { DEVELOPMENT: "folds 1-6", CONFIRMATION: "folds 7-12", HOLDOUT: "t42-49", PRODUCTION: "none" },
  production: { available: false, meaning: "No production performance exists until delayed labels arrive." },
  holdout: { result_type: "HOLDOUT", period: "t42-49", exception: null, n_addresses: 100, positives: 5, prevalence: 0.05,
             address: { nap: 0.548, "P@100": 1 }, calibration: { ece: 0.0096 }, reliability: null, slices_nap: {},
             by_first_seen_step: { t42: { n: 10, positives: 2, nap: 0.7 }, t43: { n: 10, positives: 1, nap: 0.13 } }, sha256: "abc" },
} as unknown as ModelDetail;

describe("ModelsPage", () => {
  it("labels every number with its result type and says production performance is unknown", async () => {
    vi.stubGlobal("fetch", routeFetch([[/\/api\/models\/v5/, DETAIL], [/\/api\/models$/, LIST]]));
    renderAt("/models", "/models", <ModelsPage />);
    await waitFor(() => expect(screen.getByText("Holdout by timestep")).toBeInTheDocument());
    for (const t of ["DEVELOPMENT", "CONFIRMATION", "HOLDOUT", "PRODUCTION"]) {
      expect(screen.getAllByText(t).length).toBeGreaterThan(0);
    }
    expect(screen.getByText("unknown")).toBeInTheDocument();
    expect(screen.getByText("Known failure")).toBeInTheDocument();
    expect(screen.getByText("Withdrawn")).toBeInTheDocument();
  });
});

// ---- search -------------------------------------------------------------

describe("CommandPalette", () => {
  it("shows only what the search API returned", async () => {
    const fetchMock = routeFetch([
      [/\/api\/investigations/, { investigations: [], scope: "owned" }],
      [/\/api\/search\?q=1EYi/, { query: "1EYi", results: [{ kind: "address", id: ADDR, label: ADDR }] }],
    ]);
    vi.stubGlobal("fetch", fetchMock);
    render(<MemoryRouter><CommandPalette open onClose={() => {}} /></MemoryRouter>);
    await userEvent.type(screen.getByRole("combobox", { name: "Search" }), "1EYi");
    const list = await screen.findByRole("listbox");
    await waitFor(() => expect(within(list).getAllByRole("option")).toHaveLength(1));
    expect(within(list).getByText(ADDR)).toBeInTheDocument();
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("/api/search?q=1EYi"))).toBe(true);
  });

  it("says so when nothing matches instead of suggesting anything", async () => {
    vi.stubGlobal("fetch", routeFetch([
      [/\/api\/investigations/, { investigations: [], scope: "owned" }],
      [/\/api\/search/, { query: "zzzz", results: [] }],
    ]));
    render(<MemoryRouter><CommandPalette open onClose={() => {}} /></MemoryRouter>);
    await userEvent.type(screen.getByRole("combobox", { name: "Search" }), "zzzz");
    await waitFor(() => expect(screen.getByText(/No address, transaction, alert or investigation matches/)).toBeInTheDocument());
  });
});

// ---- deep links survive a hard refresh ----------------------------------

describe("session bootstrap", () => {
  it("never reports 'logged out' for an aborted probe (StrictMode double effect)", async () => {
    const seen: string[] = [];
    function Probe() {
      const { identity, loading } = useAuth();
      seen.push(loading ? "loading" : identity ? "alice" : "anonymous");
      return null;
    }
    vi.stubGlobal("fetch", vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      await new Promise((r) => setTimeout(r, 20));
      if (init?.signal?.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
      return { ok: true, status: 200, json: async () => ({
        user: { id: "u", username: "alice", display_name: "A", role: "INVESTIGATOR", active: true, created_at: "" },
        capabilities: [] }) } as Response;
    }));
    render(<StrictMode><AuthProvider><Probe /></AuthProvider></StrictMode>);
    await waitFor(() => expect(seen.at(-1)).toBe("alice"));
    expect(seen).not.toContain("anonymous");
  });
});
