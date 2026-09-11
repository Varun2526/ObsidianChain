# ObsidianChain — Investigation Console

React/TypeScript dashboard over the **Phase 7 alert API**. The API is the only
data source: nothing is scored, aggregated or derived in the browser, and no
field is displayed that the backend did not return.

## Running the demo

Two processes. The backend serves precomputed artifacts; the dev server
proxies `/api` to it, so no CORS configuration and no backend change is
needed.

```bash
# terminal 1 — the read-only API (from the repository root)
make serve

# terminal 2 — the dashboard
cd frontend && npm install && npm run dev   # http://localhost:5173
```

`npm run build` produces static assets in `dist/` for any static host.

## Checks

```bash
npm run typecheck   # tsc --noEmit, strict
npm test            # vitest, against REAL captured API responses
npm run build       # production bundle
```

`fixtures/` holds responses dumped from the running Phase 7 API. They are not
hand-written: a hand-written fixture encodes what the author believed the
contract was and keeps passing after the contract changes underneath it.

## What the UI will not do

These are enforced by tests, not just by convention.

- **No invented data.** A `null` from the API renders as `n/a`, never as `0`.
  The backend distinguishes "not computable" from "computed as zero" and the
  client preserves that to the last step.
- **No ownership or identity claims from network data.** Network evidence is
  labelled `NETWORK_CONTEXT`, carries the synthetic-data warning, and repeats
  the backend's own caveat verbatim. A test asserts that any ownership
  language in a rendered response sits inside a negation.
- **No transaction nodes in the graph.** The alert API exposes
  address-to-address relationships only, so drawing transaction nodes would
  mean inventing objects the backend never described. The legend says so.
- **No silent staleness.** A 409 stale run fingerprint gets its own panel
  explaining that a cluster id does not necessarily refer to the same cluster
  across runs — the one error this system exists to make visible.

## Structure

```
src/api/types.ts     mirrors the API response shapes exactly
src/api/client.ts    fetch wrapper preserving the backend error taxonomy
src/components/      AlertQueue, AlertDetail, WhyFlagged, EvidencePanel,
                     InvestigationGraph (Cytoscape), Timeline,
                     NetworkContextPanel, ProvenancePanel, ErrorState
```
