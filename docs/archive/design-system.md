# ObsidianChain design system

Authority: the Garden skill `web-design-engineer` (ConardLi/garden-skills),
applied as a **Redesign · Overhaul** of the visual language that preserves
routes, authentication, casework behaviour and the behavioural contracts the
frontend tests pin. Tokens: `frontend/src/design/tokens.css`.

## Design read

| | |
|---|---|
| Artifact | operational investigation workstation |
| Audience | blockchain investigators, reviewers, administrators, ML reviewers |
| Visual language | institutional information architecture (terminal density), with hairline restraint from builder tools |
| Mode | redesign · overhaul (visuals), preserve (routes, contracts) |
| Visual variance | 3: a stable shell and grid, so investigators can build muscle memory |
| Motion | 2: state feedback only; no entrance choreography |
| Information density | 8: analytical; strong grouping, a persistent inspector, tables before cards |
| Asset dependence | 2: the data is the interface; no imagery |
| Brand fidelity | 4: the name survives; the cyan neon and glowing dot do not |

### Why the old look was retired

Pure black with `#00f0ff` neon and glow is the "cyber-neon" failure pattern.
It reads as a demo, not as a tool someone works in for eight hours. Neon is
also the most-used colour in the UI, so no colour signalled importance.

## Tokens

- **Ground and surfaces:** warm graphite `#0e0f11` / `#15171a` / `#1b1d21` / `#22252a`.
  Panels are separated by 1px hairlines (`#2a2d33`), not by shadows.
- **Text:** `#e9e7e2` / `#a9a69f` / `#908d86`. Every text colour is at least
  4.6:1 on every surface (checked).
- **One accent:** signal amber `#e7a33e`, used only for focus, selection,
  the primary action and active navigation.
- **Severity:** critical `#f06368`, high `#e8743b`, medium `#d9b44a`,
  low `#9a9ca2`, info `#8f9198`. Always printed with the word, never colour alone.
- **Evidence classes:** model `#7aa7d9`, on-chain `#9fb98a`, network `#b59ad6`,
  watchlist `#d98f6a`, rule `#c9b98a`. Always printed with the class name.
- **Type:** IBM Plex Sans 12-16px for UI; IBM Plex Mono with tabular figures
  for every address, txid, hash and number. Scale 11/12/13/14/16/20/26.
- **Space:** 4px grid.
- **Radius:** 2px chips, 4px controls, 6px panels and overlays. Nothing rounder.
- **Elevation:** a shadow only on overlays (menus, dialogs, drawers).
- **Motion:** 120ms `cubic-bezier(0.2, 0, 0, 1)` on hover, press and focus;
  zero under `prefers-reduced-motion`.

## Rules

1. **Model output is never styled as a fact.** Model evidence carries the
   MODEL class and the word "association"; on-chain facts carry ON-CHAIN.
   Scores show the model version beside them.
2. **Result types are labelled.** DEVELOPMENT, CONFIRMATION, HOLDOUT and
   PRODUCTION metrics are never shown unlabelled or side by side without a label.
3. **No fabricated data.** An empty state says what is missing and which
   command or action produces it.
4. **Keyboard first.** Every control is a real button, link or input with a
   visible amber focus ring; the graph has button equivalents for every
   gesture.

## Known asset gap

No logo exists. The identity is a typographic wordmark until a real mark is
supplied. No drawn substitute is used.

## Information architecture

The rail follows the investigation workflow. Role decides which groups are
drawn; the backend re-checks every request.

| Group | Route | What it is |
|---|---|---|
| Investigate | `/` | Overview: your cases, top of the reference alert run, serving model |
| | `/alerts`, `/alerts/:id` | Alert queue and alert detail (money flow, why flagged, evidence) |
| | `/investigations`, `/inv/:id/*` | Casework: overview, alerts, money flow, timeline, network, evidence, notes, report, review, history |
| | `/graph` | Graph explorer: trace from addresses, transactions or an alert |
| | `/entity/:address`, `/tx/:txid` | Address and transaction intelligence |
| Review | `/reviewer` | Review queue (reviewer, admin) |
| Intelligence | `/models` | Model registry, evaluation by result type, holdout, gate, drift |
| | `/evaluation` | Synthetic control evaluation (labelled SYNTHETIC) |
| Administration | `/admin/*` | Deployment facts, users, datasets, audit log |

Search (`/` or Cmd/Ctrl-K) queries `/api/search` and the user's cases. It
shows nothing the API did not return.

## Graph

- Library: Cytoscape.js with `cytoscape-dagre` (flow: left to right, value
  moving rightwards) and `cytoscape-fcose` (explore). Loaded on demand, so the
  entry bundle does not carry them.
- Data: `/api/graph/trace` (address/transaction seeds, direction, hops, node
  cap, time window) and `/api/alerts/{id}/graph` (members traced, plus the
  cluster and relay peers). Both come from the offline chain index
  (`make run ARGS="build-chain-index"`); no class labels are in it.
- Flow layout ranks only value-flow elements (addresses, transactions,
  SPENDS/PAYS). Clusters sit left of the flow, relay peers right of it.
  Graphs over 120 nodes open in the explore layout, because a wide fan-out
  ranks into a column no screen can show.
- Interactions: select, double-click to expand one hop, expand upstream or
  downstream, show sources/destinations (predecessors/successors over the
  loaded graph), shortest path between two nodes (directed first, undirected
  only when no directed path exists, and labelled as such), filters by kind,
  model risk, flagged-only and timestep, find in graph, zoom, fit, re-layout,
  reset. Every gesture has a button; the element list is the text alternative.
- Encoding: ring colour is model severity, a halo is watchlist attribution,
  amber is the seed. Legend and inspector name the evidence class of each.

## Measured performance (headless Chrome, 1440x900, local API)

| Case | Layout | Nodes / edges | Layout time |
|---|---|---|---|
| Address, 1 hop | flow | 49 / 51 | 23 ms |
| Address, 2 hops, cap 600 | explore | 600 / 751 | about 180-220 ms |
| Address, 3 hops, cap 1,500 | explore (draft) | 1,500 / 1,746 | 122 ms |
| Alert, 2 hops, cap 1,500 | explore | 676 / 966 | about 410 ms |

Read with `performance.getEntriesByName("oc-graph-layout")`. API warm-cache:
alert graph about 0.1 s, address profile about 0.03 s, 3-hop trace to 1,500
nodes about 0.22 s (`tests/test_investigation_real_data.py` pins budgets).
Entry bundle 366 kB (117 kB gzip); pages and the graph libraries are split.
