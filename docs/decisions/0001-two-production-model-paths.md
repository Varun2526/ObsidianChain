# ADR 0001 — Two production model paths, with declared scopes

**Date:** 2026-09-22 · **Status:** Accepted

## Context

The audit found two live ML paths and no statement of which was "the model":

| | Phase 6 | PS-native |
|---|---|---|
| Features | M0–M3 (43 cols), as-of-t | `ps_native_features/2` (24 cols), streaming as-of-t |
| Model | LightGBM + isotonic + TreeSHAP | RandomForest + isotonic |
| Unit | address, observed at its last active timestep | address, observed at its last active timestep |
| Source | Elliptic++ frozen dataset | the same raw files, re-derived through a canonical PS stream |
| Serves | the alert API, run `043ea584e99daf99` | `AnalysisRun` execution via the orchestrator |

Model comparison is meaningless while "the model" is ambiguous, and a
number from one path was at risk of being quoted against the other.

## Decision

**Both are production, with declared, non-overlapping scopes.**

- **Phase 6** owns the Elliptic++ *reference* artifacts. The alert queue, the
  SHAP explanations and the frozen run fingerprint all come from here.
- **PS-native** owns *uploaded-dataset* runs — what the orchestrator executes
  when an investigator supplies their own capture.

**Numbers are never compared across the two.** `ml/protocol.py` enforces
this: a `CandidateResult` carries a `scope`, and `compare()` /
`compare_family()` raise `ScopeMismatchError` on a mismatch. The default
scope is `undeclared`, which is also refused against a declared one, so
pooling cannot happen by omission.

## Why not pick one

Retiring Phase 6 would orphan the alert artifacts every stored case
reference is bound to, and would discard the stronger methodology (proper
as-of-t feature construction, a real ablation ladder, TreeSHAP, leakage
tests). Retiring PS-native would leave the orchestrator with no model step
and break the upload path. Neither is a cost worth paying to resolve an
ambiguity that a declared boundary resolves for free.

## Consequences

- Each scope needs its own protocol run and its own baseline. A future
  "model B beats model A" claim is valid only within one scope.
- The feature-schema versions must stay distinct so a v1 model cannot be
  served v2 columns. `ps_model.py` already validates this before scoring.
- A reader seeing two different PR-AUC figures for "the model" is seeing two
  different models on two different feature sets, and the scope label is
  what tells them so.

## Alternatives rejected

**PS-native only** — orphans the alert artifacts and the case bindings.
**Phase 6 only** — breaks the upload path the console advertises.
**Merge the feature sets** — the two units are derived from different
streams with different as-of-t machinery; merging them would mean rebuilding
one on the other's substrate, which is a project, not a decision.
