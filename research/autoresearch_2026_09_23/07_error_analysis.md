# 07 — Error Analysis (exp03_error_analysis, completed 2026-09-23)

Full results: `results/exp03_error_analysis.json`.

**Scope note (RULE 8):** this experiment uses a **single fold** (the last,
largest-history fold: `t<=38 -> t39-40`), not the full 12-fold protocol. It
happens to be the fold where LightGBM scored its minimum in exp01
(nAP 0.3033 — matches exp01's recorded `min` exactly), so this is a
**stress-case diagnostic**, not a ranking-quality claim. It is used here
because error analysis needs individual predictions, not fold-averaged
metrics, and picking the largest-history fold is the most production-
realistic single window to inspect. Every number below is read as "what
does a failure look like," not "how well does the model do."

## Precision/recall at realistic alert budgets

| Budget K | Precision | Recall |
|---|---|---|
| 20 | 0.200 | 0.007 |
| 50 | 0.420 | 0.037 |
| 100 | 0.540 | 0.096 |
| 200 | 0.615 | 0.218 |
| 500 | 0.448 | 0.396 |

Non-monotonic precision at K=20 (0.200, lower than K=50's 0.420) is
small-sample noise (4 positives among 20 items) — not a real inversion.
Precision peaks around K=200 then degrades by K=500 as the queue extends
into progressively less confident territory — expected shape for a ranker,
and a first, informal data point for the alert-budget question Phase 8
answers properly.

## What kinds of suspicious activity are being missed (false negatives)?

At K=100 (top 100 ranked addresses): 46 false positives, but **511 false
negatives** out of 565 total positives in this eval window (94.5°% recall
loss at this budget, consistent with a hard, high-prevalence-swing ranking
problem at 4.8% base rate).

**The clearest, most actionable finding in this program so far:** false
negatives are disproportionately **cold-start addresses with no prior
transaction history**.

| | n_txs_asof_t median | 25th pct | 75th pct |
|---|---|---|---|
| False negatives (n=511) | 0 | 0 | 0 |
| True positives at K=100 (n=54) | 1 | 1 | 1 |

**Three-quarters of the model's false negatives have literally zero prior
transactions as of the row being scored** — meaning every Group B
(address-history) feature is at its degenerate default for them, and the
model has almost nothing but the single transaction's own Group A shape to
work with. This connects directly to `06_ablation_results.md`: Group B was
already shown to be a real, if modest, marginal contributor — this error
analysis shows *where* that contribution matters most (distinguishing a
first-time illicit transaction from a first-time licit one), and *why*
losing it hurts recall specifically on the hardest, most operationally
important class of address: the one with no track record yet, which is
exactly the address an investigator would most want an early warning on.

## Score distribution

| | mean | median | tail |
|---|---|---|---|
| Positives | 0.319 | 0.145 | p10 = 0.0027 |
| Negatives | 0.051 | 0.015 | p90 = 0.123 |

Clear separation on average, but heavy overlap at the tails: the bottom 10%
of true positives score *below* 0.003 (indistinguishable from a typical
negative), and the top 10% of negatives score *above* 0.123 (higher than
the median positive). This overlap is exactly what caps precision@K in the
K=100–500 range above and is consistent with — not contradicting — the
"cold-start" finding: an address with a genuinely empty history is close to
un-rankable regardless of threshold, because Group A alone (per
`06_ablation_results.md`, `only_A` nAP 0.472 vs full 0.591) recovers most
but not all of the signal.

## What this means for the rest of the program

- **Phase 8 (alert policy):** any top-K or threshold policy inherits this
  ceiling — no calibration or threshold choice recovers recall on
  addresses the *features themselves* can't distinguish. This bounds what
  Phase 8 can be expected to fix.
- **Phase 4/6 (feature engineering), revisited:** the highest-value future
  feature work implied by this error analysis is **not** more graph/pattern
  columns (Group C/D, already shown weak in `06_ablation_results.md`) but
  something that helps at the cold-start address specifically — e.g.
  counterparty-side history (does *this* transaction touch an address or
  cluster with a track record, even if the subject address itself has
  none?), which is a genuinely new feature direction not covered by any
  existing Group A–D column. Not implemented or tested in this program —
  flagged as the clearest concrete lead for a future inner-loop cycle.
- **Phase 13 (gap analysis):** an investigative platform's alert language
  should distinguish "flagged from a rich history" from "flagged from a
  single transaction with no track record" — the current `ps_model.py`
  explanation output (§`01_repo_audit.md` §8) does not make this
  distinction, and arguably should, given how large a share of misses this
  is.

## IMPORTANT CORRECTION (2026-09-23, from exp08_counterparty_history_feature)

While building a counterparty-history feature to test this section's own
recommendation, a base-rate fact surfaced that **reframes the finding
above**: in the same diagnostic fold's eval set, **90.9% of ALL rows** (not
just false negatives) have `n_txs_asof_t == 0`. Cold-start is not a rare
failure-mode subpopulation — **it is the overwhelming default case for this
dataset**, a direct consequence of the last-snapshot-per-address,
boundary-spanner-dropping construction in `02_data_audit.md` (most
addresses in Elliptic++ simply don't recur often within one split window).

**Corrected reading**: the finding is not "false negatives are unusually
cold-start" in an absolute sense (almost everything is cold-start). The real
signal is **relative**: true positives caught at K=100 are disproportionately
drawn from the *rare* ~9% of rows that do have history (median
`n_txs_asof_t`=1 vs a population median of 0), meaning the model's success
cases lean on the thin slice of addresses with any track record at all, while
its performance across the dominant cold-start majority is closer to the
single-fold baseline (0.303 aggregate) than to its best-case behavior. This
is a more precise and more consequential statement than the original: **most
of this ranking problem is, structurally, a cold-start problem**, not an
edge case within an otherwise history-rich dataset. See
`04_feature_research.md`'s addendum and `exp08`'s results for the follow-up
this motivated.

## What this experiment does NOT show

- Whether this pattern holds across other folds — single-fold, as stated.
  A proper test would need the false-negative cold-start rate measured per
  fold and compared; not done here (flagged as a follow-up, not run to keep
  this diagnostic pass fast).
- Causal claims about *why* cold-start addresses are harder beyond the
  mechanical one (Group B features are degenerate for them) — no
  counterfactual feature-engineering experiment was run to confirm a new
  counterparty-history feature would actually help.
