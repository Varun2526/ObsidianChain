"""Render docs/feature_catalog.md from contracts/features.CATALOG.

The catalog in code is what the feature contract enforces; this file is its
readable copy and is regenerated, never edited by hand. A test checks the
two agree (tests/test_docs_in_sync.py).
"""

from __future__ import annotations

from pathlib import Path

from obsidianchain.contracts.features import CATALOG, FEATURE_CONTRACT_VERSION
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS, PS_FEATURE_SCHEMA_VERSION

OUT = Path(__file__).resolve().parents[1] / "docs" / "feature_catalog.md"


def render() -> str:
    core = set(CORE_PS_FEATURE_COLUMNS)
    lines = [
        "# Feature catalog",
        "",
        f"Generated from `src/obsidianchain/contracts/features.py` ({FEATURE_CONTRACT_VERSION}); "
        f"engine schema `{PS_FEATURE_SCHEMA_VERSION}`. Do not edit by hand: run "
        "`python scripts/render_feature_catalog.py`.",
        "",
        "Information sets: `TX_ITSELF` = this transaction only; `PRIOR_EVENTS` = events strictly before "
        "this one in (timestamp, event_order); `PRIOR_EVENTS_AND_TX` = both. No feature reads a label.",
        "",
        "| feature | group | in model | information set | same-step behaviour | NaN means | range | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in CATALOG.values():
        lines.append(f"| `{s.name}` | {s.group} | {'yes' if s.name in core else 'no'} | {s.as_of} | "
                     f"{s.same_step} | {s.nullable or 'never (violation)'} | [{s.low}, {s.high}] | {s.notes} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT}")
