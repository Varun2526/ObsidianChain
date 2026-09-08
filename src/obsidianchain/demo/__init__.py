"""Deterministic demonstration scenarios - SYNTHETIC, NOT A MEASUREMENT.

Everything in this package exists to show a judge the constraint mechanism
working end to end. None of it is a result. The two statements a viewer must
leave with are both true and both have to stay visible:

1. the mechanism works - a network cannot-link can veto a co-spend merge
   before it happens, and a contradiction discovered after the fact is
   recorded rather than silently kept;
2. **the frozen Elliptic++ dataset never triggers it.** Forty of 253,429
   proposed unions reached the pooled minimum and none of them separated.

A demonstration that hid (2) would be a lie of omission, so the flag that
marks this output as synthetic is carried structurally - through the API
envelope and into the UI - rather than written once in a caption. See
:mod:`obsidianchain.demo.api` for the flag contract and
:mod:`obsidianchain.demo.report` for the renderer that refuses unflagged
input.

Namespace
---------
Fixtures and outputs live under ``data/demo/`` and nowhere else.
:func:`obsidianchain.demo.scenarios.assert_demo_namespace` refuses any other
root, so a demo run cannot write into ``data/processed/`` even by mistake.
"""

from __future__ import annotations

from obsidianchain.demo.api import (
    DEMO_BANNER,
    DEMO_FLAG_FIELD,
    PROVENANCE,
    DemoFlagError,
    assert_demo_flagged,
    build_envelope,
)
from obsidianchain.demo.scenarios import (
    DEMO_DIRNAME,
    DEMO_SEED,
    SCENARIOS,
    DemoNamespaceError,
    DemoOutcome,
    ScenarioSpec,
    assert_demo_namespace,
    build_fixture,
)

__all__ = [
    "DEMO_BANNER",
    "DEMO_DIRNAME",
    "DEMO_FLAG_FIELD",
    "DEMO_SEED",
    "PROVENANCE",
    "SCENARIOS",
    "DemoFlagError",
    "DemoNamespaceError",
    "DemoOutcome",
    "ScenarioSpec",
    "assert_demo_flagged",
    "assert_demo_namespace",
    "build_envelope",
    "build_fixture",
]
