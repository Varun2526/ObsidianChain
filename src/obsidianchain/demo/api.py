"""The demonstration API: a JSON envelope that cannot lose its DEMO flag.

Why the flag is structural
--------------------------
A synthetic demonstration mislabelled as a result is the single worst thing
this project could put in front of a judge, and the usual defence - a caption
saying "synthetic" - fails the moment one panel, one screenshot or one copied
JSON fragment travels on its own. So the marking is attached to **every
object in the payload**, not to the document that contains them.

:func:`assert_demo_flagged` enforces exactly that: it walks the payload and
raises on any object that lacks ``demo: true`` and the provenance string.
The renderer in :mod:`obsidianchain.demo.report` calls it before drawing a
single pixel, which makes the flag load-bearing rather than decorative -
an unflagged payload does not render at all, so the marking cannot be
dropped by editing a template.

The two statements
------------------
Every envelope carries both, side by side, because a judge needs both to
read the demonstration correctly:

* ``mechanism`` - what these scenarios show the engine doing;
* ``frozen_dataset`` - that the frozen Elliptic++ run does not trigger any
  of it, with the numbers that say so.

They are fields, not prose in a header, so neither can be rendered without
the other.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The field every object in a demonstration payload must carry.
DEMO_FLAG_FIELD = "demo"

#: ...alongside this one.
PROVENANCE_FIELD = "provenance"

PROVENANCE = "SYNTHETIC_DEMONSTRATION"

DEMO_BANNER = "SYNTHETIC DEMONSTRATION - NOT A MEASUREMENT"

#: What the demonstration shows.
MECHANISM_STATEMENT = (
    "These five scenarios exercise the constraint engine end to end: a "
    "network cannot-link can veto a co-spend merge before it happens, and a "
    "contradiction found after a merge is recorded as CONTESTED rather than "
    "silently kept."
)

#: What it does not show, in the same envelope, with the numbers.
FROZEN_DATASET_STATEMENT = (
    "The frozen Elliptic++ dataset does not trigger any of this. Of 253,429 "
    "proposed unions, 40 reached the pooled minimum of 25 and none of them "
    "separated; the median component pools zero usable observations. Both "
    "statements are true and both are part of the result."
)


class DemoFlagError(AssertionError):
    """Raised when a payload object is missing its demonstration marking."""


def record(**fields) -> dict:
    """One payload object, flagged.

    Every dict that reaches the API goes through here. Building them by hand
    is how a flag goes missing.
    """
    return {DEMO_FLAG_FIELD: True, PROVENANCE_FIELD: PROVENANCE, **fields}


def assert_demo_flagged(payload, path: str = "$") -> None:
    """Raise unless every object in ``payload`` carries the demonstration flag.

    Walks dicts and lists to any depth. The check is deliberately absolute -
    there is no "internal" object exempt from it - because an exemption is
    the crack a mislabelled figure escapes through.
    """
    if isinstance(payload, dict):
        if payload.get(DEMO_FLAG_FIELD) is not True:
            raise DemoFlagError(
                f"{path} is missing {DEMO_FLAG_FIELD}=true. Every object in a "
                f"demonstration payload must be marked; got keys "
                f"{sorted(payload)}."
            )
        if payload.get(PROVENANCE_FIELD) != PROVENANCE:
            raise DemoFlagError(
                f"{path} has provenance {payload.get(PROVENANCE_FIELD)!r}, "
                f"expected {PROVENANCE!r}."
            )
        for key, value in payload.items():
            assert_demo_flagged(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_demo_flagged(value, f"{path}[{index}]")


def build_envelope(
    scenarios: list[dict],
    fixture: dict,
    rule: dict,
    totals: dict,
    seed: int,
    namespace: str,
) -> dict:
    """Assemble the payload the UI consumes.

    ``scenarios`` are already-flagged records from
    :mod:`obsidianchain.demo.runner`; everything else is assembled here so
    the top-level shape lives in one place.
    """
    envelope = record(
        schema="obsidianchain.demo/1",
        banner=DEMO_BANNER,
        # ProvenanceType.DEMO, spelled as a literal so this module keeps no
        # import it does not otherwise need. Asserted against the enum in
        # tests/test_provenance.py so the two cannot drift.
        provenance_type="DEMO",
        not_a_measurement=True,
        seed=seed,
        namespace=namespace,
        statements=record(
            mechanism=MECHANISM_STATEMENT,
            frozen_dataset=FROZEN_DATASET_STATEMENT,
        ),
        fixture=fixture,
        rule=rule,
        totals=totals,
        scenarios=scenarios,
    )
    assert_demo_flagged(envelope)
    return envelope


def write_json(envelope: dict, path) -> Path:
    """Write the envelope, refusing to write an unflagged one."""
    assert_demo_flagged(envelope)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return path


def read_json(path) -> dict:
    """Read an envelope back, refusing an unflagged one.

    The check runs on the way in as well as on the way out. A payload that
    lost its marking between processes is exactly the case worth catching,
    and it costs one tree walk.
    """
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_demo_flagged(envelope)
    return envelope
