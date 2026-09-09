"""GET /api/demo/scenarios - the Phase 3.4 demonstration payload.

The payload is served with its structure intact. There is no response model
renaming its fields and no second copy of the scenario data: ``scenarios.json``
is the single source, written by ``make demo`` and read here.

Why no Pydantic model
---------------------
A model over this payload would have to enumerate every field of the envelope,
each scenario and each decision, and would then be a second definition of a
shape that :mod:`obsidianchain.demo.api` already defines and tests. The
failure it would prevent - a malformed payload - is already prevented by
``assert_demo_flagged``, which is stricter than a schema in the way that
matters here: it requires the DEMO marking on **every object**, at any depth,
rather than only on the fields a model happened to declare.

What is added on the way out
----------------------------
Nothing is removed and nothing is renamed. Two things are checked, and the
request fails rather than serving a payload that fails either:

* the four DEMO markers are present and correct at the envelope level, and
  the whole tree is verified flagged by ``read_json``;
* no evaluation-only truth field appears anywhere in the tree.
"""

from __future__ import annotations

from obsidianchain.api import artifacts, boundary

#: Envelope fields that must be present and correct before anything is served.
#: These are the markers a consumer relies on to know what it is looking at.
REQUIRED_DEMO_MARKERS = {
    "demo": True,
    "provenance": "SYNTHETIC_DEMONSTRATION",
    "provenance_type": "DEMO",
    "not_a_measurement": True,
}

#: The five Phase 3.4 scenarios. Checked so a truncated artifact is a failure
#: rather than a short list a consumer would read as "only three states exist".
EXPECTED_SCENARIO_KEYS = ("A", "B", "C", "D", "E")


class DemoProvenanceError(ValueError):
    """Raised when the payload is not unmistakably a demonstration."""


def assert_demo_provenance(payload: dict) -> None:
    """Refuse to serve anything a consumer could mistake for a measurement.

    ``read_json`` has already established that every object in the tree is
    flagged. This adds the envelope-level contract: the four markers a
    consumer actually reads, checked by value rather than by presence, so a
    payload with ``demo: false`` or a weakened provenance string is rejected
    rather than served with a reassuring-looking field.
    """
    for field, expected in REQUIRED_DEMO_MARKERS.items():
        if field not in payload:
            raise DemoProvenanceError(
                f"envelope is missing the DEMO marker {field!r}; refusing to "
                f"serve a payload a consumer could read as production"
            )
        if payload[field] != expected:
            raise DemoProvenanceError(
                f"envelope has {field}={payload[field]!r}, expected "
                f"{expected!r}. The DEMO marking must not be weakened."
            )

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise DemoProvenanceError(
            f"envelope 'scenarios' is {type(scenarios).__name__}, "
            f"expected a list"
        )
    keys = tuple(s.get("key") for s in scenarios if isinstance(s, dict))
    if keys != EXPECTED_SCENARIO_KEYS:
        raise DemoProvenanceError(
            f"expected the five Phase 3.4 scenarios "
            f"{EXPECTED_SCENARIO_KEYS}, got {keys}. A partial set would "
            f"misrepresent which engine states exist."
        )


def get_demo_scenarios(root=None) -> dict:
    """Load, validate, and return the persisted payload unchanged.

    Raises rather than degrading. A demonstration endpoint that quietly
    served an unflagged payload would be worse than one that returned 500.
    """
    payload = artifacts.load_demo_scenarios(root)
    assert_demo_provenance(payload)
    boundary.assert_no_truth_fields(payload)
    return payload
