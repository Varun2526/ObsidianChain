"""What the API package may not import, and why.

Declared as data so a test can check it, rather than as a convention nobody
verifies. The Phase 4 audit found the truth-isolation whitelist had exactly
that failure mode: a manual list meant a new package defaulted to unchecked.

Two separate bans, for two separate reasons.

FORBIDDEN_RECOMPUTATION
    Modules whose purpose is to derive something. Importing one does no harm
    by itself, but it puts a recomputation path one call away from a request
    handler, and the whole guarantee of this layer is that a response is a
    file that was already written. Having the import be impossible is
    cheaper than reviewing every handler forever.

FORBIDDEN_FIELDS
    Evaluation-only columns. ``phase33_decisions.csv`` carries
    ``truth_category``, ``entities_a`` and ``entities_b`` - generated entity
    labels. Serving them would put truth into a presentation path for the
    first time in this project, where it would read as a finding.

A note on what is deliberately NOT listed here
----------------------------------------------
The shouted accessor names and the quarantined directory names are checked
by ``tests/test_api_boundary.py``, not by this module. They are scan patterns
for a source check, and putting them here would mean this file contained the
very strings that ``tests/test_truth_isolation.py`` enumerates every module
for - so the module that refuses truth would be indistinguishable from one
that reads it. Excusing this file from that check was the alternative, and it
would have put a hole in the mechanism Phase 5.1 exists to close.

The columns of ``ground_truth.csv`` (``true_entity_id``, ``true_origin_id``,
``broadcaster_flag``) are likewise absent, for a stronger reason: the API has
no path to that file at all. It cannot import
:mod:`obsidianchain.network.boundary`, which is the only loader, and
``test_the_import_graph_is_exactly_what_was_reviewed`` pins that.
"""

from __future__ import annotations

#: Import prefixes the API package must never reach for.
FORBIDDEN_RECOMPUTATION = (
    "obsidianchain.cluster.pipeline",
    "obsidianchain.cluster.replay",
    "obsidianchain.cluster.constrained",
    "obsidianchain.cluster.unionfind",
    "obsidianchain.cluster.change",
    "obsidianchain.cluster.index",
    "obsidianchain.network.separation",
    "obsidianchain.network.arrivals",
    "obsidianchain.network.audit",
    "obsidianchain.io.elliptic",
    "obsidianchain.eval",
)

#: Symbols and paths that carry evaluation-only truth.
#: Response fields that must never appear, whatever artifact they came from.
#: Checked against the payload, not only against the source, because a field
#: can arrive through data as easily as through an import.
FORBIDDEN_FIELDS = (
    "truth_category",
    "entities_a",
    "entities_b",
    "entity_assignment",
)


def assert_no_truth_fields(payload, path: str = "$") -> None:
    """Raise if any object in ``payload`` carries an evaluation-only field.

    Walks the whole structure. A denylist on the top level would miss a field
    nested inside a decision record, which is exactly where the Phase 3.3
    truth columns live.
    """
    if isinstance(payload, dict):
        for field in FORBIDDEN_FIELDS:
            if field in payload:
                raise TruthLeakInResponseError(
                    f"{path}.{field} is evaluation-only ground truth and must "
                    f"not reach a response. It is generated, never observed, "
                    f"and a consumer would read it as a finding."
                )
        for key, value in payload.items():
            assert_no_truth_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_no_truth_fields(value, f"{path}[{index}]")


class TruthLeakInResponseError(AssertionError):
    """Raised when evaluation-only truth would have been served."""
