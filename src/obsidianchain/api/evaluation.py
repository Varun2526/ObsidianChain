"""GET /api/evaluation/synthetic - the controlled synthetic evaluation.

What this serves, and why it is kept apart
------------------------------------------
The coherent synthetic world is a SEPARATE artifact tree with its own
provenance. Nothing it contains is a measurement of Bitcoin, and nothing it
contains may be confused with the production analytical run - the two never
share an endpoint, a fingerprint or a page.

So this route exists to make the controlled results visible WITHOUT letting
them leak into the investigator's case work. Every response is stamped
``SYNTHETIC_CONTROL``, carries the banner text the UI renders, and is served
from a different path than anything a case reads.

Reads two files the ``synthetic-world-build`` and ``synthetic-world-overlap``
commands wrote. Computes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from obsidianchain.api import artifacts

#: Where ``synthetic-world-build`` puts the world, relative to the data root.
WORLD_DIRNAME = "synthetic_world"
MANIFEST = "world_manifest.json"
OVERLAP = "overlap_experiment.json"

BANNER = (
    "SYNTHETIC EVALUATION. Every figure below is a property of a generated "
    "world built to exercise the detectors under known conditions. None of "
    "it is a measurement of Bitcoin, none of it describes the production "
    "analytical run, and no alert or investigation draws on it."
)

NOT_GENERATED = (
    "The controlled synthetic evaluation has not been generated for this "
    "deployment. Run 'make run ARGS=\"synthetic-world-build\"' followed by "
    "'make run ARGS=\"synthetic-world-overlap\"'. Nothing is claimed in its "
    "absence."
)


def world_root(root=None) -> Path:
    return artifacts.data_root(root) / WORLD_DIRNAME


def _read(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        # A corrupt evaluation file is reported as absent rather than raised.
        # These are controlled results with no bearing on any case; failing a
        # request over one would be disproportionate, and claiming a number
        # from a file that would not parse would be worse.
        return None


def get_synthetic_evaluation(root=None) -> dict:
    """The world manifest and the overlap experiment, or an honest absence."""
    base = world_root(root)
    manifest = _read(base / MANIFEST)
    experiment = _read(base / OVERLAP)

    if manifest is None:
        return {
            "available": False,
            "provenance_type": "SYNTHETIC_CONTROL",
            "banner": BANNER,
            "detail": NOT_GENERATED,
        }

    return {
        "available": True,
        "provenance_type": "SYNTHETIC_CONTROL",
        "banner": BANNER,
        "world": {
            "generator_version": manifest.get("generator_version"),
            "network_generator_version": manifest.get(
                "network_generator_version"
            ),
            "world_fingerprint": manifest.get("world_fingerprint"),
            "seed": manifest.get("seed"),
            "created_at": manifest.get("created_at"),
            "counts": manifest.get("counts"),
            "behaviours": manifest.get("behaviours"),
            "adversarial_behaviours": manifest.get("adversarial_behaviours"),
            "positive_class": manifest.get("positive_class"),
            "positive_class_meaning": manifest.get("positive_class_meaning"),
            "behaviour_meaning": manifest.get("behaviour_meaning"),
            "notes": manifest.get("notes"),
        },
        # Present only once the overlap command has run. Absent is absent.
        "overlap": (
            {
                "meaning": experiment.get("meaning"),
                "abstention_meaning": experiment.get("abstention_meaning"),
                "levels": experiment.get("levels"),
                "notes": experiment.get("notes"),
            }
            if experiment else None
        ),
        "layer_comparison": (
            experiment.get("layer_comparison") if experiment else None
        ),
        "experiment_available": experiment is not None,
    }
