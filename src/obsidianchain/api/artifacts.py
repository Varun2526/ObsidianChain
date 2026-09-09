"""Locate and load precomputed artifacts. Never builds one.

Every loader here raises when its artifact is missing, and the message names
the command that produces it. That is deliberate: a loader that built on a
cache miss would turn a stray request into a two-second clustering pass and,
worse, would produce numbers no manifest describes.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Same environment variable the CLI and the loaders already use, so the API
#: reads the data root the pipeline wrote to without a second convention.
DEFAULT_DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))

#: The Phase 3.4 demonstration payload, relative to the data root.
DEMO_SCENARIOS = Path("demo") / "output" / "scenarios.json"


class ArtifactMissingError(FileNotFoundError):
    """Raised when a precomputed artifact the API needs is not on disk."""


class ArtifactInvalidError(ValueError):
    """Raised when an artifact exists but is not what it claims to be."""


def data_root(override=None) -> Path:
    return Path(override) if override is not None else DEFAULT_DATA_ROOT


def demo_scenarios_path(root=None) -> Path:
    return data_root(root) / DEMO_SCENARIOS


def load_demo_scenarios(root=None) -> dict:
    """Read the persisted Phase 3.4 demonstration payload.

    Validation is delegated to :func:`obsidianchain.demo.api.read_json`, which
    already walks every object in the envelope and refuses one that is not
    flagged ``demo: true`` with the right provenance string. Reusing it is the
    point - a second validator here could drift from the one the writer uses,
    and then the API and the demo would disagree about what a demonstration
    payload is.

    These are the Phase **3.4** demonstration scenarios. They are not the
    Phase 3.3 controlled worlds, which are ``SYNTHETIC_CONTROL`` rather than
    ``DEMO`` and are a real experiment on generated data.
    """
    from obsidianchain.demo import api as demo_api

    path = demo_scenarios_path(root)
    if not path.is_file():
        raise ArtifactMissingError(
            f"{path} not found. Run 'make demo' to generate the "
            f"demonstration payload; the API does not build it on request."
        )
    try:
        payload = demo_api.read_json(path)
    except demo_api.DemoFlagError as exc:
        raise ArtifactInvalidError(
            f"{path} is not a valid demonstration payload: {exc}"
        ) from exc
    except ValueError as exc:  # json.JSONDecodeError subclasses ValueError
        raise ArtifactInvalidError(
            f"{path} is not readable as JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ArtifactInvalidError(
            f"{path} holds {type(payload).__name__}, expected an object"
        )
    return payload
