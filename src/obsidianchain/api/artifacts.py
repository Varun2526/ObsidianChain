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

#: The production evidence funnel and the Stage 0 index it joins against.
#: Relative to the data root; the API never composes a path into raw/ or into
#: any *_truth/ directory.
EVIDENCE_FUNNEL = Path("processed") / "evidence_funnel.parquet"
ADDRESS_CLUSTERS = Path("processed") / "address_clusters.parquet"
CLUSTERS = Path("processed") / "clusters.parquet"

#: Phase 7. The alert artifacts, all written by one ``phase7-alerts`` run.
#: ``ALERTS`` carries the provenance the others are checked against; the four
#: detail tables are joined to it by ``alert_id`` and share its fingerprint.
ALERTS = Path("processed") / "alerts.parquet"
ALERT_TABLES = {
    "alert_members": Path("processed") / "alert_members.parquet",
    "alert_explanations": Path("processed") / "alert_explanations.parquet",
    "alert_timeline": Path("processed") / "alert_timeline.parquet",
    "alert_relationships": Path("processed") / "alert_relationships.parquet",
    "alert_network": Path("processed") / "alert_network.parquet",
}

#: Command that regenerates each, named in the error so an operator is not
#: left guessing. The API never runs these.
_BUILD_COMMAND = {
    EVIDENCE_FUNNEL: 'make run ARGS="evidence-funnel"',
    ADDRESS_CLUSTERS: 'make run ARGS="build-cluster-index"',
    CLUSTERS: 'make run ARGS="build-cluster-index"',
    ALERTS: 'make run ARGS="phase7-alerts"',
    **{p: 'make run ARGS="phase7-alerts"' for p in ALERT_TABLES.values()},
}


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


def _require(root, relative: Path) -> Path:
    path = data_root(root) / relative
    if not path.is_file():
        raise ArtifactMissingError(
            f"{path} not found. Run '{_BUILD_COMMAND[relative]}' first; the "
            f"API reads precomputed artifacts and never builds them on "
            f"request."
        )
    return path


def evidence_funnel_path(root=None) -> Path:
    return _require(root, EVIDENCE_FUNNEL)


def address_clusters_path(root=None) -> Path:
    return _require(root, ADDRESS_CLUSTERS)


def clusters_path(root=None) -> Path:
    return _require(root, CLUSTERS)


def load_evidence_funnel(root=None, columns=None):
    """Read the production evidence funnel, provenance-gated.

    A plain ``read_parquet``. Measured: 253,429 rows in a 3.8 MB file, so a
    column-projected read is a few milliseconds and DuckDB would add a
    dependency to solve a problem that does not exist. If that stops being
    true, measure before reaching for one.

    Returns ``(frame, sidecar)`` so a caller cannot end up holding rows whose
    provenance it never looked at.
    """
    import pandas as pd

    from obsidianchain.api import provenance_gate

    path = evidence_funnel_path(root)
    meta = provenance_gate.require_production(path)
    return pd.read_parquet(path, columns=columns), meta


def load_address_clusters(root=None, columns=None):
    """Read the Stage 0 address index, provenance-gated."""
    import pandas as pd

    from obsidianchain.api import provenance_gate

    path = address_clusters_path(root)
    meta = provenance_gate.require_production(path)
    return pd.read_parquet(path, columns=columns), meta


def load_clusters(root=None, columns=None):
    """Read the Stage 0 cluster summary, provenance-gated."""
    import pandas as pd

    from obsidianchain.api import provenance_gate

    path = clusters_path(root)
    meta = provenance_gate.require_production(path)
    return pd.read_parquet(path, columns=columns), meta


def alerts_path(root=None) -> Path:
    return _require(root, ALERTS)


def load_alerts(root=None, columns=None):
    """Read the ranked alert table, provenance-gated.

    Returns ``(frame, sidecar)`` so a caller cannot hold alerts whose
    provenance it never looked at - the same contract as the evidence funnel.
    """
    import pandas as pd

    from obsidianchain.api import provenance_gate

    path = alerts_path(root)
    meta = provenance_gate.require_production(path)
    return pd.read_parquet(path, columns=columns), meta


def load_alert_table(root=None, name: str = "alert_members", columns=None):
    """Read one of the four alert detail tables.

    Gated on its own sidecar, and additionally checked against the ranked
    table's fingerprint by ``provenance_gate.require_same_run`` at the route
    level: a detail table from a different run would join by ``alert_id``
    without error and silently describe a different cluster.
    """
    import pandas as pd

    from obsidianchain.api import provenance_gate

    if name not in ALERT_TABLES:
        raise ArtifactInvalidError(
            f"{name!r} is not an alert table; expected one of "
            f"{sorted(ALERT_TABLES)}"
        )
    path = _require(root, ALERT_TABLES[name])
    meta = provenance_gate.require_production(path)
    index_meta = provenance_gate.require_production(alerts_path(root))
    provenance_gate.require_same_run(index_meta, meta, name)
    return pd.read_parquet(path, columns=columns)
