"""The operational console: MUTABLE investigator state.

The separation this package exists to enforce
---------------------------------------------
``obsidianchain.api`` serves IMMUTABLE ANALYTICAL TRUTH - alerts, risk
scores, SHAP explanations, M0-M3 evidence, network observations. Every
response there is a file the offline pipeline already wrote, under a rule a
manifest already records. Nothing in that layer has an owner, and nothing in
it can be changed by a request.

This package holds the other half: MUTABLE INVESTIGATOR STATE - who is
logged in, which case they own, what they uploaded, which alerts they
referenced, what they decided and why, what they wrote down, and what they
reported. All of it is attributable to a person and none of it is analytical
output.

The two never mix. This package REFERENCES analytical artifacts by their own
identifiers (``alert_id``, ``evidence_id``, ``run_fingerprint``) and copies
none of them. A cluster's risk score lives in exactly one place - the
parquet the pipeline wrote - and a case that wants to show it reads it from
there through the analytical API. Copying a score into SQLite would fork the
fingerprint discipline and let a case display a number no manifest
describes, which is the failure the whole provenance layer exists to
prevent.

Storage is one SQLite file beside the artifacts. That is a deliberate
choice, not a placeholder: this is an offline Linux application, SQLite
needs no server, it is transactional, and it backs up by copying a file.

Import discipline
-----------------
Like ``obsidianchain.api``, this package may not reach a recomputation or
evaluation path. ``tests/test_console_boundary.py`` checks that at source
level against the same denylist.
"""

from __future__ import annotations

__all__ = [
    "audit",
    "casework",
    "datasets",
    "db",
    "deps",
    "errors",
    "investigations",
    "passwords",
    "rbac",
    "reports",
    "runs",
    "sessions",
    "users",
]
