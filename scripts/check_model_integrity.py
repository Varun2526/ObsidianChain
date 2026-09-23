"""CI check: production model artifacts are exactly what the registry says.

Fails when any registered artifact differs from its registered hash, when
the champion lacks a PASS production-gate report or an attested source
commit, or when champion and fallback do not match the live feature schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from obsidianchain.ml import registry
from obsidianchain.pipeline.features_ps import PS_FEATURE_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    reg = registry.Registry.open(ROOT / "data" / "models" / "ps_native")
    problems = []
    for version in sorted(reg.data["models"]):
        try:
            reg.verify(version)
        except registry.RegistryError as exc:
            problems.append(str(exc))
    champion, fallback = reg.role("champion"), reg.role("fallback")
    if champion is None:
        problems.append("no champion assigned")
    else:
        gate = ROOT / "data" / "models" / "ps_native" / "gates" / f"{champion}.json"
        if not gate.is_file() or json.loads(gate.read_text()).get("decision") != "PASS":
            problems.append(f"champion {champion} has no PASS gate report")
        if not reg.entry(champion).get("attested_source_commit"):
            problems.append(f"champion {champion} has no attested source commit")
        if not reg.entry(champion).get("holdout"):
            problems.append(f"champion {champion} has no locked holdout result")
    for role, version in (("champion", champion), ("fallback", fallback)):
        if version and reg.entry(version)["feature_schema_version"] != PS_FEATURE_SCHEMA_VERSION:
            problems.append(f"{role} {version} is not on the live schema {PS_FEATURE_SCHEMA_VERSION}")
    for p in problems:
        print("FAIL:", p)
    print(f"{len(reg.data['models'])} registered versions checked; champion={champion} fallback={fallback}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
