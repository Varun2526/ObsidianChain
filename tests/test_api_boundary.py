"""The API package may not reach a recomputation or evaluation path.

Source-level and import-graph level, because the two catch different things.
A source scan catches an import that is written but never executed on the
path a test happens to exercise; an import-graph check catches something
pulled in transitively that no source line in this package names.

Why this matters more than it looks: the guarantee of the presentation layer
is that a response is a file the pipeline already wrote, under a rule a
manifest already records. A handler one call away from ``build_oracle`` could
produce a number that no manifest describes, and nothing downstream would be
able to tell.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from obsidianchain.api import boundary

API_DIR = Path(__file__).resolve().parents[1] / "src" / "obsidianchain" / "api"

#: Truth-access patterns the API source must not contain. These live here
#: rather than in the api package on purpose: a module holding these strings
#: is indistinguishable, to the enumeration in test_truth_isolation.py, from
#: one that actually reads truth. Keeping them on the test side lets the api
#: package be checked on merit instead of excused.
TRUTH_ACCESS_PATTERNS = (
    "FOR_EVALUATION_ONLY",
    "network_truth",
    "worlds_truth",
    # The synthetic world's quarantined truth: entity identity, origin
    # identity and the behaviour that produced each transaction.
    "world_truth",
    "true_entity_id",
    "true_origin_id",
)


def truth_isolation_module():
    """Load the truth-isolation test module by path.

    Imported by path rather than by name because pytest's rootdir-relative
    import of test modules is not guaranteed to put them on sys.path, and
    this assertion is worth making robustly: it is the check that the
    inverted mechanism actually covers this package.
    """
    path = Path(__file__).resolve().parent / "test_truth_isolation.py"
    spec = importlib.util.spec_from_file_location("_ti_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def api_modules() -> list[Path]:
    return sorted(API_DIR.rglob("*.py"))


def imported_names(path: Path) -> set[str]:
    """Every module name this file imports, at any indentation.

    Parsed rather than grepped so a lazy import inside a function is caught
    exactly like a top-level one - a deferred import is still an import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_the_api_package_is_not_empty() -> None:
    """Guard against a vacuous suite if the package is ever moved."""
    assert api_modules(), f"no modules found under {API_DIR}"


@pytest.mark.parametrize("path", api_modules(), ids=lambda p: p.name)
def test_no_module_imports_a_recomputation_path(path: Path) -> None:
    offenders = sorted(
        name
        for name in imported_names(path)
        for forbidden in boundary.FORBIDDEN_RECOMPUTATION
        if name == forbidden or name.startswith(forbidden + ".")
    )
    assert not offenders, (
        f"{path.name} imports {offenders}. The API serves precomputed "
        f"artifacts; a recomputation path must not be reachable from a "
        f"request handler."
    )


@pytest.mark.parametrize("path", api_modules(), ids=lambda p: p.name)
def test_no_module_imports_an_evaluation_module(path: Path) -> None:
    offenders = sorted(
        name for name in imported_names(path)
        if name.startswith("obsidianchain.eval")
    )
    assert not offenders, f"{path.name} imports evaluation code: {offenders}"


@pytest.mark.parametrize("path", api_modules(), ids=lambda p: p.name)
def test_no_module_references_a_truth_loader(path: Path) -> None:
    """The shouted accessors, and the quarantined directory names.

    Docstrings are stripped with the same ``code_only`` the project already
    uses, because a module may legitimately *discuss* truth in prose - the
    convention separation.py established when it explained at length that it
    must not read it. Only executable references matter.
    """
    code_only = truth_isolation_module().code_only
    executable = code_only(path.read_text(encoding="utf-8"))
    offenders = [m for m in TRUTH_ACCESS_PATTERNS if m in executable]
    assert not offenders, (
        f"{path.name} references truth access: {offenders}. No module in "
        f"this package may name an accessor or a quarantined directory, not "
        f"even to refuse it - see api/boundary.py on why."
    )


def test_the_denylist_is_not_empty() -> None:
    """A vacuous denylist would make every test above pass trivially."""
    assert len(boundary.FORBIDDEN_RECOMPUTATION) >= 8
    assert len(TRUTH_ACCESS_PATTERNS) >= 3
    assert len(boundary.FORBIDDEN_FIELDS) >= 3


def test_the_denylist_names_modules_that_exist() -> None:
    """A typo in the denylist would silently protect nothing."""
    src = API_DIR.parent
    for name in boundary.FORBIDDEN_RECOMPUTATION:
        relative = name.replace("obsidianchain.", "").replace(".", "/")
        assert (src / f"{relative}.py").is_file() or (src / relative).is_dir(), (
            f"{name} is on the denylist but no such module exists; the entry "
            f"protects nothing"
        )


# ---- the live import graph ----------------------------------------------


#: Every obsidianchain module reachable by importing the API. Pinned so that
#: growth is a visible, reviewed change rather than a silent one.
#:
#: network.synthetic appears because ``obsidianchain/demo/__init__.py``
#: imports ``demo.scenarios``, which imports it, and Python always runs a
#: package's __init__ on the way to a submodule. It is the GENERATOR that
#: wrote the demo fixture, not a recomputation or evaluation path, and the
#: API never calls it. Recorded rather than hidden.
EXPECTED_IMPORT_GRAPH = {
    "obsidianchain",
    "obsidianchain.api",
    "obsidianchain.api.app",
    "obsidianchain.api.artifacts",
    "obsidianchain.api.boundary",
    "obsidianchain.api.demo",
    # Phase 5.2, reviewed: the evidence handler and the provenance gate are
    # both read-only presentation code, and run_fingerprint is stdlib-only
    # (hashlib + pathlib, asserted in tests/test_run_fingerprint.py) so it
    # carries nothing this layer is forbidden.
    "obsidianchain.api.evidence",
    "obsidianchain.api.provenance_gate",
    "obsidianchain.run_fingerprint",
    # Phase 5.3-B, reviewed: the artifact's public contract - schema string,
    # reason codes, row-config labels and the frozen wordings. Standard
    # library only (asserted in tests/test_evidence_contract.py), and it
    # exists precisely so the API can name these without importing eval/.
    "obsidianchain.evidence_contract",
    # Phase 7, reviewed: the alert layer's read path and its public
    # contract. Both are presentation code over precomputed artifacts.
    # api.alerts scores nothing - obsidianchain.ml, obsidianchain.features
    # and obsidianchain.alerts.build are all on FORBIDDEN_RECOMPUTATION, so
    # neither the model nor the feature builders are reachable from a
    # handler. alerts.contract and alerts.feature_groups are standard
    # library only (asserted in tests/test_phase7_alerts.py); the latter
    # exists precisely so the API can name a feature's group without
    # importing the builders that define it.
    # Phase 9, reviewed: ingestion validates an uploaded file and reports on
    # it. That is not recomputation - it produces no score, touches no
    # artifact, and cannot be done offline because the file does not exist
    # until the request arrives. io.ingest is a parser (pandas, json,
    # xml.etree) and imports nothing from this project. geoip resolves an
    # address against the IANA special-purpose registry and is standard
    # library only.
    "obsidianchain.api.ingest",
    "obsidianchain.io",
    "obsidianchain.io.ingest",
    "obsidianchain.geoip",
    "obsidianchain.alerts",
    "obsidianchain.alerts.contract",
    "obsidianchain.alerts.feature_groups",
    "obsidianchain.api.alerts",
    "obsidianchain.demo",
    "obsidianchain.demo.api",
    "obsidianchain.demo.scenarios",
    "obsidianchain.network",
    "obsidianchain.network.synthetic",
    "obsidianchain.provenance",
    # Phase 10, reviewed: the alert -> separation-evidence join. Reads three
    # artifacts the API already loads (alerts, address_clusters,
    # evidence_funnel) and selects rows by address code. It derives no
    # verdict - the production verdict and reason code are read from the
    # persisted columns - and it refuses the join rather than guessing when
    # the two artifacts cannot be shown to share a cluster space.
    "obsidianchain.api.separation",
    # Phase 11, reviewed: structural patterns behind one alert. Reads the
    # alert tables and the ADDITIVE tx_mixing.parquet and groups rows. It
    # classifies nothing - obsidianchain.features.mixing made the
    # classification offline and is on FORBIDDEN_RECOMPUTATION, which is why
    # the mixing vocabulary lives in alerts/contract.py where this layer can
    # name a class without the detector being reachable from a handler.
    "obsidianchain.api.patterns",
    # Phase 11, reviewed: the controlled synthetic evaluation. Reads two JSON
    # files the world commands wrote and reshapes them. It imports nothing
    # from obsidianchain.world - the generator is not reachable from a
    # request - and the payload is stamped SYNTHETIC_CONTROL so it can never
    # be read as the production run.
    "obsidianchain.api.evaluation",
    # Frontend redesign, reviewed: the investigation read path and model
    # intelligence. api.investigation reads processed/chain_edges.parquet,
    # chain_transactions.parquet and watchlist_seeds.parquet, which the
    # offline 'build-chain-index' command writes with a PRODUCTION sidecar
    # (obsidianchain/chain_index.py); it never opens raw/ and serves no class
    # labels (the index contains none, asserted in tests/test_investigation_api.py).
    # Tracing is a bounded walk over that written graph - traversal, not
    # scoring. api.models reads the registry and evaluation JSON the ML
    # pipeline already wrote, and deliberately does not import
    # obsidianchain.ml (still on FORBIDDEN_RECOMPUTATION).
    "obsidianchain.api.investigation",
    "obsidianchain.api.models",
    # Phase 10, reviewed: the application layer. These modules hold MUTABLE
    # investigator state - users, sessions, cases, datasets, dispositions,
    # notes, reports, audit - in SQLite, and they are mounted on the same
    # FastAPI application so one process serves one origin.
    #
    # They are reachable from the API package and that is the point to
    # examine, so: none of them computes an analytical quantity, none copies
    # one, and none is on FORBIDDEN_RECOMPUTATION. What they store about the
    # analytical layer is identifiers - alert_id, run_fingerprint - so a
    # score still lives in exactly one place. console.casework and
    # console.deps read the analytical layer through api.alerts and
    # api.artifacts, which are already in this set; console.datasets
    # validates an upload through api.ingest, which is too.
    #
    # tests/test_console_boundary.py applies the same source-level denylist
    # to this package that this file applies to obsidianchain.api.
    "obsidianchain.console",
    "obsidianchain.console.audit",
    "obsidianchain.console.casework",
    "obsidianchain.console.datasets",
    "obsidianchain.console.db",
    "obsidianchain.console.deps",
    "obsidianchain.console.errors",
    "obsidianchain.console.integrity",
    "obsidianchain.console.investigations",
    "obsidianchain.console.passwords",
    "obsidianchain.console.rbac",
    "obsidianchain.console.reports",
    "obsidianchain.console.run_alerts",
    "obsidianchain.console.routes_auth",
    "obsidianchain.console.routes_casework",
    "obsidianchain.console.routes_investigations",
    "obsidianchain.console.runs",
    "obsidianchain.console.sessions",
    "obsidianchain.console.users",
}


#: Probe run in a SUBPROCESS, not in-process.
#:
#: The in-process version of this deleted obsidianchain entries from
#: sys.modules so the graph could be measured from a clean slate, and that
#: corrupted every other test module in the same session: their module-level
#: imports still referenced the old objects, so enum identity and isinstance
#: checks began failing in test_demo.py with errors that had nothing to do
#: with the demo. A subprocess gives a genuinely clean interpreter and
#: cannot reach into this one.
_PROBE = """
import json, sys
from pathlib import Path
from obsidianchain.api.app import create_app
from obsidianchain.api import demo as api_demo
try:
    api_demo.get_demo_scenarios(Path("/nonexistent-root-for-probe"))
except Exception:
    pass
print(json.dumps(sorted(n for n in sys.modules if n.startswith("obsidianchain"))))
"""


def _graph_after_serving() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, check=True,
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_the_import_graph_contains_no_forbidden_module() -> None:
    graph = _graph_after_serving()
    offenders = sorted(
        name for name in graph
        for forbidden in boundary.FORBIDDEN_RECOMPUTATION
        if name == forbidden or name.startswith(forbidden + ".")
    )
    assert not offenders, f"reachable by import: {offenders}"


def test_the_import_graph_is_exactly_what_was_reviewed() -> None:
    """Pinned, so a new transitive dependency has to be looked at.

    Failing here is not necessarily a bug - it means the reachable set grew
    and somebody should decide whether the new module belongs in a read-only
    presentation layer.
    """
    graph = _graph_after_serving()
    unexpected = sorted(graph - EXPECTED_IMPORT_GRAPH)
    assert not unexpected, (
        f"the API import graph grew: {unexpected}. Review whether these "
        f"belong in a read-only layer, then update EXPECTED_IMPORT_GRAPH."
    )


def test_no_evaluation_module_is_reachable() -> None:
    graph = _graph_after_serving()
    assert not [n for n in graph if n.startswith("obsidianchain.eval")]


# ---- truth isolation covers the api package -----------------------------


def test_truth_isolation_enumerates_the_api_package() -> None:
    """The inverted mechanism must actually be picking these files up.

    Phase 4 audit §6.5: the old whitelist meant a new package defaulted to
    unchecked. This asserts the inversion landed, by importing the test
    module's own enumeration rather than trusting that it works.
    """
    ti = truth_isolation_module()

    checked = set(ti.inference_modules())
    api_relative = {
        f"api/{p.name}" for p in api_modules() if p.name != "__init__.py"
    }
    missing = api_relative - checked
    assert not missing, (
        f"{sorted(missing)} are not covered by the truth-isolation "
        f"enumeration; the api package would be unchecked"
    )


def test_the_api_package_is_not_excused_from_truth_isolation() -> None:
    """Coverage by enumeration, not by an exemption that looks like coverage."""
    ti = truth_isolation_module()

    for module in ti.EXCUSED:
        assert not module.startswith("api/"), (
            f"{module} is excused from the truth ban; the api package must "
            f"be checked, not whitelisted"
        )
