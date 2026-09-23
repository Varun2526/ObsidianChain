"""Model registry: immutable artifacts, roles, lineage, promotion and rollback.

``data/models/ps_native/registry.json`` is tracked in git. It is the only
authority on which artifact serves production, and it pins the SHA-256 of
every file of every registered version. A model directory's own
``manifest.json`` also carries hashes, but a manifest sitting beside its
artifact can be replaced together with it; the registry lives in version
control, so tampering with an artifact shows up as a hash mismatch AND as a
diff.

Roles
    champion   serves production decisions
    candidate  runs in shadow only: scored, recorded, compared, never used
    fallback   served if the champion fails verification or compatibility
Every role change is appended to ``history`` with a reason; nothing is
deleted from the registry.

Immutability
    ``register`` refuses a version that already exists with different file
    hashes. A retrained model is a new version.

Compatibility
    A model is loadable only if its ``feature_schema_version`` equals the
    live engine's. Rolling back to a model of an older schema therefore needs
    the code of that schema too; ``rollback`` refuses and says which.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "obsidianchain.model_registry/1"
#: Relative default, kept for callers that pass it explicitly. Serving uses
#: default_root(), which honours OBSIDIANCHAIN_DATA: in the container the
#: code runs from /app and the data is mounted at /data, so a cwd-relative
#: path would find no model at all.
DEFAULT_ROOT = Path("data") / "models" / "ps_native"


def default_root() -> Path:
    import os
    return Path(os.environ.get("OBSIDIANCHAIN_DATA") or "data") / "models" / "ps_native"
REGISTRY_FILE = "registry.json"
ROLES = ("champion", "candidate", "fallback")


class RegistryError(RuntimeError):
    """A registry operation that would break an invariant."""


class ArtifactVerificationError(RegistryError):
    """An artifact file is missing or does not match its registered hash."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hashes(model_dir: Path) -> dict[str, str]:
    return {p.name: file_sha256(p) for p in sorted(model_dir.iterdir()) if p.is_file()}


@dataclass
class Registry:
    root: Path
    data: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.root / REGISTRY_FILE

    @classmethod
    def open(cls, root: str | Path | None = None) -> "Registry":
        root = Path(root) if root is not None else default_root()
        path = root / REGISTRY_FILE
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != REGISTRY_SCHEMA:
                raise RegistryError(f"unknown registry schema {data.get('schema')!r}")
        else:
            data = {"schema": REGISTRY_SCHEMA, "models": {}, "roles": dict.fromkeys(ROLES), "history": []}
        return cls(root, data)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- queries -------------------------------------------------------

    def role(self, role: str) -> str | None:
        return self.data["roles"].get(role)

    def entry(self, version: str) -> dict[str, Any]:
        if version not in self.data["models"]:
            raise RegistryError(f"version {version!r} is not registered")
        return self.data["models"][version]

    def model_dir(self, version: str) -> Path:
        return self.root / self.entry(version)["path"]

    def verify(self, version: str) -> None:
        """Every registered file present and byte-identical. Raises otherwise."""
        entry, model_dir = self.entry(version), self.model_dir(version)
        for name, expected in entry["artifacts"].items():
            path = model_dir / name
            if not path.is_file():
                raise ArtifactVerificationError(f"{version}: {name} is missing")
            actual = file_sha256(path)
            if actual != expected:
                raise ArtifactVerificationError(
                    f"{version}: {name} sha256 {actual[:12]} does not match registered {expected[:12]}")

    # ---- mutations -----------------------------------------------------

    def register(self, version: str, rel_path: str, *, feature_schema_version: str,
                 lineage: dict[str, Any] | None = None, notes: str = "") -> dict[str, Any]:
        model_dir = self.root / rel_path
        if not (model_dir / "model.joblib").is_file():
            raise RegistryError(f"{model_dir} has no model.joblib")
        hashes = directory_hashes(model_dir)
        existing = self.data["models"].get(version)
        if existing is not None:
            if existing["artifacts"] != hashes:
                raise RegistryError(
                    f"{version} is already registered with different artifacts; "
                    f"artifacts are immutable - register a new version")
            return existing
        entry = {"path": rel_path, "feature_schema_version": feature_schema_version,
                 "artifacts": hashes, "registered_at": _now(), "lineage": lineage or {},
                 "notes": notes}
        self.data["models"][version] = entry
        self._log("REGISTER", version, notes or "registered")
        return entry

    def assign(self, role: str, version: str | None, reason: str) -> None:
        if role not in ROLES:
            raise RegistryError(f"unknown role {role!r}")
        if not reason.strip():
            raise RegistryError("a role change needs a written reason")
        if version is not None:
            self.verify(version)
        previous = self.data["roles"].get(role)
        self.data["roles"][role] = version
        self._log(f"ASSIGN_{role.upper()}", version, reason, previous=previous)

    def rollback(self, reason: str, live_schema: str) -> str:
        """Champion <- fallback. The old champion becomes the fallback."""
        fallback, champion = self.role("fallback"), self.role("champion")
        if fallback is None:
            raise RegistryError("no fallback is registered; nothing to roll back to")
        fb_schema = self.entry(fallback)["feature_schema_version"]
        if fb_schema != live_schema:
            raise RegistryError(
                f"fallback {fallback} needs feature schema {fb_schema}, the running code is "
                f"{live_schema}: roll back the code to that schema as well "
                f"(lineage.source_commit = {self.entry(fallback)['lineage'].get('source_commit')})")
        self.verify(fallback)
        self.data["roles"]["champion"], self.data["roles"]["fallback"] = fallback, champion
        self._log("ROLLBACK", fallback, reason, previous=champion)
        return fallback

    def attest(self, version: str, commit: str, read_blob) -> None:
        """Record ``commit`` as the verified source of ``version``.

        ``read_blob(commit, path) -> bytes`` returns a file as it is in that
        commit. Every file in the manifest's ``lineage.code_sha256`` must hash
        to its recorded value there, or nothing is recorded. The artifact
        hashes are untouched: attestation is metadata ABOUT the artifact.
        """
        import json as _json
        manifest = _json.loads((self.model_dir(version) / "manifest.json").read_text())
        code = manifest.get("lineage", {}).get("code_sha256") or {}
        if not code:
            raise RegistryError(f"{version} manifest records no code hashes to attest")
        for path, expected in code.items():
            actual = hashlib.sha256(read_blob(commit, path)).hexdigest()
            if actual != expected:
                raise RegistryError(f"{path} at {commit[:12]} hashes {actual[:12]}, the model was trained "
                                    f"from {expected[:12]}: not the training code")
        entry = self.entry(version)
        entry["attested_source_commit"] = commit
        self._log("ATTEST_SOURCE", version, f"code_sha256 of {len(code)} files verified at {commit}")

    def record_holdout(self, version: str, result_path: Path) -> None:
        """Attach the locked holdout result to a version (metadata, append-only)."""
        entry = self.entry(version)
        if entry.get("holdout"):
            raise RegistryError(f"{version} already has a holdout result; the holdout is not reopened")
        lock = result_path.with_suffix(".lock")
        body = result_path.read_bytes()
        if not lock.is_file() or lock.read_text().strip() != hashlib.sha256(body).hexdigest():
            raise RegistryError(f"{result_path} does not match its lock file")
        card = json.loads(body)["models"][version]["scorecard"]
        entry["holdout"] = {"result": str(result_path.relative_to(self.root)), "sha256": lock.read_text().strip(),
                            "nap": card["address"]["nap"], "P@100": card["address"]["P@100"],
                            "ece": card["calibration"]["ece"]}
        self._log("RECORD_HOLDOUT", version, f"holdout result {entry['holdout']['sha256'][:12]}")

    def _log(self, event: str, version: str | None, reason: str, previous: str | None = None) -> None:
        self.data["history"].append({"at": _now(), "event": event, "version": version,
                                     "previous": previous, "reason": reason})


def resolve(role: str = "champion", root: str | Path | None = None,
            live_schema: str | None = None) -> tuple[str, Path]:
    """The verified, schema-compatible model directory for ``role``.

    Raises ``RegistryError`` rather than returning something unverified.
    """
    reg = Registry.open(root)
    version = reg.role(role)
    if version is None:
        raise RegistryError(f"no {role} is assigned")
    entry = reg.entry(version)
    if live_schema is not None and entry["feature_schema_version"] != live_schema:
        raise RegistryError(f"{role} {version} declares {entry['feature_schema_version']}, "
                            f"the engine emits {live_schema}")
    reg.verify(version)
    return version, reg.model_dir(version)
