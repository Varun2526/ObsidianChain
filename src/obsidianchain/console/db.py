"""The application database: connection, schema, migrations.

What lives here and what does not
---------------------------------
Mutable application state ONLY: users, sessions, investigations, datasets,
analysis runs, alert references, dispositions, notes, reports, audit events.

No analytical artifact is ever written to this database. Clusters, risk
scores, SHAP contributions, M0-M3 evidence and network observations stay in
the parquet files the pipeline wrote, addressed by ``alert_id`` and
``run_fingerprint``. This database stores those identifiers and nothing
else from the analytical layer.

Why SQLite
----------
Offline Linux application, single workstation, no server process, ACID
transactions, and a backup is ``cp``. A database server here would be
infrastructure with no requirement behind it.

Why the triggers
----------------
``audit_events`` and ``alert_dispositions`` are append-only, and that is
enforced in the schema rather than by convention in the route layer. A
forensic record whose immutability depends on nobody writing the wrong
handler is not immutable. The triggers make an UPDATE or DELETE fail at the
storage layer regardless of what the application does.

Why nothing cascades
--------------------
There is no delete path in this application. A case is CLOSED, never
removed, because removing it would destroy the audit trail that makes its
output defensible. The foreign keys are therefore plain references with no
``ON DELETE`` clause: a delete is refused by the constraint rather than
silently propagating through a table whose triggers would then abort it
halfway.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from contextlib import contextmanager
from pathlib import Path

#: The database file, relative to the data root the API already uses.
DB_FILENAME = "obsidianchain.sqlite3"

#: Uploaded datasets, relative to the data root. Content-addressed; see
#: :mod:`obsidianchain.console.datasets`.
UPLOAD_DIRNAME = "uploads"

#: Bumped by appending to MIGRATIONS, never by editing an applied entry.
SCHEMA_VERSION = 1


class StorageError(RuntimeError):
    """The application database could not be reached or is inconsistent."""


def utcnow() -> str:
    """One timestamp format across the whole application layer.

    ISO-8601 with an explicit ``Z``. Stored as TEXT because SQLite has no
    date type and a string that sorts lexicographically also sorts
    chronologically, which is what every ``ORDER BY`` here relies on.
    """
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


_V1 = """
CREATE TABLE users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL
                  CHECK (role IN ('ADMIN', 'INVESTIGATOR', 'REVIEWER')),
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at    TEXT NOT NULL
);

-- The cookie carries a random token; this table stores only its SHA-256.
-- Reading the database therefore does not yield a usable session.
CREATE TABLE sessions (
    token_sha256 TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id),
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    revoked_at   TEXT
);
CREATE INDEX idx_sessions_user ON sessions(user_id);

-- id is opaque and unguessable; case_number is the human label (OC-0001).
-- Two identifiers on purpose: an investigator needs a case number they can
-- say out loud, and a URL should not let anyone enumerate how many cases
-- exist or probe for someone else's.
CREATE TABLE investigations (
    id          TEXT PRIMARY KEY,
    case_number INTEGER NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    owner_id    TEXT NOT NULL REFERENCES users(id),
    status      TEXT NOT NULL CHECK (status IN (
                    'DRAFT', 'VALIDATING', 'ACTIVE', 'REVIEW', 'CLOSED')),
    -- The analytical run this case's referenced results came from. Set by an
    -- explicit authorised action (referencing an alert, or binding), never
    -- by rendering a page. Once set it does not change silently.
    bound_run_fingerprint TEXT,
    bound_run_at          TEXT,
    bound_by              TEXT REFERENCES users(id),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    closed_at   TEXT
);
CREATE INDEX idx_investigations_owner ON investigations(owner_id);

-- storage_path is derived from sha256 alone. The user-supplied filename is
-- metadata and never a path component; see console/datasets.py.
CREATE TABLE datasets (
    id                TEXT PRIMARY KEY,
    investigation_id  TEXT NOT NULL REFERENCES investigations(id),
    filename          TEXT NOT NULL,
    sha256            TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    format            TEXT NOT NULL,
    uploaded_by       TEXT NOT NULL REFERENCES users(id),
    uploaded_at       TEXT NOT NULL,
    storage_path      TEXT NOT NULL,
    validation_report TEXT NOT NULL,
    status            TEXT NOT NULL
                      CHECK (status IN ('RECEIVED', 'VALIDATED', 'REJECTED'))
);
CREATE INDEX idx_datasets_investigation ON datasets(investigation_id);

-- A run fingerprint may exist ONLY on a COMPLETE run. NOT_RUN carries NULL,
-- and the CHECK makes that structural rather than a convention: the honest
-- statement "this dataset has not been scored" cannot be typed away.
CREATE TABLE analysis_runs (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES datasets(id),
    status          TEXT NOT NULL CHECK (status IN (
                        'NOT_RUN', 'QUEUED', 'RUNNING', 'COMPLETE', 'FAILED')),
    run_fingerprint TEXT,
    started_at      TEXT,
    completed_at    TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL,
    CHECK (run_fingerprint IS NULL OR status = 'COMPLETE'),
    CHECK (status <> 'COMPLETE' OR run_fingerprint IS NOT NULL)
);
CREATE INDEX idx_runs_dataset ON analysis_runs(dataset_id);

-- A REFERENCE to a global immutable alert, never a copy. run_fingerprint is
-- recorded at the moment of referencing so a later regeneration makes the
-- reference detectably stale instead of silently re-pointing it.
CREATE TABLE alert_references (
    investigation_id TEXT NOT NULL REFERENCES investigations(id),
    alert_id         TEXT NOT NULL,
    run_fingerprint  TEXT NOT NULL,
    added_by         TEXT NOT NULL REFERENCES users(id),
    added_at         TEXT NOT NULL,
    assigned_to      TEXT REFERENCES users(id),
    assigned_at      TEXT,
    PRIMARY KEY (investigation_id, alert_id)
);

-- Append-only. A changed decision supersedes the previous row; it does not
-- overwrite it. A forensic record has to show that someone changed their
-- mind, not just where they ended up.
CREATE TABLE alert_dispositions (
    id               TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    alert_id         TEXT NOT NULL,
    state            TEXT NOT NULL CHECK (state IN (
                        'NEW', 'TRIAGED', 'IN_REVIEW',
                        'CONFIRMED', 'DISMISSED', 'ESCALATED')),
    rationale        TEXT NOT NULL DEFAULT '',
    decided_by       TEXT NOT NULL REFERENCES users(id),
    decided_at       TEXT NOT NULL,
    superseded_by    TEXT REFERENCES alert_dispositions(id),
    FOREIGN KEY (investigation_id, alert_id)
        REFERENCES alert_references(investigation_id, alert_id)
);
CREATE INDEX idx_dispositions_alert
    ON alert_dispositions(investigation_id, alert_id);

CREATE TRIGGER alert_dispositions_no_delete
BEFORE DELETE ON alert_dispositions
BEGIN
    SELECT RAISE(ABORT, 'alert_dispositions is append-only');
END;

-- The only permitted update is stamping superseded_by exactly once. Every
-- other column, and a second supersede, aborts.
CREATE TRIGGER alert_dispositions_supersede_once
BEFORE UPDATE ON alert_dispositions
WHEN OLD.superseded_by IS NOT NULL
     OR NEW.id               <> OLD.id
     OR NEW.investigation_id <> OLD.investigation_id
     OR NEW.alert_id         <> OLD.alert_id
     OR NEW.state            <> OLD.state
     OR NEW.rationale        <> OLD.rationale
     OR NEW.decided_by       <> OLD.decided_by
     OR NEW.decided_at       <> OLD.decided_at
BEGIN
    SELECT RAISE(ABORT,
        'alert_dispositions rows are immutable except for a one-time supersede');
END;

CREATE TABLE investigator_notes (
    id               TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id),
    alert_id         TEXT,
    body             TEXT NOT NULL,
    author_id        TEXT NOT NULL REFERENCES users(id),
    created_at       TEXT NOT NULL,
    updated_at       TEXT
);
CREATE INDEX idx_notes_investigation
    ON investigator_notes(investigation_id, id);

CREATE TABLE reports (
    id                TEXT PRIMARY KEY,
    investigation_id  TEXT NOT NULL REFERENCES investigations(id),
    version           INTEGER NOT NULL,
    title             TEXT NOT NULL,
    executive_summary TEXT NOT NULL DEFAULT '',
    content           TEXT NOT NULL DEFAULT '',
    run_fingerprint   TEXT,
    generated_by      TEXT NOT NULL REFERENCES users(id),
    generated_at      TEXT NOT NULL,
    content_sha256    TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('DRAFT', 'FINAL')),
    finalised_by      TEXT REFERENCES users(id),
    finalised_at      TEXT,
    UNIQUE (investigation_id, version)
);

-- Append-only, enforced by trigger. An audit log a user can edit is not an
-- audit log.
CREATE TABLE audit_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id         TEXT,
    action           TEXT NOT NULL,
    object_type      TEXT NOT NULL,
    object_id        TEXT,
    investigation_id TEXT,
    at               TEXT NOT NULL,
    detail_json      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_audit_investigation ON audit_events(investigation_id, id);
CREATE INDEX idx_audit_actor ON audit_events(actor_id, id);

CREATE TRIGGER audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;

CREATE TRIGGER audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit_events is append-only');
END;
"""

#: ``(version, script)`` in order. Append only - editing an applied entry
#: would leave already-migrated databases silently disagreeing with the code.
MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _V1),)


def database_path(data_root) -> Path:
    return Path(data_root) / DB_FILENAME


def connect(data_root, *, migrate_on_open: bool = True) -> sqlite3.Connection:
    """Open the application database, creating and migrating it if needed.

    One connection per request. SQLite connections are not safe to share
    across threads and FastAPI runs synchronous handlers in a threadpool, so
    a per-request connection is both the simplest and the only correct
    option here.
    """
    path = database_path(data_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because a connection opened by a
        # dependency (which FastAPI runs in a worker thread) may be used by
        # an `async def` handler on the event loop thread. It is still one
        # connection per request, used by one request at a time, so access
        # is serialised - which is the property SQLite actually requires.
        # Sharing a connection BETWEEN requests would not be safe and is
        # exactly what the per-request dependency prevents.
        conn = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False
        )
    except OSError as exc:
        raise StorageError(f"could not open {path}: {exc}") from exc

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    if migrate_on_open:
        migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every migration newer than the database's recorded version.

    ``PRAGMA user_version`` is the version store rather than a table,
    because it is written atomically with the rest of the transaction and
    cannot itself need a migration.
    """
    current = schema_version(conn)
    for version, script in MIGRATIONS:
        if version <= current:
            continue
        # BEGIN and COMMIT go INSIDE the script rather than around it.
        # ``executescript`` issues an implicit COMMIT before it runs, so a
        # transaction opened in Python would be closed out from under it and
        # every statement would land in autocommit - a migration that failed
        # halfway would leave half a schema behind. SQLite's DDL is
        # transactional, so a BEGIN in the script itself is atomic.
        #
        # The version is an int from this module's own tuple, never a
        # caller's value; PRAGMA takes a literal and cannot be parameterised.
        try:
            conn.executescript(
                "BEGIN;\n"
                + script
                + f"\nPRAGMA user_version = {int(version)};\nCOMMIT;"
            )
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        current = version
    return current


@contextmanager
def transaction(conn: sqlite3.Connection):
    """A write transaction that rolls back on any exception.

    ``BEGIN IMMEDIATE`` rather than deferred: every caller of this helper
    writes, and taking the write lock up front turns a concurrent-write
    conflict into a clean wait instead of a mid-transaction failure.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
