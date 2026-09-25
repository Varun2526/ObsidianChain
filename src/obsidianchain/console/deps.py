"""FastAPI wiring: the database connection, the session, the current run.

Authorisation happens HERE and in the domain modules, never in the browser.
A React route guard decides what to draw; these dependencies decide what a
caller is allowed to receive, and they run whether the request came from the
application, from ``curl``, or from a deep link someone was sent.
"""

from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import Depends, Request

from obsidianchain.console import db, errors, sessions
from obsidianchain.console.rbac import Capability, has
from obsidianchain.console.users import User


def data_root_of(request: Request):
    """The data root this application instance was built with.

    Falls back to the artifact layer's default so the console and the
    analytical routes always read from the same place. Resolving it through
    ``app.state`` rather than the environment keeps a test able to point one
    application at a temporary directory without mutating process state -
    the same reason ``create_app`` takes the parameter at all.
    """
    root = getattr(request.app.state, "data_root", None)
    if root is not None:
        return root
    from obsidianchain.api import artifacts

    return artifacts.DEFAULT_DATA_ROOT


def get_connection(request: Request) -> Iterator[sqlite3.Connection]:
    """One SQLite connection per request, closed when the request ends.

    Not a pool. SQLite connections are cheap to open and unsafe to share
    across threads, and FastAPI runs synchronous handlers in a threadpool, so
    per-request is both the simplest and the only correct choice.
    """
    conn = db.connect(data_root_of(request))
    try:
        yield conn
    finally:
        conn.close()


def optional_user(
    request: Request, conn: sqlite3.Connection = Depends(get_connection)
) -> User | None:
    """The authenticated user, or None. For routes that adapt rather than refuse."""
    try:
        return sessions.resolve(conn, request.cookies.get(sessions.COOKIE_NAME))
    except errors.ConsoleError:
        return None


def current_user(
    request: Request, conn: sqlite3.Connection = Depends(get_connection)
) -> User:
    """The authenticated user, or raise 401.

    Every case-scoped route depends on this. There is no path into case data
    that does not pass through it.
    """
    return sessions.resolve(conn, request.cookies.get(sessions.COOKIE_NAME))


def require_capability(capability: Capability):
    """Dependency factory for a route gated on one capability.

    Used where the check does not depend on a particular case - user
    management, system-wide audit. Case-scoped checks go through
    ``investigations.require_writable``, which also has to consider
    ownership.
    """

    def _dependency(user: User = Depends(current_user)) -> User:
        if not has(user.role, capability):
            raise errors.AccessDenied(
                f"role {user.role.value} may not {capability.value}"
            )
        return user

    return _dependency


def current_artifact_run(
    request: Request,
    investigation_id: str | None = None,
    conn: sqlite3.Connection = Depends(get_connection),
) -> str | None:
    """The run a case's alerts are current against, or None.

    On a case route (``investigation_id`` in the path) whose case is bound
    to one of its own completed uploaded-dataset runs, that run: it is
    immutable and belongs to the case (``console/run_alerts.py``).
    Otherwise the public fingerprint of the reference alert artifact on
    disk.

    Returns None when the artifact is missing or unreadable rather than
    raising, because the console must stay usable when the analytical
    pipeline has not been run: a case, its notes and its audit trail are
    still meaningful without alerts on disk.

    None then propagates as "staleness could not be checked", which is
    reported distinctly from "not stale". Those are different answers and
    collapsing them would let an unverifiable reference render as a
    verified one.
    """
    from obsidianchain.api import alerts as alerts_api, artifacts

    from obsidianchain.console import run_alerts

    try:
        _frame, sidecar = artifacts.load_alerts(data_root_of(request))
        reference = alerts_api.current_run_fingerprint(sidecar)
    except Exception:
        # Deliberately broad: every failure mode here - artifact absent,
        # sidecar refused, torn publish, unreadable parquet - means the same
        # thing to this caller, which is that the current run is unknown.
        # The analytical routes still report each of them precisely.
        reference = None
    return run_alerts.effective_run(conn, investigation_id, reference)
