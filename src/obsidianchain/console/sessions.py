"""Server-side sessions.

What a session is here
----------------------
A row in SQLite. The browser holds an opaque random token in an HttpOnly
cookie and nothing else - no role, no user id, no expiry it can edit. Every
authenticated request re-reads the row, so revoking a session takes effect
on the next request rather than whenever a token happens to expire.

Only the token's SHA-256 is stored. Reading the database therefore yields no
usable session, which matters because the database file sits beside the
artifacts on a workstation whose whole threat model is "someone else has the
disk".

Why the cookie is not marked Secure
-----------------------------------
This is an offline application served over plain HTTP on loopback. Setting
``Secure`` would make the browser refuse to send the cookie and break login
entirely, so it is off, deliberately, and recorded here rather than left to
be rediscovered. ``HttpOnly`` and ``SameSite=Lax`` are both on: the first
keeps the token out of reach of any script on the page, the second stops a
cross-site request from carrying it.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass

from obsidianchain.console import db, errors, users
from obsidianchain.console.users import User

#: The cookie name. Distinct from the retired ``obsidianchain_session``
#: localStorage key so that a stale browser cannot present the old shape.
COOKIE_NAME = "obsidianchain_sid"

#: Absolute lifetime. Not sliding: a session that renews itself whenever it
#: is used never ends, which is the opposite of what an unattended forensic
#: workstation needs.
TTL_SECONDS = 12 * 60 * 60

#: Token entropy. 32 bytes is far beyond guessing and keeps the cookie short.
TOKEN_BYTES = 32


@dataclass(frozen=True)
class Session:
    token_sha256: str
    user_id: str
    created_at: str
    expires_at: str
    last_seen_at: str


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _expiry_from(now_iso: str, seconds: int) -> str:
    import datetime as dt

    started = dt.datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    ends = started + dt.timedelta(seconds=seconds)
    return ends.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create(
    conn: sqlite3.Connection, user: User, *, ttl_seconds: int = TTL_SECONDS
) -> str:
    """Open a session and return the RAW token, which is never stored."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO sessions (token_sha256, user_id, created_at,"
            " expires_at, last_seen_at) VALUES (?, ?, ?, ?, ?)",
            (_digest(token), user.id, now,
             _expiry_from(now, ttl_seconds), now),
        )
    return token


def resolve(conn: sqlite3.Connection, token: str | None) -> User:
    """Return the authenticated user, or raise.

    Three distinct refusals, because they mean different things to the
    person at the keyboard: no cookie at all (never logged in), a cookie
    whose row is gone or revoked (logged out here or elsewhere), and a row
    past its expiry (session simply ended).
    """
    if not token:
        raise errors.AuthenticationRequired("no session cookie was presented")

    row = conn.execute(
        "SELECT * FROM sessions WHERE token_sha256 = ?", (_digest(token),)
    ).fetchone()
    if row is None:
        raise errors.AuthenticationRequired("the session is not recognised")
    if row["revoked_at"] is not None:
        raise errors.SessionExpired("the session was revoked")

    now = db.utcnow()
    if row["expires_at"] <= now:
        raise errors.SessionExpired("the session has expired")

    user = users.get(conn, row["user_id"])
    if user is None or not user.active:
        # The account was deactivated while the session was open. The
        # session dies with it rather than outliving the account.
        raise errors.SessionExpired("the account is no longer active")

    conn.execute(
        "UPDATE sessions SET last_seen_at = ? WHERE token_sha256 = ?",
        (now, row["token_sha256"]),
    )
    return user


def revoke(conn: sqlite3.Connection, token: str | None) -> None:
    """Invalidate one session. Idempotent - logging out twice is not an error."""
    if not token:
        return
    with db.transaction(conn):
        conn.execute(
            "UPDATE sessions SET revoked_at = ? "
            "WHERE token_sha256 = ? AND revoked_at IS NULL",
            (db.utcnow(), _digest(token)),
        )


def revoke_all_for_user(conn: sqlite3.Connection, user_id: str) -> int:
    """Used when an account is deactivated or its password changes."""
    with db.transaction(conn):
        cursor = conn.execute(
            "UPDATE sessions SET revoked_at = ? "
            "WHERE user_id = ? AND revoked_at IS NULL",
            (db.utcnow(), user_id),
        )
    return cursor.rowcount


def purge_expired(conn: sqlite3.Connection) -> int:
    """Delete rows no longer capable of authenticating anything.

    Housekeeping only. Revocation is what makes a session stop working; this
    just keeps the table from growing without bound.
    """
    with db.transaction(conn):
        cursor = conn.execute(
            "DELETE FROM sessions WHERE expires_at <= ?", (db.utcnow(),)
        )
    return cursor.rowcount
