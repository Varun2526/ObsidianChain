"""User records. Creation, lookup, credential verification.

There is no self-registration. Accounts are created by an administrator
through the CLI (``obsidianchain console-user-add``) or by an authenticated
ADMIN through the API. An offline forensic workstation has no reason to let
an unauthenticated caller mint an identity, and the absence of that path is
the simplest way to guarantee it.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from dataclasses import dataclass

from obsidianchain.console import db, errors, passwords
from obsidianchain.console.rbac import Role, parse_role

#: Conservative on purpose: usernames appear in audit records and in report
#: headers, so they must round-trip through display and comparison without
#: normalisation surprises.
USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{1,63}$")


@dataclass(frozen=True)
class User:
    id: str
    username: str
    display_name: str
    role: Role
    active: bool
    created_at: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role.value,
            "active": self.active,
            "created_at": self.created_at,
        }


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        role=parse_role(row["role"]),
        active=bool(row["active"]),
        created_at=row["created_at"],
    )


def new_id() -> str:
    return "usr_" + secrets.token_hex(8)


def create(
    conn: sqlite3.Connection,
    *,
    username: str,
    password: str,
    role,
    display_name: str = "",
) -> User:
    """Create one account. Raises rather than overwriting an existing one."""
    username = (username or "").strip()
    if not USERNAME.match(username):
        raise errors.ValidationFailed(
            f"{username!r} is not a valid username: 2-64 characters, letters, "
            f"digits, dot, underscore or hyphen, starting with a letter or "
            f"digit"
        )
    try:
        resolved = parse_role(role)
    except ValueError as exc:
        raise errors.ValidationFailed(str(exc)) from exc
    try:
        password_hash = passwords.hash_password(password)
    except passwords.PasswordError as exc:
        raise errors.ValidationFailed(str(exc)) from exc

    user = User(
        id=new_id(),
        username=username,
        display_name=(display_name or username).strip(),
        role=resolved,
        active=True,
        created_at=db.utcnow(),
    )
    try:
        with db.transaction(conn):
            conn.execute(
                "INSERT INTO users (id, username, display_name, password_hash,"
                " role, active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                (user.id, user.username, user.display_name, password_hash,
                 user.role.value, user.created_at),
            )
    except sqlite3.IntegrityError as exc:
        raise errors.Conflict(f"username {username!r} already exists") from exc
    return user


def get(conn: sqlite3.Connection, user_id: str) -> User | None:
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return _row_to_user(row) if row else None


def by_username(conn: sqlite3.Connection, username: str) -> User | None:
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    return _row_to_user(row) if row else None


def listing(conn: sqlite3.Connection) -> list[User]:
    rows = conn.execute("SELECT * FROM users ORDER BY username").fetchall()
    return [_row_to_user(row) for row in rows]


def count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def authenticate(conn: sqlite3.Connection, username: str, password: str) -> User:
    """Verify credentials, or raise :class:`errors.InvalidCredentials`.

    Every failure mode - unknown user, wrong password, deactivated account -
    raises the same exception with the same message. Distinguishing them
    would turn the login endpoint into a username oracle.

    A dummy hash is verified when the user does not exist so that the
    response time does not depend on whether the account is real.
    """
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", ((username or "").strip(),)
    ).fetchone()

    if row is None:
        passwords.verify(password or "", _DUMMY_HASH)
        raise errors.InvalidCredentials("invalid username or password")
    if not passwords.verify(password or "", row["password_hash"]):
        raise errors.InvalidCredentials("invalid username or password")
    if not bool(row["active"]):
        raise errors.InvalidCredentials("invalid username or password")
    return _row_to_user(row)


def set_active(conn: sqlite3.Connection, user_id: str, active: bool) -> None:
    with db.transaction(conn):
        conn.execute(
            "UPDATE users SET active = ? WHERE id = ?",
            (1 if active else 0, user_id),
        )


def set_password(conn: sqlite3.Connection, user_id: str, password: str) -> None:
    try:
        encoded = passwords.hash_password(password)
    except passwords.PasswordError as exc:
        raise errors.ValidationFailed(str(exc)) from exc
    with db.transaction(conn):
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (encoded, user_id),
        )


def count_active_admins(conn: sqlite3.Connection) -> int:
    """Return the number of active accounts holding the ADMIN role."""
    row = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role = 'ADMIN' AND active = 1"
    ).fetchone()
    return int(row[0]) if row else 0


def set_role(conn: sqlite3.Connection, user_id: str, new_role) -> User:
    """Change an account's role. Refuses to demote the last active admin."""
    target = get(conn, user_id)
    if target is None:
        raise errors.NotFound(f"no user {user_id!r}")
    resolved = parse_role(new_role)
    if target.role is Role.ADMIN and resolved is not Role.ADMIN:
        if target.active and count_active_admins(conn) <= 1:
            raise errors.ValidationFailed(
                "cannot remove the ADMIN role from the only active administrator"
            )
    with db.transaction(conn):
        conn.execute(
            "UPDATE users SET role = ? WHERE id = ?",
            (resolved.value, user_id),
        )
    return get(conn, user_id)


def delete_user(conn: sqlite3.Connection, user_id: str) -> None:
    """Exceptional administrative deletion. Refuses to delete the last admin
    or a user owning investigations (which must be reassigned or deactivated)."""
    target = get(conn, user_id)
    if target is None:
        raise errors.NotFound(f"no user {user_id!r}")
    if target.role is Role.ADMIN and count_active_admins(conn) <= 1:
        raise errors.ValidationFailed("cannot delete the only active administrator")

    # Check if this user owns investigations
    inv_count = int(conn.execute(
        "SELECT COUNT(*) FROM investigations WHERE owner_id = ?", (user_id,)
    ).fetchone()[0])
    if inv_count > 0:
        raise errors.Conflict(
            f"user owns {inv_count} investigation(s). Investigations cannot be "
            f"orphaned; deactivate the user instead to preserve audit attribution."
        )

    # Check if this user has notes, reports, or dispositions
    notes_count = int(conn.execute(
        "SELECT COUNT(*) FROM investigator_notes WHERE author_id = ?", (user_id,)
    ).fetchone()[0])
    reports_count = int(conn.execute(
        "SELECT COUNT(*) FROM reports WHERE generated_by = ?", (user_id,)
    ).fetchone()[0])
    if notes_count > 0 or reports_count > 0:
        raise errors.Conflict(
            "user has authored case notes or reports. Deactivate the user "
            "to preserve historical audit attribution."
        )

    with db.transaction(conn):
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM saved_filters WHERE user_id = ?", (user_id,))
        try:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        except sqlite3.IntegrityError as exc:
            raise errors.Conflict(
                "user is referenced by existing audit or case records; "
                "deactivate the user instead to maintain data integrity."
            ) from exc


#: A real scrypt hash of a random value, computed once at import. Verifying
#: against it costs the same as verifying a genuine account, which is the
#: point - see :func:`authenticate`.
_DUMMY_HASH = passwords.hash_password(secrets.token_hex(32))

