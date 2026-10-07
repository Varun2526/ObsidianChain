"""One-click demo sign-in for evaluation deployments.

Off unless ``OBSIDIANCHAIN_DEMO_LOGIN=1`` is set. When on, the login screen
offers the three built-in roles and signs the visitor in as that role's
demo account without a password, so evaluators can open the product and
test every workflow directly.

What stays true in demo mode:
- Every action is still attributed to a named account and audited; the
  sign-in itself is recorded as a LOGIN with ``method: DEMO_LOGIN``.
- Role-based access control is unchanged: a demo Reviewer can do exactly
  what any Reviewer can do, no more.
- An administrator can still deactivate a demo account, and a deactivated
  account cannot be entered this way.

It is not for a deployment holding real casework: anyone who can reach the
server can act as any of the three roles.
"""

from __future__ import annotations

import os
import secrets
import sqlite3

from obsidianchain.console import errors, users
from obsidianchain.console.rbac import Role

ENV_FLAG = "OBSIDIANCHAIN_DEMO_LOGIN"

#: The built-in demo accounts, one per role, in the order the screen shows them.
DEMO_ACCOUNTS = {
    Role.INVESTIGATOR: ("investigator", "Lead Investigator",
                        "Create a case, upload a capture, analyse it, decide on alerts, write the report."),
    Role.REVIEWER: ("reviewer", "Quality Reviewer",
                    "Review submitted cases, approve or return them, sign off reports."),
    Role.ADMIN: ("admin", "System Administrator",
                 "Accounts, datasets, the audit log and system state."),
}


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def roles() -> list[dict]:
    return [{"role": role.value, "username": username, "display_name": name, "description": text}
            for role, (username, name, text) in DEMO_ACCOUNTS.items()]


def account_for(conn: sqlite3.Connection, role: Role) -> users.User:
    """The demo account for ``role``, created if it does not exist yet.

    A created account gets a random password nobody knows, so it can only be
    entered through demo sign-in (or after an administrator resets it).
    """
    username, display_name, _ = DEMO_ACCOUNTS[role]
    user = users.by_username(conn, username)
    if user is None:
        user = users.create(conn, username=username, password=secrets.token_urlsafe(24),
                            role=role, display_name=display_name)
    if not user.active:
        raise errors.InvalidCredentials("this demo account has been deactivated by an administrator")
    if user.role is not role:
        raise errors.InvalidCredentials(
            f"the account {username!r} no longer has the {role.value} role")
    return user
