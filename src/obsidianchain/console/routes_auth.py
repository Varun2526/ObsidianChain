"""/api/auth/* and /api/users/* - identity, sessions, accounts.

The cookie is the whole client-side session. The browser never receives the
user's role as trusted state, never stores a token in ``localStorage``, and
cannot extend its own expiry. ``GET /api/auth/me`` is the only way the
frontend learns who it is talking as, and it re-reads the database every
time.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Body, Depends, Request, Response

from obsidianchain.console import audit, deps, errors, sessions, users
from obsidianchain.console.rbac import CAPABILITIES, Capability, Role, parse_role
from obsidianchain.console.users import User

router = APIRouter(prefix="/api", tags=["auth"])


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=sessions.COOKIE_NAME,
        value=token,
        max_age=sessions.TTL_SECONDS,
        httponly=True,
        samesite="lax",
        # Not `secure`: this application is served over plain HTTP on
        # loopback in an offline deployment, and a Secure cookie would never
        # be sent. Recorded in console/sessions.py rather than left to be
        # rediscovered.
        secure=False,
        path="/",
    )


def _identity(user: User) -> dict:
    """What the frontend may know about itself.

    Capabilities are included for RENDERING only - so the UI can hide a
    control the backend would refuse anyway. They are re-derived from the
    role on the server on every request and are never trusted on the way
    back in.
    """
    return {
        "user": user.as_dict(),
        "capabilities": sorted(
            c.value for c in CAPABILITIES.get(user.role, frozenset())
        ),
    }


@router.post("/auth/login", summary="Exchange credentials for a session")
def login(
    response: Response,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
) -> dict:
    """Verify a password and open a server-side session.

    Every failure - unknown user, wrong password, deactivated account -
    returns the same 401 with the same message. Distinguishing them would
    turn this endpoint into a username oracle.
    """
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    try:
        user = users.authenticate(conn, username, password)
    except errors.InvalidCredentials:
        audit.record_standalone(
            conn, actor_id=None, action=audit.LOGIN_FAILED,
            object_type="session", object_id=None,
            detail={"username": username[:64]},
        )
        raise

    token = sessions.create(conn, user)
    audit.record_standalone(
        conn, actor_id=user.id, action=audit.LOGIN,
        object_type="session", object_id=None,
        detail={"role": user.role.value},
    )
    _set_cookie(response, token)
    return _identity(user)


@router.get("/auth/me", summary="The authenticated identity, re-read")
def me(user: User = Depends(deps.current_user)) -> dict:
    return _identity(user)


@router.post("/auth/logout", summary="Revoke the current session")
def logout(
    request: Request,
    response: Response,
    conn: sqlite3.Connection = Depends(deps.get_connection),
) -> dict:
    """Revoke server-side, then clear the cookie.

    Order matters: the session is dead in the database before the response
    is written, so a cookie that survives - copied, cached, restored by a
    browser - authenticates nothing.
    """
    token = request.cookies.get(sessions.COOKIE_NAME)
    actor = None
    try:
        actor = sessions.resolve(conn, token)
    except errors.ConsoleError:
        actor = None

    sessions.revoke(conn, token)
    if actor is not None:
        audit.record_standalone(
            conn, actor_id=actor.id, action=audit.LOGOUT,
            object_type="session", object_id=None, detail={},
        )
    response.delete_cookie(sessions.COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/users", summary="Accounts (admin)")
def list_users(
    conn: sqlite3.Connection = Depends(deps.get_connection),
    _actor: User = Depends(deps.require_capability(Capability.MANAGE_USERS)),
) -> dict:
    return {"users": [u.as_dict() for u in users.listing(conn)]}


@router.get("/users/assignable", summary="Accounts an alert may be assigned to")
def assignable(
    conn: sqlite3.Connection = Depends(deps.get_connection),
    _actor: User = Depends(deps.current_user),
) -> dict:
    """Active accounts, name and role only.

    Available to any authenticated user because assigning work requires
    knowing who exists. It deliberately carries no credential material and no
    account state beyond what a picker needs.
    """
    return {
        "users": [
            {"id": u.id, "username": u.username,
             "display_name": u.display_name, "role": u.role.value}
            for u in users.listing(conn) if u.active
        ]
    }


@router.post("/users", summary="Create an account (admin)", status_code=201)
def create_user(
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.MANAGE_USERS)),
) -> dict:
    created = users.create(
        conn,
        username=str(payload.get("username") or ""),
        password=str(payload.get("password") or ""),
        role=payload.get("role") or "INVESTIGATOR",
        display_name=str(payload.get("display_name") or ""),
    )
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.USER_CREATED,
        object_type="user", object_id=created.id,
        detail={"username": created.username, "role": created.role.value},
    )
    return created.as_dict()


@router.post("/users/{user_id}/deactivate", summary="Deactivate an account")
def deactivate(
    user_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.MANAGE_USERS)),
) -> dict:
    """Deactivate and revoke. An account that cannot log in must also stop
    being logged in - otherwise deactivation takes effect whenever the open
    session happens to expire."""
    target = users.get(conn, user_id)
    if target is None:
        raise errors.NotFound(f"no user {user_id!r}")
    if target.id == actor.id:
        raise errors.ValidationFailed(
            "an administrator cannot deactivate their own account"
        )
    if target.role is Role.ADMIN and target.active and users.count_active_admins(conn) <= 1:
        raise errors.ValidationFailed(
            "cannot deactivate the only active administrator"
        )
    users.set_active(conn, user_id, False)
    revoked = sessions.revoke_all_for_user(conn, user_id)
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.USER_DEACTIVATED,
        object_type="user", object_id=user_id,
        detail={"sessions_revoked": revoked},
    )
    return {"ok": True, "sessions_revoked": revoked}


@router.post("/users/{user_id}/activate", summary="Reactivate an account")
def activate(
    user_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.MANAGE_USERS)),
) -> dict:
    target = users.get(conn, user_id)
    if target is None:
        raise errors.NotFound(f"no user {user_id!r}")
    users.set_active(conn, user_id, True)
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.USER_ACTIVATED,
        object_type="user", object_id=user_id, detail={},
    )
    return {"ok": True, "user": users.get(conn, user_id).as_dict()}


@router.post("/users/{user_id}/role", summary="Change an account's role (admin)")
def change_role(
    user_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.MANAGE_USERS)),
) -> dict:
    target = users.get(conn, user_id)
    if target is None:
        raise errors.NotFound(f"no user {user_id!r}")
    new_role = payload.get("role")
    if not new_role:
        raise errors.ValidationFailed("role is required")
    updated = users.set_role(conn, user_id, new_role)
    # Revoke sessions so updated permissions take effect on next request
    sessions.revoke_all_for_user(conn, user_id)
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.USER_ROLE_CHANGED,
        object_type="user", object_id=user_id,
        detail={"from": target.role.value, "to": updated.role.value},
    )
    return {"ok": True, "user": updated.as_dict()}


@router.post("/users/{user_id}/reset-password", summary="Reset an account's password (admin)")
def reset_password(
    user_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.MANAGE_USERS)),
) -> dict:
    target = users.get(conn, user_id)
    if target is None:
        raise errors.NotFound(f"no user {user_id!r}")
    password = str(payload.get("password") or "")
    if not password:
        raise errors.ValidationFailed("new password is required")
    users.set_password(conn, user_id, password)
    revoked = sessions.revoke_all_for_user(conn, user_id)
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.PASSWORD_RESET,
        object_type="user", object_id=user_id,
        detail={"sessions_revoked": revoked},
    )
    return {"ok": True, "sessions_revoked": revoked}


@router.post("/auth/change-password", summary="Change own password")
def change_password(
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    old_password = str(payload.get("old_password") or "")
    new_password = str(payload.get("new_password") or "")
    if not old_password or not new_password:
        raise errors.ValidationFailed("both old_password and new_password are required")
    try:
        users.authenticate(conn, actor.username, old_password)
    except errors.InvalidCredentials:
        raise errors.ValidationFailed("existing password is incorrect")
    users.set_password(conn, actor.id, new_password)
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.PASSWORD_CHANGED,
        object_type="user", object_id=actor.id, detail={},
    )
    return {"ok": True}


@router.delete("/users/{user_id}", summary="Delete an account (admin, exceptional)")
def delete_user_route(
    user_id: str,
    payload: dict = Body(default={}),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.MANAGE_USERS)),
) -> dict:
    if user_id == actor.id:
        raise errors.ValidationFailed("an administrator cannot delete their own account")
    target = users.get(conn, user_id)
    if target is None:
        raise errors.NotFound(f"no user {user_id!r}")
    if not payload.get("confirm"):
        raise errors.ValidationFailed(
            "exceptional account deletion requires 'confirm: true' in request body"
        )
    # Record audit event BEFORE deletion
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.USER_DELETED,
        object_type="user", object_id=user_id,
        detail={"username": target.username, "role": target.role.value},
    )
    users.delete_user(conn, user_id)
    return {"ok": True}


@router.get("/roles", summary="The role -> capability policy, as served")
def roles(_actor: User = Depends(deps.current_user)) -> dict:
    """The authorisation policy itself, readable.

    Published so that what the application enforces and what a reviewer
    believes it enforces cannot drift apart unnoticed. Reading it grants
    nothing.
    """
    return {
        "roles": {
            role.value: sorted(c.value for c in caps)
            for role, caps in CAPABILITIES.items()
        },
        "note": (
            "Capabilities are necessary, not sufficient. Case-scoped actions "
            "additionally require ownership (or ADMIN); see "
            "console/rbac.py:may_write_case."
        ),
    }


__all__ = ["router", "parse_role"]
