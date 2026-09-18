"""Roles, capabilities, and case access. One policy, in one place.

Three roles, and why not four
-----------------------------
ADMIN, INVESTIGATOR, REVIEWER. An ANALYST role was considered and rejected:
no capability below separates an analyst from an investigator, so the role
would have an identical row in the table. A role that grants exactly what
another role grants is decoration, and decoration in an authorisation layer
is worse than nothing because it reads as a control.

No permission table, and why
----------------------------
Three roles do not justify a database-backed policy engine. The mapping
below is a literal, which means a change to it is a reviewed diff rather
than a row someone edited at runtime, and ``tests/test_console_rbac.py``
can assert the whole policy in a few lines. If a deployment ever needs
per-installation variation, this is the one module that has to change.

What REVIEWER may and may not do
--------------------------------
A reviewer exists so that forensic output is not signed only by its author.
They can read every case, read all evidence, read every disposition and
note, add notes of their own, and finalise a report. They cannot create
cases, upload data, set dispositions, or assign work - a reviewer who could
rewrite the investigator's decisions would not be reviewing them.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    ADMIN = "ADMIN"
    INVESTIGATOR = "INVESTIGATOR"
    REVIEWER = "REVIEWER"


class Capability(str, Enum):
    """One verb the application can perform on behalf of a user."""

    # System
    MANAGE_USERS = "manage_users"
    VIEW_ALL_INVESTIGATIONS = "view_all_investigations"
    VIEW_ALL_AUDIT = "view_all_audit"

    # Case lifecycle
    CREATE_INVESTIGATION = "create_investigation"
    EDIT_INVESTIGATION = "edit_investigation"
    CHANGE_INVESTIGATION_STATUS = "change_investigation_status"

    # Data
    UPLOAD_DATASET = "upload_dataset"
    BIND_ANALYTICAL_RUN = "bind_analytical_run"

    # Casework
    REFERENCE_ALERT = "reference_alert"
    ASSIGN_ALERT = "assign_alert"
    SET_DISPOSITION = "set_disposition"
    WRITE_NOTE = "write_note"

    # Reporting
    CREATE_REPORT = "create_report"
    FINALISE_REPORT = "finalise_report"


_INVESTIGATOR = frozenset({
    Capability.CREATE_INVESTIGATION,
    Capability.EDIT_INVESTIGATION,
    Capability.CHANGE_INVESTIGATION_STATUS,
    Capability.UPLOAD_DATASET,
    Capability.BIND_ANALYTICAL_RUN,
    Capability.REFERENCE_ALERT,
    Capability.ASSIGN_ALERT,
    Capability.SET_DISPOSITION,
    Capability.WRITE_NOTE,
    Capability.CREATE_REPORT,
})

#: A reviewer writes notes and finalises reports. Everything else they do is
#: a read, which is governed by :func:`may_read_case` rather than by a
#: capability.
_REVIEWER = frozenset({
    Capability.VIEW_ALL_INVESTIGATIONS,
    Capability.VIEW_ALL_AUDIT,
    Capability.WRITE_NOTE,
    Capability.FINALISE_REPORT,
})

_ADMIN = frozenset(Capability)

CAPABILITIES: dict[Role, frozenset[Capability]] = {
    Role.ADMIN: _ADMIN,
    Role.INVESTIGATOR: _INVESTIGATOR,
    Role.REVIEWER: _REVIEWER,
}


def parse_role(raw) -> Role:
    """Coerce a stored string to a Role, refusing anything unrecognised.

    An unknown role must not degrade to a default. "Role we do not
    recognise" and "role with no privileges" are different states, and only
    one of them should ever be silent.
    """
    try:
        return Role(str(raw).upper())
    except ValueError as exc:
        raise ValueError(f"{raw!r} is not a known role") from exc


def has(role: Role, capability: Capability) -> bool:
    return capability in CAPABILITIES.get(role, frozenset())


def may_read_case(role: Role, user_id: str, owner_id: str) -> bool:
    """Whether this user may see this case at all.

    ADMIN and REVIEWER see every case - oversight that stops at a case
    boundary is not oversight. An INVESTIGATOR sees the cases they own.
    """
    if has(role, Capability.VIEW_ALL_INVESTIGATIONS):
        return True
    return user_id == owner_id


def may_write_case(role: Role, user_id: str, owner_id: str) -> bool:
    """Whether this user may change case-owned state.

    Note that this is necessary and not sufficient: a caller must ALSO hold
    the specific capability for the operation. A REVIEWER fails here and so
    cannot set a disposition even on a case they can read; they reach notes
    and report finalisation through the explicit exemptions in
    :func:`may_write_note` and the FINALISE_REPORT capability.
    """
    if role is Role.ADMIN:
        return True
    return user_id == owner_id


def may_write_note(role: Role, user_id: str, owner_id: str) -> bool:
    """Notes are the one thing a reviewer may add to someone else's case.

    Their review comments have to live beside the work they describe, and a
    note is additive: it records what the reviewer thought without changing
    what the investigator decided.
    """
    if not has(role, Capability.WRITE_NOTE):
        return False
    if role in (Role.ADMIN, Role.REVIEWER):
        return True
    return user_id == owner_id
