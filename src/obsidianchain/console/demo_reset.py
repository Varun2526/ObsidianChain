"""Deterministic Clean Demo Database Setup.

Wipes mutable application casework and seeds only the primary operational
identities (Admin, Investigator, Reviewer) with dynamically generated
passwords (or environment variable overrides).

Leaves data/raw/ and data/processed/ analytical artifacts completely untouched.
"""

from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path

from obsidianchain.console import db, users
from obsidianchain.console.rbac import Role


def generate_secure_password() -> str:
    """Generate a high-entropy, human-typeable alphanumeric temporary password."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$"
    return "".join(secrets.choice(alphabet) for _ in range(16))


def reset_clean_demo_database(
    data_root: str | Path,
    *,
    confirm: bool = False,
    admin_password: str | None = None,
    investigator_password: str | None = None,
    reviewer_password: str | None = None,
) -> dict[str, dict[str, str]]:
    """Deterministically resets the database and uploads directory.

    Requires explicit confirmation to prevent accidental invocation in
    production.
    """
    if not confirm and os.environ.get("OBSIDIANCHAIN_DEMO_RESET") != "1":
        raise RuntimeError(
            "Demo database reset is destructive. You must provide confirm=True "
            "or set OBSIDIANCHAIN_DEMO_RESET=1."
        )

    root = Path(data_root)
    db_file = db.database_path(root)
    wal_file = root / f"{db.DB_FILENAME}-wal"
    shm_file = root / f"{db.DB_FILENAME}-shm"
    upload_dir = root / db.UPLOAD_DIRNAME

    # Remove database files
    for path in (db_file, wal_file, shm_file):
        if path.exists():
            path.unlink()

    # Clear uploaded datasets
    if upload_dir.exists():
        for item in upload_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Re-initialize clean database with all migrations
    conn = db.connect(root, migrate_on_open=True)

    # Resolve passwords
    pwd_admin = (
        admin_password
        or os.environ.get("OBSIDIANCHAIN_DEMO_ADMIN_PASSWORD")
        or generate_secure_password()
    )
    pwd_inv = (
        investigator_password
        or os.environ.get("OBSIDIANCHAIN_DEMO_INVESTIGATOR_PASSWORD")
        or generate_secure_password()
    )
    pwd_rev = (
        reviewer_password
        or os.environ.get("OBSIDIANCHAIN_DEMO_REVIEWER_PASSWORD")
        or generate_secure_password()
    )

    try:
        user_admin = users.create(
            conn,
            username="admin",
            password=pwd_admin,
            role=Role.ADMIN,
            display_name="System Administrator",
        )
        user_inv = users.create(
            conn,
            username="investigator",
            password=pwd_inv,
            role=Role.INVESTIGATOR,
            display_name="Lead Investigator",
        )
        user_rev = users.create(
            conn,
            username="reviewer",
            password=pwd_rev,
            role=Role.REVIEWER,
            display_name="Quality Reviewer",
        )
    finally:
        conn.close()

    return {
        "admin": {
            "id": user_admin.id,
            "username": user_admin.username,
            "role": user_admin.role.value,
            "password": pwd_admin,
        },
        "investigator": {
            "id": user_inv.id,
            "username": user_inv.username,
            "role": user_inv.role.value,
            "password": pwd_inv,
        },
        "reviewer": {
            "id": user_rev.id,
            "username": user_rev.username,
            "role": user_rev.role.value,
            "password": pwd_rev,
        },
    }
