"""Password hashing. scrypt, from the standard library.

Why scrypt and not bcrypt or Argon2
-----------------------------------
This image builds with ``--network none`` from a vendored wheel set, and
neither ``bcrypt`` nor ``argon2-cffi`` is in it. Adding one would mean
re-vendoring a compiled dependency to solve a problem ``hashlib.scrypt``
already solves: it is memory-hard, it ships with CPython against OpenSSL,
and it needs nothing new.

The parameters below are the interactive-login profile from the scrypt
paper's own guidance (N=2^15, r=8, p=1, ~32 MB and ~100 ms per verify on a
current machine). That cost is irrelevant for one login and expensive for
an offline dictionary attack against a stolen database file, which is the
threat this defends against.

Encoding
--------
``scrypt$N$r$p$salt_hex$hash_hex``. The parameters travel with the hash so a
future increase in N can be rolled out without invalidating existing
passwords - :func:`verify` reads them from the stored string rather than
assuming today's constants.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

#: CPU/memory cost. Must be a power of two.
N = 2 ** 15
#: Block size.
R = 8
#: Parallelisation.
P = 1
#: Derived key length, bytes.
DKLEN = 32
#: Salt length, bytes.
SALT_BYTES = 16

SCHEME = "scrypt"

#: hashlib.scrypt enforces maxmem; the default is too small for N=2^15.
#: 128 * N * r * p is the algorithm's own requirement, plus headroom.
_MAXMEM = 128 * N * R * P * 2

#: Refused above this. Not a security limit - a denial-of-service one: the
#: cost of scrypt is paid by the server, and an unbounded password would let
#: an unauthenticated caller choose how much work to ask for.
MAX_PASSWORD_BYTES = 1024

#: The shortest password this application will set. Deliberately modest: a
#: length rule is the only policy here, because an offline single-workstation
#: deployment has no password-reset channel and locking an investigator out
#: of their own case file is a worse outcome than a short passphrase.
MIN_PASSWORD_LENGTH = 8


class PasswordError(ValueError):
    """The password or its stored encoding could not be used."""


def _derive(password: str, salt: bytes, *, n: int, r: int, p: int) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise PasswordError(
            f"password is {len(encoded)} bytes; the limit is "
            f"{MAX_PASSWORD_BYTES}"
        )
    return hashlib.scrypt(
        encoded, salt=salt, n=n, r=r, p=p, dklen=DKLEN, maxmem=_MAXMEM
    )


def hash_password(password: str) -> str:
    """Return the encoded hash for a new or changed password."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt, n=N, r=R, p=P)
    return f"{SCHEME}${N}${R}${P}${salt.hex()}${digest.hex()}"


def verify(password: str, encoded: str) -> bool:
    """Check a password against a stored hash.

    Returns False for a malformed stored value rather than raising. A
    corrupt row must fail the login, not 500 the endpoint - and it must fail
    it the same way a wrong password does, so the response cannot be used to
    probe which accounts have unusable hashes.
    """
    if not isinstance(encoded, str):
        return False
    parts = encoded.split("$")
    if len(parts) != 6 or parts[0] != SCHEME:
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = bytes.fromhex(parts[4])
        expected = bytes.fromhex(parts[5])
    except ValueError:
        return False
    if n < 2 or n & (n - 1) or r < 1 or p < 1 or not salt or not expected:
        return False
    try:
        candidate = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
            dklen=len(expected), maxmem=128 * n * r * p * 2,
        )
    except (ValueError, MemoryError, PasswordError):
        return False
    # Constant time: a byte-by-byte comparison would leak how much of a
    # guessed hash was correct.
    return hmac.compare_digest(candidate, expected)
