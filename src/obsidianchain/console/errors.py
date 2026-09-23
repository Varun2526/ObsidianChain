"""Domain failures, each mapped to one HTTP status and one error code.

The analytical layer already distinguishes failures that look alike but mean
different things - ``409 alert_id_stale`` is not ``404 alert_not_found``,
because "the artifact moved under you" and "that cluster never existed" are
different facts. This layer follows the same rule: every class below exists
because collapsing it into another one would hide something an investigator
needs to know.

The disclosure decision, recorded
---------------------------------
:class:`InvestigationNotFound` is 404 and :class:`AccessDenied` is 403, so a
caller CAN distinguish "no such case" from "someone else's case". That is a
deliberate trade: it discloses the existence of a case id to anyone holding
one, and in exchange an investigator who mistypes a case number gets a
different answer from one who is genuinely not authorised - which is the
difference between a typo and a policy question.

Case ids are opaque 16-hex strings precisely so that this disclosure cannot
be turned into enumeration: knowing that ``inv_3f9a…`` exists requires
already having been given it.
"""

from __future__ import annotations


class ConsoleError(Exception):
    """Base for every application-layer failure."""

    #: HTTP status this failure is served as.
    status = 400
    #: Stable machine-readable code, the same convention the analytical
    #: routes use in their ``{"error": ...}`` bodies.
    code = "console_error"


class AuthenticationRequired(ConsoleError):
    """No valid session accompanied a request that needs one."""

    status = 401
    code = "authentication_required"


class InvalidCredentials(ConsoleError):
    """Username unknown, password wrong, or the account is deactivated.

    One class for all three on purpose: telling a caller which of the three
    it was turns the login form into a username oracle.
    """

    status = 401
    code = "invalid_credentials"


class TooManyAttempts(ConsoleError):
    """Login throttled after repeated failures for one username and client."""

    status = 429
    code = "too_many_attempts"


class SessionExpired(ConsoleError):
    """The session existed and is no longer valid.

    Distinct from :class:`AuthenticationRequired` so the UI can say "your
    session ended" rather than "please log in", which are different events
    to a person who never logged out.
    """

    status = 401
    code = "session_expired"


class AccessDenied(ConsoleError):
    """Authenticated, but this role or this user may not do this."""

    status = 403
    code = "access_denied"


class InvestigationNotFound(ConsoleError):
    status = 404
    code = "investigation_not_found"


class NotFound(ConsoleError):
    """A sub-resource of a case the caller may already see."""

    status = 404
    code = "not_found"


class ValidationFailed(ConsoleError):
    """The request body was understood and is not acceptable."""

    status = 422
    code = "validation_failed"


class Conflict(ConsoleError):
    """The request contradicts state that already exists.

    Used for the provenance conflicts in particular: referencing an alert
    from a run the case is not bound to, or re-finalising a report.
    """

    status = 409
    code = "conflict"


class RunMismatch(Conflict):
    """An alert from one analytical run met a case bound to another.

    409 rather than 400, and never a silent re-point. This is the
    case-level counterpart of ``alert_id_stale``: the analytical artifact
    and the case disagree about which run they describe, and serving either
    one as though it matched the other would attach an investigator's
    decision to a cluster they never looked at.
    """

    code = "run_mismatch"


class UploadRejected(ConsoleError):
    """The uploaded bytes could not be accepted."""

    status = 400
    code = "upload_rejected"


class AppendOnlyViolation(ConsoleError):
    """An attempt to mutate an append-only record."""

    status = 409
    code = "append_only"
