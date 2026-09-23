"""The frontend must fetch nothing from the network.

Why this test exists
--------------------
The backend has been air-gapped since Phase 0: the image builds with
``--network none`` and ``tests/test_offline.py`` asserts there is no route
out. The frontend had no such guard, and a dashboard rebuild quietly added

    <link href="https://fonts.googleapis.com/css2?family=Inter…">

to ``index.html``. On a genuinely air-gapped workstation that renders in
fallback faces; on a connected one it sends a request to a third party on
every page load. Neither is acceptable for an offline forensic tool, and
nothing in the suite noticed for a whole phase.

This scans the frontend SOURCE rather than a build output, so it fails in
development rather than at packaging time, and it needs no ``npm install``
to run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

#: Files that end up in the served page, one way or another.
SCANNED_SUFFIXES = (".html", ".css", ".ts", ".tsx", ".js", ".jsx")

#: Directories that are inputs to the build, not outputs of it.
SKIPPED_DIRS = {"node_modules", "dist", "build", ".vite", "coverage"}

#: Any absolute URL, however it is written. Deliberately broad: the point is
#: that the application reaches nothing, not that one CDN is blocked.
ABSOLUTE_URL = re.compile(r"""(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}""")

#: Hosts a well-meaning change is most likely to reintroduce. Named so the
#: failure message can say what happened rather than only that it happened.
KNOWN_REMOTE_HOSTS = (
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdn.jsdelivr.net",
    "cdnjs.cloudflare.com",
    "unpkg.com",
    "googleapis.com",
)

#: Permitted because they are never fetched. A documentation link in a code
#: comment costs nothing at runtime; a stylesheet or a font does.
ALLOWED_SUBSTRINGS = (
    "//localhost",
    "//127.0.0.1",
    "//www.w3.org/2000/svg",       # the SVG namespace, an identifier not a URL
    "//www.w3.org/1999/xhtml",
    "//reactjs.org",               # appears only inside React's own messages
)


def scanned_files() -> list[Path]:
    if not FRONTEND.is_dir():
        return []
    found = []
    for path in FRONTEND.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if SKIPPED_DIRS & set(path.relative_to(FRONTEND).parts):
            continue
        found.append(path)
    return sorted(found)


def offending_urls(text: str) -> list[str]:
    hits = []
    for match in ABSOLUTE_URL.finditer(text):
        url = match.group(0)
        if any(allowed in url for allowed in ALLOWED_SUBSTRINGS):
            continue
        hits.append(url)
    return hits


def test_there_is_a_frontend_to_scan() -> None:
    """Guard against a vacuous suite if the directory is ever moved."""
    assert scanned_files(), f"no frontend sources found under {FRONTEND}"


@pytest.mark.parametrize(
    "path", scanned_files(), ids=lambda p: str(p.relative_to(FRONTEND))
)
def test_no_source_file_references_an_external_url(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Comments are NOT stripped. In HTML and CSS a URL in a comment is one
    # uncomment away from being live, and unlike the Python source checks
    # there is no import graph here to catch it afterwards.
    offenders = sorted(set(offending_urls(text)))
    assert not offenders, (
        f"{path.relative_to(FRONTEND)} references {offenders}. This "
        f"application ships air-gapped; a remote stylesheet, font or script "
        f"renders as a fallback offline and leaks a request when online. "
        f"Vendor the asset or use a system face."
    )


def test_the_html_entry_point_loads_no_remote_stylesheet() -> None:
    """The specific regression, checked by name so the message is obvious."""
    index = FRONTEND / "index.html"
    assert index.is_file()
    text = index.read_text(encoding="utf-8")
    for host in KNOWN_REMOTE_HOSTS:
        assert host not in text, (
            f"index.html loads from {host}. That was the Phase 8 regression: "
            f"the backend builds with --network none while the page fetched "
            f"fonts from a third party on every load."
        )


def test_the_font_stacks_resolve_locally() -> None:
    """Fonts are bundled or system faces, never fetched.

    The stylesheet moved to src/design/ (tokens.css + app.css) in the
    redesign. @import is allowed there only for files that ship in the
    build: a relative path, or an @fontsource package that is a pinned
    dependency and is bundled by Vite. Every family keeps a generic fallback.
    """
    import json

    design = FRONTEND / "src" / "design"
    tokens = (design / "tokens.css").read_text(encoding="utf-8")
    app = (design / "app.css").read_text(encoding="utf-8")
    deps = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["dependencies"]
    for text in (tokens, app):
        for target in re.findall(r'@import\s+(?:url\()?["\']([^"\']+)', text):
            assert target.startswith("./") or target.startswith("@fontsource/"), (
                f"stylesheet imports {target!r}, which is neither local nor a bundled font package"
            )
            if target.startswith("@fontsource/"):
                package = "/".join(target.split("/")[:2])
                assert package in deps, f"{package} is imported but not a pinned dependency"
                assert not deps[package].startswith(("^", "~")), f"{package} is not pinned exactly"
    for variable in ("--oc-font", "--oc-mono"):
        match = re.search(rf"{variable}:\s*([^;]+);", tokens, re.S)
        assert match, f"{variable} is not defined"
        stack = match.group(1)
        assert any(generic in stack for generic in ("sans-serif", "monospace", "serif", "system-ui")), (
            f"{variable} has no generic fallback: {stack.strip()}"
        )


# ---- browser storage is not a security boundary -------------------------
#
# Asserted here rather than in the vitest suite because jsdom is configured
# without a functional Storage: a runtime assertion on `localStorage` would
# pass whatever the code did, which is worse than no test at all. A source
# scan cannot be fooled that way.

#: Keys the retired frontend used to hold identity and case state in the
#: browser. Their reappearance would mean the application had gone back to
#: trusting something the user can edit in devtools.
RETIRED_STORAGE_KEYS = (
    "obsidianchain_session",
    "obsidianchain_investigations",
    "obsidianchain_inv_ctr",
)

#: Files that must never touch browser storage at all. These hold identity,
#: authorisation and case state - the three things a user must not be able to
#: assert for themselves by editing a value in devtools.
SECURITY_SENSITIVE = (
    "src/store/auth.tsx",
    "src/store/investigation.tsx",
    "src/api/console.ts",
)

#: Storage a page may legitimately use, and for what. Nothing here survives
#: a reload into an authorisation decision: a remembered tab or page size is
#: a convenience, and the backend re-checks every protected operation anyway.
PERMITTED_STORAGE_USES = ("sidebar", "theme", "page size", "collapsed")


def code_only(source: str) -> str:
    """Executable JavaScript/TypeScript, with comments removed.

    The same rule ``tests/test_truth_isolation.py`` applies to Python: a
    module may legitimately DISCUSS a forbidden mechanism in prose - and
    these three modules do, at length, because explaining why identity no
    longer lives in ``localStorage`` is the point of their docstrings. Only
    an executable reference counts.

    String and template literals are tracked so a ``//`` inside one is not
    mistaken for the start of a comment, which would delete real code and
    let a genuine use slip past.
    """
    out: list[str] = []
    i, n = 0, len(source)
    quote: str | None = None
    while i < n:
        char = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if quote:
            out.append(char)
            if char == "\\":
                if i + 1 < n:
                    out.append(nxt)
                i += 2
                continue
            if char == quote:
                quote = None
            i += 1
            continue
        if char in "\"'`":
            quote = char
            out.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                i += 1
            continue
        if char == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (source[i] == "*" and source[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(char)
        i += 1
    return "".join(out)


def test_the_comment_stripper_does_not_eat_code() -> None:
    """A stripper that removed real lines would make the scan below vacuous."""
    assert "localStorage" in code_only('const k = localStorage; // note')
    assert "localStorage" not in code_only("// mentions localStorage")
    assert "localStorage" not in code_only("/* mentions localStorage */")
    # A protocol-relative URL in a string is not a comment.
    assert "https://x" in code_only('const u = "https://x";')


@pytest.mark.parametrize("relative", SECURITY_SENSITIVE)
def test_identity_and_case_state_never_touch_browser_storage(relative) -> None:
    path = FRONTEND / relative
    assert path.is_file(), f"{relative} is missing; update this list"
    text = code_only(path.read_text(encoding="utf-8"))
    for api in ("localStorage", "sessionStorage", "indexedDB", "document.cookie"):
        assert api not in text, (
            f"{relative} uses {api}. Identity, authorisation and case state "
            f"come from the server: the session is an HttpOnly cookie this "
            f"code cannot read, and anything the browser can write is "
            f"something a user can forge."
        )


def test_no_frontend_file_reuses_a_retired_storage_key() -> None:
    offenders = []
    for path in scanned_files():
        text = code_only(path.read_text(encoding="utf-8", errors="replace"))
        for key in RETIRED_STORAGE_KEYS:
            # The names may still appear in a comment explaining what was
            # retired and why; only a string literal is a use.
            if f'"{key}"' in text or f"'{key}'" in text:
                offenders.append(f"{path.relative_to(FRONTEND)}:{key}")
    assert not offenders, (
        f"{offenders} reintroduce browser-held identity or case state. These "
        f"keys held the session and the investigation list before the "
        f"application layer existed; localStorage is not a security boundary."
    )
