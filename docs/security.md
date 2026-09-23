# Security model — ObsidianChain console and ML serving

Threat model: an offline, single-host investigation workstation. The adversary
has access to the disk or to the local network, or is a malicious or mistaken
insider. There is no internet-facing surface.

| Area | Control | Evidence |
|---|---|---|
| Authentication | scrypt password hashes; server-side sessions; HttpOnly + SameSite=Lax cookie; `Secure` via `OBSIDIANCHAIN_COOKIE_SECURE=1` behind TLS | `tests/test_console_auth.py` |
| Brute force | 5 failed logins per (username, client) in 15 min -> 429 for 15 min; every failure audited. In-process: resets on restart (accepted for single-host) | `test_repeated_failed_logins_are_throttled` |
| Authorisation | RBAC capabilities; case-scoped routes check readability; run results resolve run -> dataset -> investigation, so another case's run is 404 | `test_run_results_are_served_to_the_case_and_only_to_it`, `tests/test_api_access.py` |
| Uploads | 64 MiB cap; content-addressed storage; the user filename is metadata only, never a path | `console/datasets.py` |
| Input validation | capture contract quarantines conflicting or invalid transactions, and refuses a capture that is more than 50% invalid; feature contract blocks out-of-range values before scoring | `tests/test_registry_and_chaos.py` |
| Model artifacts | pickle (joblib) loaded only after its SHA-256 matches BOTH the registry (in git) and the manifest; registered versions immutable; lineage attested to a commit | `ml/registry.py`, CI `model-integrity` |
| Artifact replacement | a changed byte fails `verify`, and serving falls back with a HIGH alert; a changed registry is a git diff | `test_tampered_champion_is_refused_and_fallback_serves` |
| Supply chain | every dependency pinned (including transitive); offline image from vendored wheels; `pip-audit` in CI (no known vulnerabilities, 2026-09-23) | `requirements.txt`, `.github/workflows/ci.yml` |
| Secrets | none in the repository; the console DB, uploads and runs are gitignored | `.gitignore` |
| PII | the capture holds IP addresses. They stay on the host, appear in the evidence of the case that uploaded them, and are never sent elsewhere | architecture |
| Logging | the audit log records actor, action and object for casework and model runs; it records no passwords | `console/audit.py` |

## Residual risks

- joblib is pickle. Integrity checks prevent loading a changed file but not a
  malicious file that was registered deliberately. Registration is a
  privileged, reviewed action (git history).
- The login throttle is in-process and is lost on restart.
- Plain HTTP on loopback by default. Deploying to a network requires TLS and
  `OBSIDIANCHAIN_COOKIE_SECURE=1`.
- `pip-audit` does not block CI. A new advisory must be triaged by upgrading
  and re-vendoring, or by a written, dated ignore.
