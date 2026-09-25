# Demo freeze, 2026-09-25

The code recorded in the final demo. After this point only fixes for actual
blockers go in, each as its own commit on top of tag `demo-freeze-2026-09-25`.

## What is frozen

| Item | Value |
|---|---|
| Code | commit `901ed4b` (branch `feat/ps-native-v3-production`); this file and the tag sit on top of it |
| Container image | `obsidianchain:demo`, linux/arm64, `sha256:98959b9f4441…48e3`, built offline from vendored wheels |
| Deployment | `make deploy PLATFORM=linux/arm64 TAG=demo`, console and API on one origin, port 8080 |
| Serving model | `ps_native_v5` (LightGBM, schema `ps_native_features/5`, 31 features, no network features); fallback `ps_native_v5_fallback_no_g` |
| GeoIP | DB-IP "IP to Country Lite" 2026-09, sha256 `a32bb3c3…1d0b`, CC BY 4.0 |
| Reference alert run | `043ea584e99daf99` (2,128 alerts) |

## Verification at freeze

- Backend: 2009 passed, 4 skipped (`pytest tests/`).
- Frontend: 114 passed (`vitest`); production build clean.
- **Clean end-to-end acceptance on the deployed instance** (fresh casework
  database, real HTTP, three roles): 54 checks passed, 0 failed. Covered:
  - Investigator: create a case; upload and validate a dataset; run the
    analysis in the background (QUEUED to COMPLETE).
  - Run results: 28 alerts scored by `ps_native_v5`.
  - Run graph with IP and ASN nodes.
  - Network propagation: 11 transactions, 9 peers, 7 ASNs, 2 countries
    resolved by DB-IP with the attribution; network evidence marked unfused.
  - Alert, address, transaction and trace views; referencing an alert into
    the case; the disposition, a note and a report; submitting for review.
  - Reviewer: take the case and approve it. Denied: overwriting the
    disposition, and case creation.
  - Author denied finalising their own report; the reviewer finalises it;
    the investigator closes the case.
  - Audit trail, export bundle and integrity verification.
  - Admin: users, audit log and model registry. The investigator is denied
    user administration.
- **UI on the deployed instance**: 14 screens across the three roles at
  1440x900 and 390x844 had no console errors. Checked by screenshot: the
  network panel shows resolved countries, the capture's countries as
  unverified, address class, and the DB-IP attribution.
- Blockers found: none. No code changed as a result of acceptance.

## Not changed by this freeze

The presentation file, the holdout (ADR 0004), and the production model.
