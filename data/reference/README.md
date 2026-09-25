# Offline reference data

## `dbip-country-lite-2026-09.csv.gz`

DB-IP "IP to Country Lite", September 2026 edition, as published by DB-IP.
Rows are `ip_start,ip_end,country_iso` for IPv4 and IPv6; `ZZ` means the
range is not assigned to a country.

- Source: db-ip.com, "IP to Country Lite" (free, monthly)
- Downloaded: 2026-09-25
- SHA-256: `a32bb3c384bd3de60ad9024596aa5b395a6dd5beaa27a7223407cc2edc681d0b`
- Rows: 717,170
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Attribution is required** wherever a country resolved from this file is
shown: "IP geolocation by DB-IP (db-ip.com), CC BY 4.0". The console shows
it next to every resolved country (run network panel, graph inspector).

`geoip.OfflineCSVProvider` picks up the newest `dbip-country-lite-*.csv*`
here when no `geoip_country.csv` is present. To update, place the next
month's file alongside and remove this one; runs record the version and
SHA-256 they used.

A resolved country is where the relaying peer's IP is registered. It is not
the sender's location.
