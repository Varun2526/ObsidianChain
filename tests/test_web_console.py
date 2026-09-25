"""The built console served from the API's origin (production deployment)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api import app as app_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>ObsidianChain</title>")
    (web / "assets" / "app-abc.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside the build")
    monkeypatch.setenv(app_module.WEB_DIST_ENV, str(web))
    return TestClient(app_module.create_app(tmp_path / "data"))


def test_deep_links_get_the_console(client) -> None:
    for path in ("/", "/graph?alert=x:1", "/inv/inv_1/review", "/entity/1abc"):
        r = client.get(path)
        assert r.status_code == 200 and "ObsidianChain" in r.text, path
        assert r.headers["cache-control"] == "no-cache"


def test_assets_are_served_and_cached(client) -> None:
    r = client.get("/assets/app-abc.js")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]


def test_api_paths_never_fall_back_to_the_console(client) -> None:
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
    assert client.get("/api/alerts").status_code == 401  # still session-guarded


def test_no_path_escapes_the_build_directory(client) -> None:
    r = client.get("/../secret.txt")
    assert "outside the build" not in r.text
    r = client.get("/assets/../../secret.txt")
    assert "outside the build" not in r.text


def test_security_headers_on_every_response(client) -> None:
    for path in ("/", "/api/health", "/api/alerts"):
        h = client.get(path).headers
        assert h["x-frame-options"] == "DENY"
        assert h["x-content-type-options"] == "nosniff"
        assert "default-src 'self'" in h["content-security-policy"]


def test_without_a_build_nothing_is_served(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(app_module.WEB_DIST_ENV, raising=False)
    c = TestClient(app_module.create_app(tmp_path))
    assert c.get("/graph").status_code == 404
