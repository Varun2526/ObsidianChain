"""CLI surface tests."""

from __future__ import annotations

from typer.testing import CliRunner

from obsidianchain import __version__
from obsidianchain.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "data root" in result.stdout
    assert "network" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout or "Commands" in result.stdout
