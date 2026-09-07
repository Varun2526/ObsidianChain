"""Allow `python -m obsidianchain` in addition to the installed console script."""

from obsidianchain.cli import app

if __name__ == "__main__":
    app()
