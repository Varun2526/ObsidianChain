import os
from pathlib import Path

# Ensure OBSIDIANCHAIN_DATA defaults to local repo data directory during test runs
if not os.environ.get("OBSIDIANCHAIN_DATA"):
    repo_data = Path(__file__).resolve().parents[1] / "data"
    if repo_data.is_dir():
        os.environ["OBSIDIANCHAIN_DATA"] = str(repo_data)
