"""Phase 7 - the investigation and alert layer.

Alert artifacts are generated offline by ``phase7-alerts`` and served
read-only by :mod:`obsidianchain.api.alerts`. No score is computed during a
request: the API's guarantee is that a response is a file the pipeline
already wrote.
"""

from __future__ import annotations
