"""Read-only HTTP presentation layer over precomputed artifacts.

This package serves what the pipeline already wrote. It does not cluster, does
not pool evidence, does not run a statistical test, does not evaluate, and
does not train anything. Every response is a file on disk, reshaped no further
than the transport requires.

That is a boundary, not a preference, and it is enforced three ways:

* :mod:`obsidianchain.api.boundary` lists the modules this package may not
  import, and ``tests/test_api_boundary.py`` checks the source against it;
* ``tests/test_truth_isolation.py`` enumerates every module in the package
  and requires it to be truth-free - this package is inference-by-default,
  covered from the moment it existed rather than after somebody remembered
  to add it to a list;
* loaders raise when an artifact is missing, naming the CLI command that
  builds it, instead of building it on demand.

The reason is not tidiness. An API that recomputes on request would produce
numbers that no manifest describes, under a rule nobody recorded, and the
Phase 4.1 provenance work exists precisely so that cannot happen.
"""

from __future__ import annotations

from obsidianchain.api.app import create_app

__all__ = ["create_app"]
