"""The FastAPI application. One route in Phase 5.1.

Deliberately small. There is no service layer, no repository, no dependency
container and no response model, because one read-only route over one JSON
file needs none of those and the abstractions would have to be guessed before
the other seven endpoints exist to shape them.

Failures are explicit and distinguish two different problems:

``503`` the artifact has not been generated yet. The message names the command
    that generates it. Not a client error - the request was fine, the pipeline
    has not run.
``500`` the artifact exists and is not what it claims to be: unreadable,
    unflagged, provenance weakened, or carrying evaluation-only truth. Serving
    it degraded would be worse than failing, so it fails.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from obsidianchain import __version__
from obsidianchain.api import artifacts, boundary, demo

API_PREFIX = "/api"

DESCRIPTION = (
    "Read-only presentation layer over precomputed ObsidianChain artifacts. "
    "Serves what the pipeline already wrote: no clustering, no evidence "
    "pooling, no statistical test and no evaluation happens on a request. "
    "All network data in this project is SYNTHETIC."
)


def create_app(data_root=None) -> FastAPI:
    """Build the application.

    ``data_root`` is a parameter rather than read from the environment at
    import time so a test can point the app at a temporary directory without
    mutating process state.
    """
    app = FastAPI(
        title="ObsidianChain API",
        version=__version__,
        description=DESCRIPTION,
    )
    app.state.data_root = data_root

    @app.exception_handler(artifacts.ArtifactMissingError)
    async def _missing(request: Request, exc: artifacts.ArtifactMissingError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "artifact_not_generated",
                "detail": str(exc),
                "hint": "The API reads precomputed artifacts and never "
                        "builds them on request.",
            },
        )

    @app.exception_handler(artifacts.ArtifactInvalidError)
    async def _invalid(request: Request, exc: artifacts.ArtifactInvalidError):
        return JSONResponse(
            status_code=500,
            content={"error": "artifact_invalid", "detail": str(exc)},
        )

    @app.exception_handler(demo.DemoProvenanceError)
    async def _provenance(request: Request, exc: demo.DemoProvenanceError):
        return JSONResponse(
            status_code=500,
            content={
                "error": "demo_provenance_invalid",
                "detail": str(exc),
                "hint": "A demonstration payload must be unmistakable. "
                        "Refusing to serve it rather than weakening it.",
            },
        )

    @app.exception_handler(boundary.TruthLeakInResponseError)
    async def _truth(request: Request, exc: boundary.TruthLeakInResponseError):
        return JSONResponse(
            status_code=500,
            content={"error": "truth_leak_blocked", "detail": str(exc)},
        )

    @app.get(
        f"{API_PREFIX}/demo/scenarios",
        summary="The five Phase 3.4 demonstration scenarios",
        response_description=(
            "The persisted scenarios.json payload, structure unchanged, "
            "DEMO-flagged at every level."
        ),
    )
    async def demo_scenarios() -> JSONResponse:
        """Return the persisted Phase 3.4 demonstration payload.

        SYNTHETIC. These are hand-built fixtures designed to reach each
        engine state, not a measurement. They are also **not** the Phase 3.3
        controlled worlds A-E, which are SYNTHETIC_CONTROL rather than DEMO.

        Nothing is generated, clustered, pooled or evaluated here. The file
        is read, its DEMO markers are verified, and it is returned as-is.
        """
        return JSONResponse(
            content=demo.get_demo_scenarios(app.state.data_root)
        )

    return app


#: Module-level app for ``uvicorn obsidianchain.api.app:app``.
app = create_app()
