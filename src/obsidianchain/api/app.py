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

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from obsidianchain import __version__
from obsidianchain.alerts import contract as alerts_contract
from obsidianchain.api import (
    alerts as alerts_api,
    ingest as ingest_api,
    artifacts,
    boundary,
    demo,
    evidence,
    provenance_gate,
)

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

    @app.exception_handler(evidence.EvidenceIdInvalidError)
    async def _bad_id(request: Request, exc: evidence.EvidenceIdInvalidError):
        return JSONResponse(
            status_code=400,
            content={"error": "evidence_id_invalid", "detail": str(exc)},
        )

    @app.exception_handler(evidence.EvidenceIdStaleError)
    async def _stale_id(request: Request, exc: evidence.EvidenceIdStaleError):
        # 409, never 404. A stale id and an unknown row are different
        # failures: one says the inputs changed under you, the other says
        # that row never existed. Collapsing them would hide a re-point.
        return JSONResponse(
            status_code=409,
            content={
                "error": "evidence_id_stale",
                "detail": str(exc),
                "hint": "A determining input changed; mint a new id against "
                        "the current run.",
            },
        )

    @app.exception_handler(evidence.EvidenceNotFoundError)
    async def _no_row(request: Request, exc: evidence.EvidenceNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": "evidence_not_found", "detail": str(exc)},
        )

    @app.exception_handler(evidence.EvidenceJoinError)
    async def _join(request: Request, exc: evidence.EvidenceJoinError):
        return JSONResponse(
            status_code=500,
            content={"error": "evidence_join_failed", "detail": str(exc)},
        )

    @app.exception_handler(alerts_contract.AlertIdInvalidError)
    async def _bad_alert(request: Request, exc):
        return JSONResponse(
            status_code=400,
            content={"error": "alert_id_invalid", "detail": str(exc)},
        )

    @app.exception_handler(alerts_contract.AlertIdStaleError)
    async def _stale_alert(request: Request, exc):
        # 409, never 404. A stale id and an unknown cluster are different
        # facts: one means the artifact moved under the caller, the other
        # means the cluster was never there.
        return JSONResponse(
            status_code=409,
            content={"error": "alert_id_stale", "detail": str(exc)},
        )

    @app.exception_handler(alerts_contract.AlertNotFoundError)
    async def _no_alert(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={"error": "alert_not_found", "detail": str(exc)},
        )

    @app.exception_handler(alerts_api.AlertFilterError)
    async def _bad_filter(request: Request, exc):
        return JSONResponse(
            status_code=400,
            content={"error": "alert_filter_invalid", "detail": str(exc)},
        )

    @app.exception_handler(ingest_api.UploadTooLargeError)
    async def _too_large(request: Request, exc):
        return JSONResponse(
            status_code=413,
            content={"error": "upload_too_large", "detail": str(exc)},
        )

    @app.exception_handler(ingest_api.UploadEmptyError)
    async def _empty_upload(request: Request, exc):
        return JSONResponse(
            status_code=400,
            content={"error": "upload_empty", "detail": str(exc)},
        )

    @app.exception_handler(provenance_gate.ProvenanceRefusedError)
    async def _refused(
        request: Request, exc: provenance_gate.ProvenanceRefusedError
    ):
        return JSONResponse(
            status_code=500,
            content={"error": "provenance_refused", "detail": str(exc)},
        )

    @app.get(
        f"{API_PREFIX}/evidence/{{evidence_id}}",
        summary="Network evidence recorded for one proposed merge",
        response_description=(
            "The persisted funnel row, its provenance, the probe statistics "
            "with the configuration they were computed under, the persisted "
            "production evaluation (verdict, reason, reason code) in a "
            "separate labelled block, and two availability booleans."
        ),
    )
    async def get_evidence(evidence_id: str) -> JSONResponse:
        """Return the network evidence recorded for one proposed merge.

        SYNTHETIC network data. The response describes what was recorded; it
        does not claim the evidence proves the merge, and it does not claim
        the evidence caused the production engine to allow or block it - the
        persisted funnel is the chain-only trajectory and no fused decision
        ledger exists.

        The verdict and reason are read from schema /2 columns exactly as
        ``separation_evidence()`` returned them at that proposed merge. They
        are not reconstructed here, and a /1 artifact is refused rather than
        served without them.

        Reads two precomputed artifacts and joins them. Nothing is clustered,
        pooled, tested or evaluated.
        """
        return JSONResponse(
            content=evidence.get_evidence(evidence_id, app.state.data_root)
        )

    @app.post(
        f"{API_PREFIX}/ingest",
        summary="Validate an uploaded bulk metadata file (CSV, JSON or XML)",
        response_description=(
            "What the file contained, what validated, what could be "
            "correlated, and an explicit statement that the upload was NOT "
            "scored - scoring is an offline pipeline run."
        ),
    )
    async def ingest_upload(
        request: Request,
        filename: str = Query(default="upload"),
        format: str | None = Query(default=None),
    ) -> JSONResponse:
        """Parse, normalise and validate. Nothing is scored here."""
        from obsidianchain.io import ingest as ingest_io

        body = await request.body()
        try:
            payload = ingest_api.ingest_bytes(body, filename, format)
        except ingest_io.IngestError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "ingest_failed", "detail": str(exc)},
            )
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/alerts",
        summary="Ranked investigation alerts",
        response_description=(
            "Alerts ranked by aggregated member risk, with the filters "
            "applied and the provenance of the run that produced them. An "
            "alert is a co-spend CLUSTER, not a person or an account."
        ),
    )
    async def list_alerts(
        severity: list[str] | None = Query(default=None),
        min_risk: float | None = Query(default=None, ge=0.0, le=1.0),
        max_risk: float | None = Query(default=None, ge=0.0, le=1.0),
        first_timestep: int | None = Query(default=None, ge=1, le=49),
        last_timestep: int | None = Query(default=None, ge=1, le=49),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> JSONResponse:
        """Slice one precomputed parquet file. Nothing is scored here."""
        payload = alerts_api.list_alerts(
            data_root, severity=severity, min_risk=min_risk, max_risk=max_risk,
            first_t=first_timestep, last_t=last_timestep,
            limit=limit, offset=offset,
        )
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/alerts/{{alert_id}}",
        summary="One investigation alert, opened",
        response_description=(
            "Calibrated risk and severity, the model's own SHAP explanation, "
            "the M0-M3 evidence behind it, supported address relationships, "
            "an activity timeline, network evidence as investigative context "
            "only, and full provenance."
        ),
    )
    async def get_alert(alert_id: str) -> JSONResponse:
        """Join five precomputed tables by alert_id. Nothing is computed.

        The network block never asserts that an IP owns or sent from a
        wallet, and where a quantity was not computable the response says
        INSUFFICIENT_EVIDENCE rather than returning a zero.
        """
        payload = alerts_api.get_alert(alert_id, data_root)
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

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
