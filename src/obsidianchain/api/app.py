"""The FastAPI application: two layers, mounted side by side.

The analytical layer (this package) serves IMMUTABLE ANALYTICAL TRUTH.
Deliberately small. There is no service layer, no repository, no dependency
container and no response model, because read-only routes over precomputed
files need none of those.

The application layer (``obsidianchain.console``) serves MUTABLE INVESTIGATOR
STATE - users, sessions, cases, datasets, dispositions, notes, reports, audit
- from SQLite. Its routers are mounted here so that one process serves one
origin, and its exceptions are translated by the same handler convention the
analytical routes already use.

The two never merge. The console REFERENCES analytical artifacts by their own
identifiers and copies none of them.

Failures are explicit and distinguish different problems:

``503`` the artifact has not been generated yet. The message names the command
    that generates it. Not a client error - the request was fine, the pipeline
    has not run.
``500`` the artifact exists and is not what it claims to be: unreadable,
    unflagged, provenance weakened, or carrying evaluation-only truth. Serving
    it degraded would be worse than failing, so it fails.
``401/403`` the console refused the caller rather than the request.
``409`` two things that look joinable describe different runs.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from obsidianchain import __version__
from obsidianchain.alerts import contract as alerts_contract
from obsidianchain.api import (
    alerts as alerts_api,
    ingest as ingest_api,
    artifacts,
    boundary,
    demo,
    evaluation,
    evidence,
    patterns,
    provenance_gate,
    separation,
)
from obsidianchain.console import deps as console_deps
from obsidianchain.console import errors as console_errors
from obsidianchain.console import (
    routes_auth,
    routes_casework,
    routes_investigations,
)

API_PREFIX = "/api"

#: Every route that may be reached WITHOUT a session, and why.
#:
#: Declared as data rather than left implicit so that
#: ``tests/test_api_access.py`` can walk the application's own route table
#: and prove, behaviourally, that nothing else answers an unauthenticated
#: caller. A new endpoint is therefore protected by default: forgetting the
#: dependency produces a failing test rather than an open route.
#:
#: The analytical routes are NOT here. They were public until now, which
#: meant the authenticated console could be bypassed entirely by querying
#: the artifacts directly - the alerts, the SHAP explanations and the
#: separation evidence were all readable with one unauthenticated GET.
PUBLIC_ROUTES: dict[tuple[str, str], str] = {
    ("POST", f"{API_PREFIX}/auth/login"): (
        "The endpoint that establishes a session. It cannot require one. It "
        "discloses nothing: every failure returns the same 401."
    ),
    ("POST", f"{API_PREFIX}/auth/logout"): (
        "Idempotent revocation of whatever token was presented. Requiring a "
        "valid session would make logging out of an ALREADY-expired session "
        "fail, which is the one moment a user most wants it to work. It "
        "reads nothing and returns nothing but {'ok': true}."
    ),
    ("GET", f"{API_PREFIX}/health"): (
        "Liveness only: the application version and whether the analytical "
        "artifacts are present. No alert, no case, no identity, no count."
    ),
}


def requires_session():
    """The dependency every non-public route carries.

    One mechanism, reused. This resolves the same server-side session row
    that the console's own routes resolve - there is no second credential,
    no API key, and no separate token for the analytical surface.

    Applied with ``dependencies=[...]`` rather than as a parameter because
    these handlers do not need to KNOW who is calling; they need only that
    somebody is. Authorisation for case-scoped data continues to happen
    where it always did, in the console's domain modules, against ownership
    and role. Authentication is not authorisation, and this dependency is
    only the first of the three gates.
    """
    return Depends(console_deps.current_user)


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

    @app.exception_handler(separation.SeparationBasisError)
    async def _basis(request: Request, exc: separation.SeparationBasisError):
        # 500, not 404: the artifacts are present and internally valid, and
        # the join between them cannot be established. Serving an empty list
        # would read as "no separation evidence exists for this cluster",
        # which is a different and false statement.
        return JSONResponse(
            status_code=500,
            content={"error": "separation_basis_mismatch", "detail": str(exc)},
        )

    @app.exception_handler(console_errors.ConsoleError)
    async def _console(request: Request, exc: console_errors.ConsoleError):
        """One translation for every application-layer failure.

        Each class carries its own status and stable code, so the frontend
        reads the same ``{"error": ..., "detail": ...}`` shape from the
        console that it already reads from the analytical routes.
        """
        return JSONResponse(
            status_code=exc.status,
            content={"error": exc.code, "detail": str(exc)},
        )

    @app.get(
        f"{API_PREFIX}/evidence/{{evidence_id}}",
        summary="Network evidence recorded for one proposed merge",
        dependencies=[requires_session()],
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
        dependencies=[requires_session()],
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
        dependencies=[requires_session()],
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
        dependencies=[requires_session()],
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
        f"{API_PREFIX}/alerts/{{alert_id}}/related",
        summary="Related alerts identified through shared transactions or observed network infrastructure (T2)",
        dependencies=[requires_session()],
    )
    async def get_related_alerts(
        alert_id: str,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> JSONResponse:
        payload = alerts_api.get_related_alerts(alert_id, data_root, limit=limit)
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/transactions/{{txid}}",
        summary="Transaction drilldown: inputs, outputs, counterparties, peers, and mixing classification (T2)",
        dependencies=[requires_session()],
    )
    async def get_transaction(txid: int) -> JSONResponse:
        payload = alerts_api.get_transaction_drilldown(txid, data_root)
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/demo/scenarios",
        summary="The five Phase 3.4 demonstration scenarios",
        dependencies=[requires_session()],
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

    @app.get(
        f"{API_PREFIX}/alerts/{{alert_id}}/separation-evidence",
        summary="Network-separation records for this cluster's merges",
        dependencies=[requires_session()],
        response_description=(
            "One row per proposed merge inside this cluster, each with the "
            "persisted production verdict and reason code, an evidence id "
            "resolvable through /api/evidence, and the frozen wordings that "
            "say what NOT_SEPARATED and NO_EVIDENCE do not mean."
        ),
    )
    async def alert_separation_evidence(
        alert_id: str,
        limit: int = Query(default=separation.ROWS_IN_DETAIL, ge=1, le=1000),
    ) -> JSONResponse:
        """Join three precomputed artifacts by address code. Nothing is computed.

        The join is verified rather than assumed: the two artifacts must
        declare identical chain inputs and heuristic, and this alert's own
        member addresses must resolve to its cluster in the index on disk.
        """
        payload = separation.get_separation_evidence(
            alert_id, app.state.data_root, limit=limit
        )
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/alerts/{{alert_id}}/patterns",
        summary="Peeling-chain and mixing-like structure behind this alert",
        dependencies=[requires_session()],
        response_description=(
            "The PEEL-1 chain structure the model already consumed, and the "
            "mixing-like classification of this alert's correlated "
            "transactions, each with the frozen wording saying what the "
            "structure does and does not establish."
        ),
    )
    async def alert_patterns(
        alert_id: str,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> JSONResponse:
        """Read three precomputed artifacts. Nothing is classified here.

        Both structures are OBSERVATIONS about transaction shape. Neither
        establishes laundering, ownership or an offence, and the response
        carries the wording that says so rather than leaving the UI to
        paraphrase it.
        """
        payload = patterns.get_patterns(alert_id, app.state.data_root,
                                        limit=limit)
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/alerts/{{alert_id}}/graph",
        summary="Investigation graph for this alert",
        dependencies=[requires_session()],
        response_description=(
            "Investigation graph projecting cluster, address, transaction, "
            "and announcing peer network nodes and edges."
        ),
    )
    async def alert_graph(
        alert_id: str,
        hops: int = Query(default=2, ge=1, le=5),
    ) -> JSONResponse:
        payload = alerts_api.get_alert_graph(alert_id, app.state.data_root, hops=hops)
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/evaluation/synthetic",
        summary="The controlled synthetic evaluation (SYNTHETIC_CONTROL)",
        dependencies=[requires_session()],
        response_description=(
            "The synthetic world's manifest, the overlap/degradation sweep "
            "and the chain/network/fused layer comparison, every one of them "
            "stamped SYNTHETIC_CONTROL."
        ),
    )
    async def synthetic_evaluation() -> JSONResponse:
        """Read two files the world commands wrote. Nothing is computed.

        Served from a path no case reads, and stamped SYNTHETIC_CONTROL
        throughout: these are controlled results on a generated world and
        must never be mistaken for the production analytical run.
        """
        payload = evaluation.get_synthetic_evaluation(app.state.data_root)
        boundary.assert_no_truth_fields(payload)
        return JSONResponse(content=payload)

    @app.get(
        f"{API_PREFIX}/health",
        summary="Liveness. The one analytical-side route that needs no session",
    )
    async def health() -> JSONResponse:
        """Whether the process is up and whether the artifacts are present.

        Deliberately thin. It reports no alert count, no run fingerprint and
        no case - a liveness probe that leaked the size of the dataset would
        be a disclosure channel wearing a health check's name.
        """
        try:
            artifacts.alerts_path(app.state.data_root)
            artifacts_present = True
        except artifacts.ArtifactMissingError:
            artifacts_present = False
        return JSONResponse(content={
            "status": "ok",
            "version": __version__,
            "artifacts_present": artifacts_present,
        })

    # The application layer. Mounted after the analytical routes so that the
    # read-only surface is defined first and unchanged by anything below it.
    app.include_router(routes_auth.router)
    app.include_router(routes_investigations.router)
    app.include_router(routes_casework.router)

    return app


#: Module-level app for ``uvicorn obsidianchain.api.app:app``.
app = create_app()
