"""Endpoints for generating and retrieving security reports.

``POST /reports`` documents a stored prediction: it restates the finding as a
report, writes it to the configured report directory, and records both the
``reports`` row and an ``audit_logs`` entry. ``GET /reports/{report_id}`` reads
that report back.

The two routes have deliberately different role gates. Generating a report is an
act of authorship -- it creates a durable, quotable artefact and writes to the
audit trail -- so it is reserved for
:data:`~app.core.dependencies.require_analyst_or_above`. Retrieving one is a read
of threat evidence, which every authenticated role may see, exactly as the upload
slice lets every role submit a file and defers authorisation to the slices that
read it.

Every retrieval is audited, including the CSV download. Reading a security
finding out of the system is the action a SOC most often needs to account for
afterwards, and an audit trail that only recorded the export's creation would not
answer "who has seen this".

Neither route discloses ``reports.report_path``. Clients reach the contents
through the retrieval endpoint, so the server's directory layout stays private;
this mirrors ``UploadedLogRead`` omitting ``file_path``.
"""

from __future__ import annotations

from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.dependencies import (
    get_db,
    require_analyst_or_above,
    require_viewer_or_above,
)
from app.core.logger import get_logger
from app.middleware.rate_limit import resolve_client_key
from app.models.report import Report
from app.models.user import User
from app.schemas.audit import AuditAction
from app.schemas.report import ReportDetail, ReportDocument, ReportRead, ReportRequest
from app.services.report_service import (
    AuditContext,
    MalformedPredictionError,
    MalformedReportError,
    ReportStorageError,
    ReportsDisabledError,
    ReportUnavailableError,
    UnknownPredictionError,
    UnknownReportError,
    create_report,
    read_report_export,
    record_audit_action,
    retrieve_report,
)

logger = get_logger(__name__)

router = APIRouter()

#: Status codes these endpoints can produce, documented on the routes so the
#: OpenAPI page lists them. Every 4xx below names a condition of the request or
#: of a referenced record; a 500 never does.
_ERROR_RESPONSES: dict[int | str, dict[str, str]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "description": "The bearer token is missing, invalid or expired."
    },
    status.HTTP_403_FORBIDDEN: {
        "description": "The account may not perform this action."
    },
    status.HTTP_404_NOT_FOUND: {
        "description": "No stored prediction or report matches the requested id."
    },
    status.HTTP_410_GONE: {
        "description": "The stored report file is missing or unreadable."
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "description": "The request body or path parameter is malformed."
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "description": "Report generation is disabled by configuration."
    },
}


@router.post(
    "",
    response_model=ReportDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a report for a stored prediction",
    responses=_ERROR_RESPONSES,
)
def generate_report(
    payload: ReportRequest,
    request: Request,
    current_user: User = Depends(require_analyst_or_above),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ReportDetail:
    """Document a stored prediction as a report and record the action.

    The prediction is read from the database rather than resubmitted, so a report
    can only ever describe a finding the server itself classified. The response
    carries the document as written, rather than re-reading it from disk, because
    a client that is told a report exists should not first have to prove the file
    round-trips.

    The ``reports`` row and the ``audit_logs`` entry are written by one commit
    inside :func:`~app.services.report_service.create_report`, so a generated
    report and the record of who generated it cannot come apart.

    Args:
        payload: Identifier of the prediction to document.
        request: Incoming request, used to resolve the client address for the
            audit trail.
        current_user: Authenticated Analyst or Admin generating the report.
        db: Database session.
        settings: Application settings, including the report directory and
            whether report generation and audit logging are enabled.

    Returns:
        ReportDetail: The stored report record and the document written to disk.

    Raises:
        HTTPException: 404 when no stored prediction has that id, 503 when report
            generation is disabled, and 500 when the prediction cannot be
            documented or the report or its record cannot be stored.
    """
    try:
        generated = create_report(
            db,
            prediction_id=payload.prediction_id,
            settings=settings,
            audit=AuditContext(
                user_id=current_user.id,
                action=AuditAction.REPORT_GENERATED,
                ip_address=resolve_client_key(request),
            ),
        )
    except ReportsDisabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except UnknownPredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction exists with id {exc.prediction_id}.",
        ) from exc
    except MalformedPredictionError as exc:
        # A stored finding this build cannot quote is a data-integrity problem,
        # not a bad request, so it is logged in full and reported as a 500.
        logger.error(
            "Prediction %d cannot be documented: predictions.%s holds %r -- %s",
            exc.prediction_id,
            exc.field,
            exc.value,
            exc.reason,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stored prediction could not be documented.",
        ) from exc
    except ReportStorageError as exc:
        logger.error(
            "Could not generate a report for prediction %d: %s",
            payload.prediction_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The report could not be stored.",
        ) from exc

    return _to_detail(generated.report, generated.document)


@router.get(
    "/{report_id}",
    response_model=None,
    summary="Retrieve a generated report",
    responses={
        **_ERROR_RESPONSES,
        status.HTTP_200_OK: {
            "description": (
                "The report as a validated JSON document, or the stored CSV export "
                "when ``format=csv`` is requested."
            ),
        },
    },
)
def read_report(
    request: Request,
    report_id: int = Path(..., ge=1, description="Identifier of the report to read."),
    report_format: Literal["json", "csv"] = Query(
        default="json",
        alias="format",
        description=(
            "``json`` returns the validated document; ``csv`` downloads the "
            "export stored beside it. The CSV is served from disk rather than "
            "re-rendered, so a download is always the file that was generated."
        ),
    ),
    current_user: User = Depends(require_viewer_or_above),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ReportDetail | Response:
    """Read a stored report back, as JSON or as the stored CSV export.

    ``response_model`` is deliberately ``None``: this route has two legitimate
    body types, and declaring a single response model would misdescribe whichever
    one it did not name. Both branches are documented in ``responses`` and the
    JSON branch is validated against
    :class:`~app.schemas.report.ReportDocument` inside the service before it is
    returned.

    Args:
        request: Incoming request, used to resolve the client address for the
            audit trail.
        report_id: Identifier of the report to read.
        report_format: ``json`` for the validated document, ``csv`` for the stored
            export. Exposed to clients as ``format``.
        current_user: Authenticated account reading the report.
        db: Database session.
        settings: Application settings, including whether audit logging is enabled.

    Returns:
        ReportDetail | Response: The report and its document, or the stored CSV
        export as a file download.

    Raises:
        HTTPException: 404 when no stored report has that id, 410 when the stored
            file is missing or unreadable, and 500 when the stored file is not a
            valid report.
    """
    try:
        report, document = retrieve_report(db, report_id)
        export_path = read_report_export(report) if report_format == "csv" else None
    except UnknownReportError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report exists with id {exc.report_id}.",
        ) from exc
    except ReportUnavailableError as exc:
        logger.warning(
            "Report %d recorded but its file could not be read: %s",
            exc.report_id,
            exc.report_path,
        )
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="The stored report file is no longer available.",
        ) from exc
    except MalformedReportError as exc:
        logger.error(
            "Stored report %d is not a valid report (%s): %s",
            exc.report_id,
            exc.reason,
            exc.report_path,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stored report could not be read.",
        ) from exc

    record_audit_action(
        db,
        user_id=current_user.id,
        action=(
            AuditAction.REPORT_DOWNLOADED
            if report_format == "csv"
            else AuditAction.REPORT_RETRIEVED
        ),
        ip_address=resolve_client_key(request),
        settings=settings,
    )

    if export_path is not None:
        return FileResponse(
            export_path,
            media_type="text/csv; charset=utf-8",
            filename=f"sentinel-report-{report.id}.csv",
        )

    return _to_detail(report, document)


def _to_detail(report: Report, document: ReportDocument) -> ReportDetail:
    """Render a report row and its document as the API representation.

    Both are passed through their own schemas, so the response is built from the
    validated document rather than from the in-memory object that produced it.

    Args:
        report: The persisted ``reports`` row.
        document: The validated report document.

    Returns:
        ReportDetail: The API representation.
    """
    stored = ReportRead.model_validate(report)
    return ReportDetail(**stored.model_dump(), document=document)
