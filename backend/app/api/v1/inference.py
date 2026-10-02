"""Prediction endpoint: classify an ingested log file and store the outcome.

``POST /predict`` takes the identifier of a completed upload and runs the
detection pipeline over the file the upload slice already validated and stored.
The bytes are never resubmitted: the file on disk is the single record of what
was ingested, and accepting a second copy would let two predictions disagree
about the same upload.

Only Analyst and Admin accounts may run detection, per
:data:`~app.core.dependencies.require_analyst_or_above`, which is the guard
already documented for "accounts cleared to analyse logs and create
predictions". Reading results is not a separate route here; the dashboard slice
owns aggregated reads, and the created prediction is returned in full here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.dependencies import get_db, require_analyst_or_above
from app.core.logger import get_logger
from app.models.user import User
from app.schemas.prediction import (
    PredictionRead,
    PredictionRequest,
    PredictionRunResponse,
)
from app.services.log_parser import LogParsingError
from app.services.prediction_service import (
    DetectionOutcome,
    EmptyDetectionInputError,
    InvalidConfidenceError,
    InvalidSeverityError,
    StoredLogUnavailableError,
    UnknownUploadError,
    UploadNotReadyError,
    run_detection,
)
from app.services.upload_service import UnsupportedFileTypeError

logger = get_logger(__name__)

router = APIRouter()

#: Status codes the endpoint can produce, documented on the route so the OpenAPI
#: page lists them. Every 4xx below names a condition of the request or of the
#: referenced upload; a 500 never does.
_ERROR_RESPONSES: dict[int | str, dict[str, str]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "description": "The bearer token is missing, invalid or expired."
    },
    status.HTTP_403_FORBIDDEN: {"description": "The account may not run detection."},
    status.HTTP_404_NOT_FOUND: {"description": "No ingested log has that id."},
    status.HTTP_409_CONFLICT: {
        "description": "The upload exists but was rejected or is still processing."
    },
    status.HTTP_410_GONE: {
        "description": "The stored file is no longer on disk."
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "description": "The stored file's type is no longer an accepted log type."
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "description": "The stored file holds no classifiable records."
    },
}


@router.post(
    "",
    response_model=PredictionRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Classify an ingested log file",
    responses=_ERROR_RESPONSES,
)
def create_prediction_for_upload(
    payload: PredictionRequest,
    current_user: User = Depends(require_analyst_or_above),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> PredictionRunResponse:
    """Classify a completed upload and persist the prediction.

    Args:
        payload: Identifier of the upload to classify.
        current_user: Authenticated Analyst or Admin running the detection.
        db: Database session.
        settings: Application settings naming the permitted extensions.

    Returns:
        PredictionRunResponse: The stored prediction and how many records it was
        derived from.

    Raises:
        HTTPException: 404 when no ingested log has that id, 409 when the upload
            was rejected or is still processing, 410 when the stored file is
            gone, 415 when its type is no longer accepted, 422 when it holds no
            classifiable records, and 500 when the classification cannot be
            stored.
    """
    try:
        outcome = run_detection(db, upload_id=payload.upload_id, settings=settings)
    except UnknownUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingested log exists with id {exc.upload_id}.",
        ) from exc
    except UploadNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Upload {exc.upload_id} is '{exc.upload_status}' and cannot be "
                "classified."
            ),
        ) from exc
    except StoredLogUnavailableError as exc:
        logger.warning(
            "Upload %d completed but its file is missing from disk: %s",
            exc.upload_id,
            exc.file_path,
        )
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="The stored log file is no longer available.",
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except (EmptyDetectionInputError, LogParsingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except (InvalidSeverityError, InvalidConfidenceError) as exc:
        # The detector supplies both, so reaching here means the stored value set
        # and this build disagree. The traceback is logged and the client is told
        # nothing about the internal detail.
        logger.error(
            "Could not store a prediction for upload %d: %s", payload.upload_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The prediction could not be stored.",
        ) from exc

    return _to_response(outcome)


def _to_response(outcome: DetectionOutcome) -> PredictionRunResponse:
    """Render a detection outcome as the response body.

    The prediction is revalidated through :class:`PredictionRead` so the response
    is built from the persisted row rather than from the in-memory
    classification, and the two cannot disagree.

    Args:
        outcome: The stored prediction and the classification behind it.

    Returns:
        PredictionRunResponse: The API representation of the stored prediction.
    """
    stored = PredictionRead.model_validate(outcome.prediction)
    return PredictionRunResponse(
        **stored.model_dump(),
        records_analysed=outcome.records_analysed,
    )