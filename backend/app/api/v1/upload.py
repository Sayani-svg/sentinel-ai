"""Endpoints for ingesting user-uploaded log files.

``POST /upload/logs`` accepts a single ``multipart/form-data`` part named
``file``. The endpoint is synchronous because it streams the part to disk: a
synchronous handler is run by Starlette in a worker thread, so the blocking
reads do not stall the event loop, and ``UploadFile.file`` can be read directly
instead of being awaited.

Any authenticated account may upload. The role gate is deliberately permissive:
``uploaded_by`` records who submitted the file, and restricting ingestion to
analysts would leave the Viewer role with nothing to do. Authorisation on the
records themselves happens in the slices that read them.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.dependencies import get_db, require_viewer_or_above
from app.core.logger import get_logger
from app.models.user import User
from app.schemas.upload import LogUploadRequest, LogUploadResponse, UploadedLogRead
from app.services.log_parser import LogParsingError
from app.services.upload_service import (
    STORAGE_CHUNK_BYTES,
    EmptyFileError,
    FileTooLargeError,
    LogStorageError,
    UnsafeFilenameError,
    UnsupportedFileTypeError,
    build_preview,
    ingest_log_upload,
)

logger = get_logger(__name__)

router = APIRouter()

#: Status codes the endpoint can produce, documented on the route so the OpenAPI
#: page lists them. 4xx details describe the client's file; a 500 never does.
_ERROR_RESPONSES: dict[int | str, dict[str, str]] = {
    status.HTTP_400_BAD_REQUEST: {
        "description": "The file name is unusable or the file is empty."
    },
    status.HTTP_401_UNAUTHORIZED: {
        "description": "The bearer token is missing, invalid or expired."
    },
    status.HTTP_403_FORBIDDEN: {"description": "The account holds an unusable role."},
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {
        "description": "The file exceeds the configured size limit."
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "description": "The file extension is not an accepted log type."
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "description": "The file does not match the structure its extension implies."
    },
}


def _iter_upload_chunks(
    upload: UploadFile, chunk_size: int = STORAGE_CHUNK_BYTES
) -> Iterator[bytes]:
    """Yield the bytes of a multipart part in fixed-size chunks.

    Args:
        upload: The received multipart part.
        chunk_size: Number of bytes to read per iteration.

    Yields:
        bytes: Successive chunks, ending when the part is exhausted.
    """
    handle = upload.file
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            return
        yield chunk


@router.post(
    "/logs",
    response_model=LogUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a log file for analysis",
    responses=_ERROR_RESPONSES,
)
def upload_log_file(
    file: UploadFile = File(..., description="CSV or JSON log file to ingest."),
    current_user: User = Depends(require_viewer_or_above),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> LogUploadResponse:
    """Store a submitted log file, parse it, and record the ingestion.

    The file is written to the configured upload directory under a generated
    name, then parsed. Only the extension of the submitted filename is trusted to
    decide the reader; the declared media type is ignored, because browsers send
    ``application/octet-stream`` for CSV often enough that checking it would
    reject valid uploads.

    Args:
        file: The multipart part carrying the log file.
        current_user: Authenticated account performing the upload.
        db: Database session.
        settings: Application settings supplying the upload directory, size
            limit and permitted extensions.

    Returns:
        LogUploadResponse: The stored record and a preview of its parsed
        contents. The on-disk path is never disclosed.

    Raises:
        HTTPException: 400 for an unusable filename or an empty file, 413 when
            the upload is too large, 415 for an unsupported extension, 422 when
            the contents do not match the declared structure, and 500 when the
            bytes cannot be stored.
    """
    request = LogUploadRequest(
        filename=file.filename or "",
        content_type=file.content_type,
        declared_size_bytes=file.size,
    )

    try:
        result = ingest_log_upload(
            db,
            request,
            _iter_upload_chunks(file),
            uploaded_by=current_user.id,
            settings=settings,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except (UnsafeFilenameError, EmptyFileError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except LogParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LogStorageError as exc:
        logger.error("Could not store an uploaded log file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The uploaded file could not be stored.",
        ) from exc

    # Reused through UploadedLogRead so the two schemas cannot drift apart when
    # the stored record grows a column.
    stored = UploadedLogRead.model_validate(result.log)
    return LogUploadResponse(
        **stored.model_dump(), preview=build_preview(result.parsed)
    )
