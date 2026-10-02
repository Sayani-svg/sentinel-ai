"""Business logic for log ingestion.

The service is transport agnostic. It receives a validated
:class:`~app.schemas.upload.LogUploadRequest` plus an iterable of byte chunks, so
it can be driven by a multipart upload, a fixture, or a generator in tests
without any of them appearing in the layer above. It raises the typed errors
below and never :class:`fastapi.HTTPException`; the router owns the mapping onto
status codes.

Files are stored under a generated name rather than the one the client supplied.
The original name is kept in the ``filename`` column for display, while
``file_path`` points at ``<uuid4>.<ext>``, which makes path traversal
structurally impossible on disk instead of merely filtered.

Ingestion is recorded as a lifecycle: the row is committed as
:attr:`~app.schemas.upload.UploadStatus.PROCESSING` before any bytes are read,
and finalised as ``COMPLETED`` or ``FAILED`` afterwards. A rejected upload
therefore leaves an operator-visible trace of a malformed file being submitted,
which is worth one row for an audit tool; the stored bytes are always removed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logger import get_logger
from app.models.uploaded_log import UploadedLog
from app.schemas.upload import (
    LogColumnInfo,
    LogFileFormat,
    LogPreview,
    LogUploadRequest,
    UploadStatus,
)
from app.services.log_parser import ParsedLog, parse_log_file

logger = get_logger(__name__)

#: Filenames are recorded for display only, but a bounded length keeps the
#: column sane and mirrors the limit applied to user-supplied names elsewhere.
MAX_FILENAME_LENGTH: int = 255

#: Bytes copied per iteration while streaming an upload to disk. Streaming
#: rather than buffering matters because ``MAX_UPLOAD_SIZE_MB`` defaults to 100.
STORAGE_CHUNK_BYTES: int = 1024 * 1024

#: Characters that would let a filename escape the upload directory or truncate
#: a path on the way to the filesystem.
_UNSAFE_FILENAME_CHARS: frozenset[str] = frozenset({"/", "\\", "\x00"})


class UploadServiceError(Exception):
    """Base class for log ingestion failures."""


class UnsafeFilenameError(UploadServiceError):
    """Raised when a submitted filename cannot be used safely."""


class UnsupportedFileTypeError(UploadServiceError):
    """Raised when no configured reader handles the submitted file type."""


class EmptyFileError(UploadServiceError):
    """Raised when an upload carries no bytes."""


class FileTooLargeError(UploadServiceError):
    """Raised when an upload exceeds the configured size limit."""


class LogStorageError(UploadServiceError):
    """Raised when the uploaded bytes cannot be persisted to disk."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of a successful ingestion.

    Attributes:
        log: The persisted ``uploaded_logs`` row, marked completed.
        parsed: Structural summary and retained records of the stored file.
    """

    log: UploadedLog
    parsed: ParsedLog


def sanitize_filename(raw_filename: str) -> str:
    """Validate a client-supplied filename and return its display form.

    Path separators, null bytes and control characters are rejected rather than
    stripped: a name that carries them is not a display name, and quietly
    rewriting it would hide a traversal attempt from whoever reads the logs.

    Args:
        raw_filename: Filename exactly as supplied in the multipart part.

    Returns:
        str: The trimmed filename, safe to store in the ``filename`` column.

    Raises:
        UnsafeFilenameError: If the name is empty, too long, contains a path
            separator or control character, or names a directory.
    """
    filename = raw_filename.strip()

    if not filename:
        raise UnsafeFilenameError("The uploaded part has no file name.")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise UnsafeFilenameError(
            f"The file name must not exceed {MAX_FILENAME_LENGTH} characters."
        )

    if filename in {".", ".."}:
        raise UnsafeFilenameError("The file name does not name a file.")

    if any(character in filename for character in _UNSAFE_FILENAME_CHARS):
        raise UnsafeFilenameError(
            "The file name must not contain path separators."
        )

    if any(ord(character) < 32 or ord(character) == 127 for character in filename):
        raise UnsafeFilenameError(
            "The file name must not contain control characters."
        )

    return filename


def resolve_log_format(filename: str, settings: Settings) -> LogFileFormat:
    """Determine which reader to use for an uploaded file.

    The extension is the only authority. The declared media type is ignored
    because browsers routinely send ``application/octet-stream`` for ``.csv``
    and omit it entirely, so trusting it would reject valid uploads.

    Args:
        filename: Sanitised client-supplied filename.
        settings: Application settings naming the permitted extensions.

    Returns:
        LogFileFormat: The reader to parse the file with.

    Raises:
        UnsupportedFileTypeError: If the extension is not permitted, or is
            permitted by configuration but has no implemented reader.
    """
    suffix = Path(filename).suffix.lower()
    permitted = {extension.lower() for extension in settings.SUPPORTED_UPLOAD_EXTENSIONS}

    if suffix not in permitted:
        supported = ", ".join(sorted(permitted)) or "none"
        raise UnsupportedFileTypeError(
            f"Files of type {suffix or 'unknown'} are not accepted. "
            f"Supported types: {supported}."
        )

    try:
        return LogFileFormat(suffix.removeprefix("."))
    except ValueError as exc:
        raise UnsupportedFileTypeError(
            f"Files of type {suffix} are permitted by configuration but no reader "
            "is implemented for them."
        ) from exc


def allocate_destination(upload_dir: Path, log_format: LogFileFormat) -> Path:
    """Reserve the on-disk location for an upload.

    The directory is created on demand because ``UPLOAD_DIR`` is a configured
    path that a fresh checkout does not have.

    Args:
        upload_dir: Directory holding ingested files.
        log_format: Format of the incoming file, used as the suffix.

    Returns:
        Path: An unused absolute path inside ``upload_dir``.

    Raises:
        LogStorageError: If the upload directory cannot be created.
    """
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LogStorageError(
            f"The upload directory {upload_dir} is not writable."
        ) from exc

    return upload_dir / f"{uuid4().hex}.{log_format.value}"


def write_upload(destination: Path, source: Iterable[bytes], max_bytes: int) -> int:
    """Stream an upload to disk, enforcing the size limit as it goes.

    Bytes are written to a staging file and moved into place only once the whole
    upload has arrived, so a connection that drops mid-transfer cannot leave a
    half-written file that later parses as a shorter, valid log.

    Args:
        destination: Final path for the stored file.
        source: Iterable of byte chunks, as read from the transport.
        max_bytes: Maximum number of bytes to accept.

    Returns:
        int: The number of bytes written.

    Raises:
        FileTooLargeError: If more than ``max_bytes`` bytes arrive.
        EmptyFileError: If the upload carries no bytes.
        LogStorageError: If the bytes cannot be written.
    """
    staging = destination.with_name(f".partial-{uuid4().hex}{destination.suffix}")
    written = 0

    try:
        with staging.open("wb") as handle:
            for chunk in source:
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(
                        f"The upload exceeds the maximum accepted size of "
                        f"{max_bytes} bytes."
                    )
                handle.write(chunk)

        if written == 0:
            raise EmptyFileError("The uploaded file is empty.")

        staging.replace(destination)
    except UploadServiceError:
        staging.unlink(missing_ok=True)
        raise
    except OSError as exc:
        staging.unlink(missing_ok=True)
        raise LogStorageError(f"The uploaded file could not be stored: {exc}") from exc

    return written


def build_preview(parsed: ParsedLog) -> LogPreview:
    """Render a parsed log as the API preview payload.

    Args:
        parsed: A parsed log file.

    Returns:
        LogPreview: Column summaries, counts and a small sample of records.
    """
    return LogPreview(
        file_format=parsed.file_format,
        columns=[
            LogColumnInfo(
                name=column.name,
                value_type=column.value_type,
                null_count=column.null_count,
            )
            for column in parsed.columns
        ],
        row_count=parsed.row_count,
        column_count=len(parsed.columns),
        truncated=parsed.truncated,
        sample_rows=[dict(record) for record in parsed.sample_rows()],
    )


def _create_pending_record(
    db: Session, *, filename: str, file_path: Path, uploaded_by: int
) -> UploadedLog:
    """Insert the ingestion record before any bytes are read.

    Committing first means a failure later still leaves evidence that the upload
    was attempted, and reserves the primary key while the transfer runs.

    Args:
        db: Active database session.
        filename: Sanitised client-supplied filename.
        file_path: Reserved on-disk location.
        uploaded_by: Id of the account performing the upload.

    Returns:
        UploadedLog: The persisted row, in the processing state.
    """
    uploaded_log = UploadedLog(
        filename=filename,
        file_path=str(file_path),
        upload_status=UploadStatus.PROCESSING.value,
        uploaded_by=uploaded_by,
        created_at=datetime.now(timezone.utc),
    )
    db.add(uploaded_log)
    db.commit()
    db.refresh(uploaded_log)
    return uploaded_log


def _mark_failed(db: Session, uploaded_log: UploadedLog) -> None:
    """Record an ingestion as failed without masking the original error.

    Args:
        db: Active database session.
        uploaded_log: Row to transition to the failed state.
    """
    try:
        uploaded_log.upload_status = UploadStatus.FAILED.value
        db.commit()
    except Exception as exc:
        # Deliberately broad: this runs while another exception is propagating,
        # and raising a second one would replace the real reason the upload was
        # rejected. The original failure is what the caller must see.
        db.rollback()
        logger.error(
            "Could not record the failed ingestion of log %d: %s",
            uploaded_log.id,
            exc,
        )


def ingest_log_upload(
    db: Session,
    request: LogUploadRequest,
    source: Iterable[bytes],
    *,
    uploaded_by: int,
    settings: Settings,
) -> IngestionResult:
    """Validate, store and parse a submitted log file.

    Args:
        db: Active database session.
        request: Transport-observable description of the submitted part.
        source: Iterable of byte chunks carrying the file contents.
        uploaded_by: Id of the account performing the upload.
        settings: Application settings supplying the upload directory, size
            limit and permitted extensions.

    Returns:
        IngestionResult: The completed row and the parsed representation of the
        stored file.

    Raises:
        UnsafeFilenameError: If the filename cannot be used safely.
        UnsupportedFileTypeError: If no reader handles the file type.
        EmptyFileError: If the upload carries no bytes.
        FileTooLargeError: If the upload exceeds the size limit.
        LogStorageError: If the file cannot be written to disk.
        LogParsingError: If the stored file does not match its declared layout.
    """
    filename = sanitize_filename(request.filename)
    log_format = resolve_log_format(filename, settings)
    max_bytes = settings.max_upload_size_bytes

    declared_size = request.declared_size_bytes
    if declared_size is not None and declared_size > max_bytes:
        raise FileTooLargeError(
            f"The declared upload size of {declared_size} bytes exceeds the "
            f"maximum accepted size of {max_bytes} bytes."
        )

    destination = allocate_destination(settings.UPLOAD_DIR, log_format)
    uploaded_log = _create_pending_record(
        db, filename=filename, file_path=destination, uploaded_by=uploaded_by
    )

    try:
        written = write_upload(destination, source, max_bytes)
        parsed = parse_log_file(destination, log_format)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        _mark_failed(db, uploaded_log)
        logger.warning(
            "Rejected upload %d from account %d (%s): %s",
            uploaded_log.id,
            uploaded_by,
            filename,
            exc,
        )
        raise

    uploaded_log.upload_status = UploadStatus.COMPLETED.value
    db.commit()
    db.refresh(uploaded_log)

    logger.info(
        "Ingested %s as log %d for account %d: %d byte(s), %d row(s), %d column(s)%s.",
        filename,
        uploaded_log.id,
        uploaded_by,
        written,
        parsed.row_count,
        len(parsed.columns),
        " (truncated)" if parsed.truncated else "",
    )

    return IngestionResult(log=uploaded_log, parsed=parsed)
