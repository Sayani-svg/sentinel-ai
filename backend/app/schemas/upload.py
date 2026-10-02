"""Pydantic schemas for the log ingestion endpoints.

``upload_status`` on the ``uploaded_logs`` table is a plain text column, so the
permitted lifecycle values are declared here as a :class:`enum.StrEnum` and
validated by the service rather than by the database. The same pattern is used
for ``users.role`` in :mod:`app.schemas.user`.

``UploadedLogRead`` omits ``file_path``. It is a server-side absolute path, and
disclosing it would leak the deployment's directory layout to any authenticated
caller. This mirrors how :class:`~app.schemas.user.UserRead` omits
``password_hash``: the column exists, but no route can return it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class UploadStatus(StrEnum):
    """Lifecycle states of an ingested log file.

    The literal values are stored in the ``upload_status`` column of the
    ``uploaded_logs`` table, so they are never renamed or re-cased.
    """

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


#: Canonical upload status values, kept as plain strings so they can be compared
#: against values loaded straight out of the database.
VALID_UPLOAD_STATUS_VALUES: frozenset[str] = frozenset(
    status.value for status in UploadStatus
)


class LogFileFormat(StrEnum):
    """Structural format of an accepted log file.

    Derived from the file extension, which is the only thing the service trusts:
    content sniffing is not attempted, because a declared type that disagrees
    with the bytes is a stronger signal of a malformed upload than a mismatch
    found by guessing.
    """

    CSV = "csv"
    JSON = "json"


class LogValueType(StrEnum):
    """Inferred type of a parsed log column.

    ``MIXED`` means the column holds more than one unrelated type, which usually
    indicates an upstream join produced a ragged column. ``INTEGER`` and
    ``FLOAT`` combine into ``FLOAT`` rather than reporting as mixed, because a
    numeric column that happens to contain whole numbers is still numeric.
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    NULL = "null"
    MIXED = "mixed"


#: Scalar values a parsed log cell may hold. Parsing deliberately normalises to
#: this closed set so that the prediction slice can hand the records straight to
#: a dataframe without re-inspecting every cell.
LogValue: TypeAlias = str | int | float | bool | None

#: One parsed log line. This is the row shape the ML pipeline consumes.
LogRecord: TypeAlias = dict[str, LogValue]


class LogUploadRequest(BaseModel):
    """Describes a single log file submitted to the upload endpoint.

    Built by the router from the received multipart part rather than validated as
    a JSON body, so it holds only what the transport can observe about the file.
    Every property that can be rejected -- the filename, the extension, the
    declared and actual size, the parseable content -- is checked by
    :mod:`app.services.upload_service`, which raises the typed errors that the
    router maps onto status codes. Keeping that policy in one place is what lets
    ``UnsupportedFileTypeError`` become a 415 instead of the 422 that a plain
    field constraint would produce.
    """

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(
        min_length=1,
        description="Original client-supplied file name, used for display only.",
    )
    content_type: str | None = Field(
        default=None,
        description="Media type declared by the client for the multipart part.",
    )
    declared_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Size the transport claims, when it reports one. Advisory only: the "
            "service enforces the limit against bytes actually received."
        ),
    )


class LogColumnInfo(BaseModel):
    """Type summary for one column of a parsed log file."""

    name: str
    value_type: LogValueType
    null_count: int = Field(ge=0, description="Cells in the parsed rows that were null.")


class LogPreview(BaseModel):
    """Structural summary of a parsed log file.

    Returned alongside the stored record so a client can confirm the file landed
    in the shape it expected without downloading it again.
    """

    file_format: LogFileFormat
    columns: list[LogColumnInfo]
    row_count: int = Field(ge=0, description="Number of parsed records.")
    column_count: int = Field(ge=0, description="Number of distinct columns.")
    truncated: bool = Field(
        description=(
            "True when the file held more records than the parser retains. The "
            "file on disk is complete; only the in-memory summary is capped."
        )
    )
    sample_rows: list[LogRecord] = Field(
        description="First few records, for display. Never the whole file."
    )


class UploadedLogRead(BaseModel):
    """Public representation of an ingested log file.

    ``file_path`` is intentionally absent so that no route can disclose the
    server-side location where uploads are stored.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    upload_status: UploadStatus
    uploaded_by: int
    created_at: datetime


class LogUploadResponse(UploadedLogRead):
    """Ingestion result: the stored record plus a preview of its contents."""

    preview: LogPreview
