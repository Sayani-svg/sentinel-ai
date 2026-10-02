"""Tests for the log ingestion slice: parsing, storage and the upload endpoint.

The suite is layered to match the module layout. The parsing and storage helpers
in :mod:`app.services.log_parser` and :mod:`app.services.upload_service` are
exercised directly, because the interesting behaviour -- delimiter sniffing,
type coercion, the size ceiling, the staging-file lifecycle -- is invisible from
the outside. The endpoint tests then cover what a client actually observes: the
response body, the status code for each rejection, and the state left in the
database and on disk afterwards.

Two conventions are worth stating up front.

First, the upload directory is redirected into ``tmp_path`` for every test. The
configured default points inside the repository, and a suite that wrote there
would leave untracked files behind on every run.

Second, the size ceiling is lowered to one megabyte so that the rejection paths
can be proven without allocating the hundred megabytes the production default
allows. Every other payload in this file is a few dozen bytes.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.core.dependencies import get_db
from app.core.security import create_access_token, hash_password
from app.database.base import Base
from app.main import create_app
from app.models.uploaded_log import UploadedLog
from app.models.user import User
from app.schemas.upload import (
    LogFileFormat,
    LogRecord,
    LogUploadRequest,
    LogValueType,
    UploadStatus,
)
from app.schemas.user import UserRole
from app.services import log_parser
from app.services.log_parser import (
    EmptyLogError,
    MalformedLogError,
    ParsedLog,
    build_columns,
    coerce_scalar,
    collect_column_names,
    detect_csv_delimiter,
    parse_csv_log,
    parse_json_log,
    parse_log_file,
)
from app.services.upload_service import (
    EmptyFileError,
    FileTooLargeError,
    LogStorageError,
    UnsafeFilenameError,
    UnsupportedFileTypeError,
    allocate_destination,
    build_preview,
    ingest_log_upload,
    resolve_log_format,
    sanitize_filename,
    write_upload,
)

if TYPE_CHECKING:
    from fastapi.responses import Response
    from fastapi.testclient import TestClient as TestClientType

UPLOAD_URL = "/api/v1/upload/logs"

VALID_PASSWORD = "correct-horse-battery-staple"

#: Size ceiling used by every fixture in this module, in megabytes. One is the
#: smallest value the settings validator accepts.
TEST_MAX_UPLOAD_MB = 1

TEST_MAX_UPLOAD_BYTES = TEST_MAX_UPLOAD_MB * 1024 * 1024

#: A well-formed CSV export: three columns, mixed types, one empty cell.
VALID_CSV = "src_ip,bytes,label\n10.0.0.1,1024,benign\n10.0.0.2,,DoS Hulk\n"

#: The same content as a JSON array, which is the format the frontend sends.
VALID_JSON_RECORDS: list[dict[str, object]] = [
    {"src_ip": "10.0.0.1", "bytes": 1024, "label": "benign"},
    {"src_ip": "10.0.0.2", "bytes": 2048, "label": "DoS Hulk"},
]

#: One object per line, the third layout the parser accepts.
VALID_JSON_LINES = "\n".join(json.dumps(record) for record in VALID_JSON_RECORDS)

#: A CSV row that carries one field more than the header declares.
RAGGED_CSV = "a,b\n1,2,3\n"


def stream(text: str) -> io.StringIO:
    """Return ``text`` as the text stream the readers consume.

    Args:
        text: Decoded file contents.

    Returns:
        io.StringIO: A stream positioned at the start of the contents.
    """
    return io.StringIO(text)


def seed_user(engine: Engine, *, role: str, email: str = "ada@example.com") -> int:
    """Insert an account that may authenticate against the endpoint.

    Args:
        engine: Test database engine.
        role: Role stored on the row.
        email: Email address for the new account.

    Returns:
        int: The generated primary key.
    """
    with Session(bind=engine) as db:
        user = User(
            name="Ada Lovelace",
            email=email,
            password_hash=hash_password(VALID_PASSWORD),
            role=role,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user)
        db.commit()
        return int(user.id)


def bearer(user_id: int) -> dict[str, str]:
    """Build an authorization header for an account id.

    Args:
        user_id: Account id to embed as the token subject.

    Returns:
        dict[str, str]: A bearer authorization header.
    """
    return {"Authorization": f"Bearer {create_access_token(subject=str(user_id))}"}


def upload_settings(upload_dir: Path) -> Settings:
    """Build settings that store uploads under ``upload_dir``.

    Args:
        upload_dir: Directory that receives ingested files.

    Returns:
        Settings: A copy of the cached settings with the upload fields changed.
    """
    return get_settings().model_copy(
        update={
            "UPLOAD_DIR": upload_dir,
            "MAX_UPLOAD_SIZE_MB": TEST_MAX_UPLOAD_MB,
        }
    )


def post_log(
    client: TestClientType,
    *,
    filename: str,
    content: bytes,
    content_type: str = "text/csv",
    user_id: int | None = None,
) -> Response:
    """Submit one log file to the upload endpoint.

    Args:
        client: Client bound to an application with a test upload directory.
        filename: File name to present in the multipart part.
        content: Raw bytes to upload.
        content_type: Media type declared for the part.
        user_id: Account to authenticate as, or ``None`` to send no credentials.

    Returns:
        Response: The client's response to the upload.
    """
    headers = {} if user_id is None else bearer(user_id)
    return client.post(
        UPLOAD_URL,
        files={"file": (filename, content, content_type)},
        headers=headers,
    )


def load_logs(engine: Engine) -> list[UploadedLog]:
    """Read every ``uploaded_logs`` row in insertion order.

    Args:
        engine: Test database engine.

    Returns:
        list[UploadedLog]: Persisted ingestion records, detached from a session.
    """
    with Session(bind=engine) as db:
        return list(db.scalars(select(UploadedLog).order_by(UploadedLog.id)))


def csv_payload() -> bytes:
    """Return the standard CSV fixture as upload bytes.

    Returns:
        bytes: The encoded CSV fixture.
    """
    return VALID_CSV.encode("utf-8")


@pytest.fixture()
def upload_client(tmp_path: Path, engine: Engine) -> Iterator[tuple[TestClient, Path]]:
    """Yield a client whose uploads land in a throwaway directory.

    Args:
        tmp_path: Per-test temporary directory.
        engine: Test database engine.

    Yields:
        tuple[TestClient, Path]: The client and the directory receiving uploads.
    """
    upload_dir = tmp_path / "uploads"
    settings = upload_settings(upload_dir)
    application = create_app(settings)

    def override_get_db() -> Iterator[Session]:
        """Serve route handlers from the per-test database."""
        db = Session(bind=engine, expire_on_commit=False)
        try:
            yield db
        finally:
            db.close()

    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[get_settings] = lambda: settings

    with TestClient(application) as client:
        yield client, upload_dir


@pytest.fixture()
def ingestion() -> Iterator[tuple[Session, Engine]]:
    """Yield a private database for driving the service without a transport.

    The service commits its own work, so it cannot share the rolled-back session
    the ``db_session`` fixture provides.

    Yields:
        tuple[Session, Engine]: An open session and the engine backing it.
    """
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = Session(bind=engine, expire_on_commit=False)
    try:
        yield db, engine
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def seed_owner(db: Session) -> int:
    """Insert the account an ingestion is attributed to.

    Args:
        db: Session to insert through.

    Returns:
        int: Id of the stored account.
    """
    user = User(
        name="Ada Lovelace",
        email="ada@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=UserRole.ANALYST.value,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    db.commit()
    return int(user.id)


class TestLogUploadRequestSchema:
    """``LogUploadRequest``, the transport-level description of one part."""

    def test_an_empty_filename_is_refused_by_the_schema(self) -> None:
        """A part with no name cannot be described, so the field requires one.

        A name that only looks empty survives the schema and is caught by
        ``sanitize_filename``, which trims before judging it.
        """
        with pytest.raises(ValidationError):
            LogUploadRequest(filename="")

        assert LogUploadRequest(filename="  ").filename == "  "

    def test_a_negative_declared_size_is_refused(self) -> None:
        """A transport cannot report a negative size for a file it received."""
        with pytest.raises(ValidationError):
            LogUploadRequest(filename="events.csv", declared_size_bytes=-1)

    def test_unknown_fields_are_refused(self) -> None:
        """Extra keys are a client bug rather than something to ignore silently."""
        with pytest.raises(ValidationError):
            LogUploadRequest(filename="events.csv", unexpected="value")


class TestSanitizeFilename:
    """``sanitize_filename``."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("events.csv", "events.csv"),
            ("  events.csv  ", "events.csv"),
            ("logs from monday.csv", "logs from monday.csv"),
            ("réseau.txt", "réseau.txt"),
        ],
    )
    def test_usable_names_are_returned_trimmed(self, raw: str, expected: str) -> None:
        """A name that is safe for display survives with its whitespace removed."""
        assert sanitize_filename(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            ".",
            "..",
            "../escape.csv",
            "..\\escape.csv",
            "nested/events.csv",
            "nested\\events.csv",
            "null\x00byte.csv",
            "bell\x07name.csv",
            "newline\nname.csv",
            "a" * 256 + ".csv",
        ],
    )
    def test_unusable_names_are_rejected(self, raw: str) -> None:
        """Traversal, control characters and overlong names never reach the column."""
        with pytest.raises(UnsafeFilenameError):
            sanitize_filename(raw)

    def test_a_name_at_the_length_limit_is_accepted(self) -> None:
        """The boundary itself is legal; only what exceeds it is refused."""
        name = "a" * (255 - len(".csv")) + ".csv"

        assert len(sanitize_filename(name)) == 255


class TestResolveLogFormat:
    """``resolve_log_format``."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("events.csv", LogFileFormat.CSV),
            ("EVENTS.CSV", LogFileFormat.CSV),
            ("events.json", LogFileFormat.JSON),
            ("EVENTS.JSON", LogFileFormat.JSON),
            ("archive.tar.csv", LogFileFormat.CSV),
        ],
    )
    def test_permitted_extensions_select_a_reader(
        self, filename: str, expected: LogFileFormat
    ) -> None:
        """The extension alone decides, and matching is case insensitive."""
        assert resolve_log_format(filename, get_settings()) is expected

    @pytest.mark.parametrize(
        "filename", ["notes.txt", "archive.tar.gz", "noextension", ""]
    )
    def test_unconfigured_extensions_are_refused(self, filename: str) -> None:
        """A file type the deployment never configured has no reader."""
        with pytest.raises(UnsupportedFileTypeError) as failure:
            resolve_log_format(filename, get_settings())

        assert ".csv" in str(failure.value)

    def test_a_configured_but_unimplemented_extension_is_refused(self) -> None:
        """Configuration cannot promise a reader the code does not contain.

        ``SUPPORTED_UPLOAD_EXTENSIONS`` is deployment data, so an operator can
        legitimately add ``.txt`` to it. Accepting the file and then failing to
        read it would be worse than refusing it here.
        """
        settings = get_settings().model_copy(
            update={"SUPPORTED_UPLOAD_EXTENSIONS": [".txt"]}
        )

        with pytest.raises(UnsupportedFileTypeError) as failure:
            resolve_log_format("notes.txt", settings)

        assert "no reader is implemented" in str(failure.value)


class TestAllocateDestination:
    """``allocate_destination``."""

    def test_directory_is_created_on_demand(self, tmp_path: Path) -> None:
        """A fresh checkout has no upload directory, so it is made on first use."""
        upload_dir = tmp_path / "missing" / "uploads"

        destination = allocate_destination(upload_dir, LogFileFormat.CSV)

        assert upload_dir.is_dir()
        assert destination.parent == upload_dir
        assert destination.suffix == ".csv"

    def test_names_do_not_derive_from_the_clients_name(self, tmp_path: Path) -> None:
        """The stored name is generated, which is what makes traversal impossible."""
        destination = allocate_destination(tmp_path, LogFileFormat.JSON)

        assert destination.name != "events.csv"
        assert destination.suffix == ".json"

    def test_each_call_reserves_a_distinct_path(self, tmp_path: Path) -> None:
        """Two concurrent uploads never collide on one file."""
        first = allocate_destination(tmp_path, LogFileFormat.CSV)
        second = allocate_destination(tmp_path, LogFileFormat.CSV)

        assert first != second

    def test_an_unusable_directory_is_reported_as_a_storage_failure(
        self, tmp_path: Path
    ) -> None:
        """A path that cannot be created fails loudly rather than mid-transfer.

        A regular file occupies the parent, so the directory cannot exist.
        """
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(LogStorageError):
            allocate_destination(blocker / "uploads", LogFileFormat.CSV)


class TestWriteUpload:
    """``write_upload``."""

    def test_chunks_are_streamed_to_the_destination(self, tmp_path: Path) -> None:
        """The bytes arrive intact and the count reflects what was written."""
        destination = tmp_path / "log.csv"

        written = write_upload(destination, [b"abc", b"", b"de"], max_bytes=64)

        assert written == 5
        assert destination.read_bytes() == b"abcde"

    def test_no_partial_file_survives_a_successful_write(self, tmp_path: Path) -> None:
        """The staging file is moved into place, not copied."""
        destination = tmp_path / "log.csv"

        write_upload(destination, [b"payload"], max_bytes=64)

        assert [path.name for path in tmp_path.iterdir()] == ["log.csv"]

    def test_exceeding_the_ceiling_is_rejected_and_cleans_up(
        self, tmp_path: Path
    ) -> None:
        """The limit is enforced while streaming, so the check cannot be skipped."""
        destination = tmp_path / "log.csv"

        with pytest.raises(FileTooLargeError):
            write_upload(destination, [b"12345678", b"90"], max_bytes=9)

        assert list(tmp_path.iterdir()) == []

    def test_an_upload_of_exactly_the_limit_is_accepted(self, tmp_path: Path) -> None:
        """The boundary belongs to the accepted side."""
        destination = tmp_path / "log.csv"

        written = write_upload(destination, [b"123456789"], max_bytes=9)

        assert written == 9
        assert destination.read_bytes() == b"123456789"

    @pytest.mark.parametrize("source", [[], [b""], [b"", b""]])
    def test_an_upload_without_bytes_is_rejected(
        self, tmp_path: Path, source: list[bytes]
    ) -> None:
        """A part carrying nothing is refused and leaves no file behind."""
        destination = tmp_path / "log.csv"

        with pytest.raises(EmptyFileError):
            write_upload(destination, source, max_bytes=64)

        assert list(tmp_path.iterdir()) == []


class TestCoerceScalar:
    """``coerce_scalar``."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("benign", "benign"),
            ("10.0.0.1", "10.0.0.1"),
            ("42", 42),
            ("-42", -42),
            ("+42", 42),
            ("007", 7),
            ("3.5", 3.5),
            ("-0.5", -0.5),
            (".5", 0.5),
            ("1e3", 1000.0),
            ("2.5E-2", 0.025),
            ("true", True),
            ("TRUE", True),
            ("False", False),
            (True, True),
            (False, False),
            (7, 7),
            (1.5, 1.5),
            (float("inf"), "inf"),
            (float("nan"), "nan"),
        ],
    )
    def test_cells_are_normalised_to_the_closed_value_set(
        self, raw: object, expected: object
    ) -> None:
        """Every cell becomes a scalar a dataframe can hold."""
        assert coerce_scalar(raw) == expected

    @pytest.mark.parametrize("raw", ["1_0", "nan", "inf", "-inf", "0x1f", "12,5"])
    def test_spellings_that_are_not_numbers_stay_text(self, raw: str) -> None:
        """Python and locale number spellings are not silently converted.

        ``1_0`` is a valid Python integer literal and ``12,5`` a valid decimal in
        much of the world; neither is the number an operator wrote in the log.
        """
        assert coerce_scalar(raw) == raw

    def test_nested_structures_are_serialised_rather_than_dropped(self) -> None:
        """A nested object survives the trip as deterministic text."""
        coerced = coerce_scalar({"b": 2, "a": [1, {"c": 3}]})

        assert coerced == '{"a":[1,{"c":3}],"b":2}'
        assert json.loads(str(coerced)) == {"a": [1, {"c": 3}], "b": 2}


class TestColumnSummaries:
    """``collect_column_names`` and ``build_columns``."""

    def test_column_names_are_the_union_in_first_seen_order(self) -> None:
        """Ragged JSON records report every key they use, in first-seen order."""
        records: tuple[LogRecord, ...] = (
            {"src_ip": "10.0.0.1"},
            {"label": "benign", "src_ip": "10.0.0.2"},
        )

        assert collect_column_names(records) == ("src_ip", "label")

    def test_null_counts_include_keys_absent_from_a_record(self) -> None:
        """A key another record carried counts as null for the records missing it."""
        records: tuple[LogRecord, ...] = (
            {"src_ip": "10.0.0.1", "label": None},
            {"label": "benign"},
        )

        columns = build_columns(collect_column_names(records), records)

        assert {column.name: column.null_count for column in columns} == {
            "src_ip": 1,
            "label": 1,
        }

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (["1", "2"], LogValueType.INTEGER),
            (["1", "2.5"], LogValueType.FLOAT),
            (["true", "false"], LogValueType.BOOLEAN),
            (["benign", "10.0.0.1"], LogValueType.STRING),
            (["", "benign"], LogValueType.STRING),
            (["", "  "], LogValueType.NULL),
            (["1", "benign"], LogValueType.MIXED),
            (["true", "1"], LogValueType.MIXED),
        ],
    )
    def test_column_types_are_reduced_to_one_reported_type(
        self, values: list[str], expected: LogValueType
    ) -> None:
        """Integers widen to float; anything else heterogeneous reports as mixed."""
        records: tuple[LogRecord, ...] = tuple(
            {"column": coerce_scalar(value)} for value in values
        )

        assert build_columns(("column",), records)[0].value_type is expected


class TestParseCsvLog:
    """``parse_csv_log``."""

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("a,b,c", ","),
            ("a;b;c", ";"),
            ("a\tb\tc", "\t"),
            ("a|b|c", "|"),
            ("only", ","),
        ],
    )
    def test_the_delimiter_is_taken_from_the_header(
        self, header: str, expected: str
    ) -> None:
        """Security exports use whichever separator their tool defaults to."""
        assert detect_csv_delimiter(header) == expected

    @pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
    def test_every_supported_delimiter_parses(self, delimiter: str) -> None:
        """Each candidate delimiter round-trips into records."""
        text = f"src_ip{delimiter}bytes\n10.0.0.1{delimiter}1024\n"

        parsed = parse_csv_log(stream(text))

        assert parsed.rows == ({"src_ip": "10.0.0.1", "bytes": 1024},)

    def test_records_are_coerced_and_summarised(self) -> None:
        """The preview describes the file without a second pass by the client."""
        parsed = parse_csv_log(stream(VALID_CSV))

        assert parsed.file_format is LogFileFormat.CSV
        assert parsed.row_count == 2
        assert parsed.column_names == ("src_ip", "bytes", "label")
        assert [column.value_type for column in parsed.columns] == [
            LogValueType.STRING,
            LogValueType.INTEGER,
            LogValueType.STRING,
        ]
        assert parsed.columns[1].null_count == 1

    def test_blank_lines_are_skipped(self) -> None:
        """A trailing newline is not a structural defect."""
        parsed = parse_csv_log(stream("a,b\n1,2\n\n\n3,4\n\n"))

        assert parsed.row_count == 2

    def test_quoted_fields_containing_the_delimiter_survive(self) -> None:
        """Embedded separators stay inside their field."""
        parsed = parse_csv_log(stream('message,count\n"a,b,c",2\n'))

        assert parsed.rows == ({"message": "a,b,c", "count": 2},)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (RAGGED_CSV, "3 fields but the header declares 2"),
            ("a,b\n1\n", "1 fields but the header declares 2"),
            ("a,,c\n1,2\n", "no name"),
            ("a,a\n1,2\n", "more than once"),
            ("\n\n", "readable content"),
        ],
    )
    def test_structural_defects_are_rejected(self, text: str, expected: str) -> None:
        """A file that does not match its layout promises a usable error."""
        with pytest.raises((MalformedLogError, EmptyLogError)) as failure:
            parse_csv_log(stream(text))

        assert expected in str(failure.value)

    def test_a_header_without_data_rows_is_rejected(self) -> None:
        """Column names alone are not a log."""
        with pytest.raises(EmptyLogError):
            parse_csv_log(stream("a,b\n"))

    def test_the_row_cap_is_reported_rather_than_silently_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client can tell a capped summary from a complete one."""
        monkeypatch.setattr(log_parser, "MAX_LOG_ROWS", 2)
        parsed = parse_csv_log(stream("a\n1\n2\n3\n4\n"))

        assert parsed.row_count == 2
        assert parsed.truncated is True


class TestParseJsonLog:
    """``parse_json_log``."""

    def test_an_array_of_objects_is_parsed(self) -> None:
        """The array form the frontend sends becomes typed records."""
        parsed = parse_json_log(json.dumps(VALID_JSON_RECORDS))

        assert parsed.file_format is LogFileFormat.JSON
        assert parsed.row_count == 2
        assert parsed.column_names == ("src_ip", "bytes", "label")
        assert parsed.truncated is False

    def test_json_lines_are_parsed(self) -> None:
        """One object per line is the third accepted layout."""
        parsed = parse_json_log(VALID_JSON_LINES)

        assert parsed.row_count == 2
        assert parsed.rows[1] == {
            "src_ip": "10.0.0.2",
            "bytes": 2048,
            "label": "DoS Hulk",
        }

    def test_a_lone_object_is_read_as_a_single_record(self) -> None:
        """A one-line export is a log, not a malformed file."""
        parsed = parse_json_log('{"src_ip": "10.0.0.1"}')

        assert parsed.rows == ({"src_ip": "10.0.0.1"},)

    def test_records_with_different_schemas_are_merged(self) -> None:
        """A key missing from a record is reported as null rather than dropped."""
        parsed = parse_json_log('[{"a": 1}, {"b": "x"}]')

        assert parsed.column_names == ("a", "b")
        assert parsed.columns[0].null_count == 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[]", "empty array"),
            ("   ", "readable content"),
            ("[1, 2]", "must be a JSON object"),
            ('["a", "b"]', "must be a JSON object"),
            ('{"a": 1', "not valid JSON"),
            ('{"a": 1} {"a": 2', "not valid JSON"),
            ('"just a string"', "neither a JSON array"),
            ("42", "neither a JSON array"),
            ('[{"a": 1}, 7]', "must be a JSON object"),
        ],
    )
    def test_malformed_json_is_rejected(self, text: str, expected: str) -> None:
        """The error names the real problem instead of a speculative one."""
        with pytest.raises((MalformedLogError, EmptyLogError)) as failure:
            parse_json_log(text)

        assert expected in str(failure.value)

    def test_the_row_cap_is_reported_rather_than_silently_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truncated JSON summary is labelled as such."""
        monkeypatch.setattr(log_parser, "MAX_LOG_ROWS", 1)
        parsed = parse_json_log(json.dumps(VALID_JSON_RECORDS))

        assert parsed.row_count == 1
        assert parsed.truncated is True


class TestParseLogFile:
    """``parse_log_file``."""

    def test_a_csv_file_is_read_from_disk(self, tmp_path: Path) -> None:
        """The service hands the parser a path, not an open handle."""
        stored = tmp_path / "uuid.csv"
        stored.write_text(VALID_CSV, encoding="utf-8")

        parsed = parse_log_file(stored, LogFileFormat.CSV)

        assert parsed.file_format is LogFileFormat.CSV
        assert parsed.row_count == 2

    def test_a_byte_order_mark_does_not_corrupt_the_first_column(
        self, tmp_path: Path
    ) -> None:
        """Spreadsheet exports add a BOM that must not reach the column name."""
        stored = tmp_path / "uuid.csv"
        stored.write_bytes(b"\xef\xbb\xbf" + VALID_CSV.encode("utf-8"))

        parsed = parse_log_file(stored, LogFileFormat.CSV)

        assert parsed.column_names == ("src_ip", "bytes", "label")

    def test_a_json_file_is_read_from_disk(self, tmp_path: Path) -> None:
        """The JSON reader accepts either layout from the same extension."""
        stored = tmp_path / "uuid.json"
        stored.write_text(VALID_JSON_LINES, encoding="utf-8")

        assert parse_log_file(stored, LogFileFormat.JSON).row_count == 2


class TestParsedLogAccessors:
    """The read-only helpers on ``ParsedLog``."""

    @staticmethod
    def parsed() -> ParsedLog:
        """Return a fixed parsed log holding three records.

        Returns:
            ParsedLog: A three-row single-column log.
        """
        return parse_csv_log(stream("a\n1\n2\n3\n"))

    def test_sample_rows_are_capped(self) -> None:
        """A preview is a sample, never the whole file."""
        parsed = self.parsed()

        assert len(parsed.sample_rows()) == 3
        assert len(parsed.sample_rows(limit=2)) == 2

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_sample_limit_yields_nothing(self, limit: int) -> None:
        """A caller cannot accidentally slice from the end of the records."""
        assert self.parsed().sample_rows(limit=limit) == ()

    def test_to_records_returns_independent_copies(self) -> None:
        """A caller may mutate the result without corrupting the parsed log."""
        parsed = self.parsed()

        records = parsed.to_records()
        records[0]["a"] = "mutated"

        assert parsed.rows[0]["a"] == 1


class TestBuildPreview:
    """``build_preview``."""

    def test_the_preview_reports_shape_counts_and_a_sample(self) -> None:
        """One round trip gives the client everything it needs to confirm the file."""
        preview = build_preview(parse_csv_log(stream(VALID_CSV)))

        assert preview.file_format is LogFileFormat.CSV
        assert preview.row_count == 2
        assert preview.column_count == 3
        assert preview.truncated is False
        assert preview.sample_rows == [
            {"src_ip": "10.0.0.1", "bytes": 1024, "label": "benign"},
            {"src_ip": "10.0.0.2", "bytes": None, "label": "DoS Hulk"},
        ]
        assert [column.name for column in preview.columns] == [
            "src_ip",
            "bytes",
            "label",
        ]


class TestIngestLogUploadService:
    """``ingest_log_upload`` driven without a transport."""

    def test_a_valid_file_is_stored_parsed_and_marked_completed(
        self, ingestion: tuple[Session, Engine], tmp_path: Path
    ) -> None:
        """The service completes the lifecycle the router relies on."""
        db, _engine = ingestion
        settings = upload_settings(tmp_path / "uploads")

        result = ingest_log_upload(
            db,
            LogUploadRequest(filename="events.csv"),
            [VALID_CSV.encode("utf-8")],
            uploaded_by=seed_owner(db),
            settings=settings,
        )

        assert result.log.upload_status == UploadStatus.COMPLETED.value
        assert result.parsed.row_count == 2
        assert Path(result.log.file_path).read_text(encoding="utf-8-sig") == VALID_CSV

    def test_the_stored_name_is_generated_not_the_clients(
        self, ingestion: tuple[Session, Engine], tmp_path: Path
    ) -> None:
        """The display name and the on-disk name are deliberately different."""
        db, _engine = ingestion
        settings = upload_settings(tmp_path / "uploads")

        result = ingest_log_upload(
            db,
            LogUploadRequest(filename="events.csv"),
            [VALID_CSV.encode("utf-8")],
            uploaded_by=seed_owner(db),
            settings=settings,
        )

        stored = Path(result.log.file_path)
        assert result.log.filename == "events.csv"
        assert stored.name != result.log.filename
        assert stored.parent == settings.UPLOAD_DIR

    def test_a_declared_size_over_the_limit_is_refused_before_anything_is_written(
        self, ingestion: tuple[Session, Engine], tmp_path: Path
    ) -> None:
        """A part that announces too much is rejected without storing a byte."""
        db, engine = ingestion
        settings = upload_settings(tmp_path / "uploads")

        with pytest.raises(FileTooLargeError):
            ingest_log_upload(
                db,
                LogUploadRequest(
                    filename="events.csv",
                    declared_size_bytes=TEST_MAX_UPLOAD_BYTES + 1,
                ),
                [VALID_CSV.encode("utf-8")],
                uploaded_by=seed_owner(db),
                settings=settings,
            )

        assert load_logs(engine) == []
        assert not (tmp_path / "uploads").exists()

    def test_undeclared_bytes_over_the_limit_are_refused_while_streaming(
        self, ingestion: tuple[Session, Engine], tmp_path: Path
    ) -> None:
        """The ceiling is enforced on the bytes, not only on the client's claim.

        A transport that reports no size cannot be trusted, so the guard on
        received bytes has to stand on its own.
        """
        db, engine = ingestion
        settings = upload_settings(tmp_path / "uploads")

        def source() -> Iterator[bytes]:
            """Yield one megabyte at a time, past the configured ceiling."""
            while True:
                yield b"x" * (1024 * 1024)

        with pytest.raises(FileTooLargeError):
            ingest_log_upload(
                db,
                LogUploadRequest(filename="events.csv", declared_size_bytes=None),
                source(),
                uploaded_by=seed_owner(db),
                settings=settings,
            )

        assert load_logs(engine)[0].upload_status == UploadStatus.FAILED.value
        assert list((tmp_path / "uploads").iterdir()) == []

    @pytest.mark.parametrize(
        ("filename", "content", "failure", "expect_record"),
        [
            ("notes.txt", b"a,b\n1,2\n", UnsupportedFileTypeError, False),
            ("../escape.csv", b"a,b\n1,2\n", UnsafeFilenameError, False),
            ("events.csv", b"", EmptyFileError, True),
            ("events.csv", RAGGED_CSV.encode("utf-8"), MalformedLogError, True),
            ("events.json", b"not json", MalformedLogError, True),
        ],
    )
    def test_a_rejected_upload_leaves_a_failed_record_and_no_file(
        self,
        ingestion: tuple[Session, Engine],
        tmp_path: Path,
        filename: str,
        content: bytes,
        failure: type[Exception],
        expect_record: bool,
    ) -> None:
        """A malformed submission is visible to an operator but stores nothing.

        Filenames and extensions are settled before the record exists, so those
        rejections leave no trace. Anything discovered while reading the bytes
        does, which is what makes a repeatedly rejected export diagnosable.
        """
        db, engine = ingestion
        settings = upload_settings(tmp_path / "uploads")

        with pytest.raises(failure):
            ingest_log_upload(
                db,
                LogUploadRequest(filename=filename),
                [content],
                uploaded_by=seed_owner(db),
                settings=settings,
            )

        if not expect_record:
            assert load_logs(engine) == []
            assert not (tmp_path / "uploads").exists()
            return

        failed = load_logs(engine)
        assert [row.upload_status for row in failed] == [UploadStatus.FAILED.value]
        assert list((tmp_path / "uploads").iterdir()) == []


class TestUploadEndpoint:
    """``POST /api/v1/upload/logs``."""

    def test_a_csv_upload_is_accepted_and_recorded(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """The happy path returns the record plus a preview of the contents."""
        client, _upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client,
            filename="events.csv",
            content=csv_payload(),
            user_id=user_id,
        )

        assert response.status_code == 201
        body = response.json()
        assert body["filename"] == "events.csv"
        assert body["upload_status"] == UploadStatus.COMPLETED.value
        assert body["uploaded_by"] == user_id
        assert isinstance(body["id"], int)
        datetime.fromisoformat(body["created_at"])

        preview = body["preview"]
        assert preview["file_format"] == "csv"
        assert preview["row_count"] == 2
        assert preview["column_count"] == 3
        assert preview["truncated"] is False
        assert preview["sample_rows"] == [
            {"src_ip": "10.0.0.1", "bytes": 1024, "label": "benign"},
            {"src_ip": "10.0.0.2", "bytes": None, "label": "DoS Hulk"},
        ]

        stored = load_logs(engine)
        assert len(stored) == 1
        assert stored[0].id == body["id"]
        assert stored[0].upload_status == UploadStatus.COMPLETED.value

    @pytest.mark.parametrize(
        ("filename", "content", "expected_format"),
        [
            ("events.csv", VALID_CSV, "csv"),
            ("events.json", json.dumps(VALID_JSON_RECORDS), "json"),
            ("events.json", VALID_JSON_LINES, "json"),
        ],
    )
    def test_every_accepted_layout_ingests(
        self,
        upload_client: tuple[TestClient, Path],
        engine: Engine,
        filename: str,
        content: str,
        expected_format: str,
    ) -> None:
        """CSV, a JSON array and JSON Lines all reach the completed state."""
        client, _upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client,
            filename=filename,
            content=content.encode("utf-8"),
            user_id=user_id,
        )

        assert response.status_code == 201
        body = response.json()
        assert body["preview"]["file_format"] == expected_format
        assert body["preview"]["row_count"] == 2

    def test_the_stored_bytes_match_what_was_uploaded(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """Streaming to disk is lossless, including trailing whitespace."""
        client, upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        post_log(client, filename="events.csv", content=csv_payload(), user_id=user_id)

        stored = load_logs(engine)[0]
        assert Path(stored.file_path).read_bytes() == VALID_CSV.encode("utf-8")
        assert [path.name for path in upload_dir.iterdir()] == [
            Path(stored.file_path).name
        ]

    def test_the_server_side_path_is_never_disclosed(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """The response schema omits ``file_path`` entirely.

        The column exists and later slices need it, but a deployment's directory
        layout is not a client's business.
        """
        client, upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client, filename="events.csv", content=csv_payload(), user_id=user_id
        )
        body = response.json()
        serialised = json.dumps(body)

        assert set(body) == {
            "id",
            "filename",
            "upload_status",
            "uploaded_by",
            "created_at",
            "preview",
        }
        assert "file_path" not in serialised
        assert str(upload_dir) not in serialised

    @pytest.mark.parametrize("role", [role.value for role in UserRole])
    def test_every_role_may_ingest(
        self,
        upload_client: tuple[TestClient, Path],
        engine: Engine,
        role: str,
    ) -> None:
        """Uploading is open to any usable account, not just analysts."""
        client, _upload_dir = upload_client
        user_id = seed_user(engine, role=role, email=f"{role}@example.com")

        response = post_log(
            client, filename="events.csv", content=csv_payload(), user_id=user_id
        )

        assert response.status_code == 201
        assert response.json()["uploaded_by"] == user_id

    def test_an_unrecognised_role_is_refused(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """The role guard runs before the file is touched."""
        client, upload_dir = upload_client
        user_id = seed_user(engine, role="Overlord")

        response = post_log(
            client, filename="events.csv", content=csv_payload(), user_id=user_id
        )

        assert response.status_code == 403
        assert load_logs(engine) == []
        assert not upload_dir.exists()

    def test_an_anonymous_upload_is_refused(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """Ingestion requires a bearer token."""
        client, _upload_dir = upload_client

        response = post_log(client, filename="events.csv", content=csv_payload())

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert load_logs(engine) == []

    def test_a_missing_file_part_is_refused(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """The endpoint declares ``file`` as required, so omitting it is a 422."""
        client, _upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = client.post(UPLOAD_URL, headers=bearer(user_id))

        assert response.status_code == 422
        assert load_logs(engine) == []

    @pytest.mark.parametrize("filename", ["notes.txt", "archive.tar.gz", "noextension"])
    def test_an_unsupported_type_is_refused_with_415(
        self,
        upload_client: tuple[TestClient, Path],
        engine: Engine,
        filename: str,
    ) -> None:
        """An unusable media type is distinct from unusable contents."""
        client, _upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client, filename=filename, content=csv_payload(), user_id=user_id
        )

        assert response.status_code == 415
        assert load_logs(engine) == []

    @pytest.mark.parametrize("filename", ["../escape.csv", "nested\\events.csv", "  "])
    def test_an_unusable_filename_is_refused_with_400(
        self,
        upload_client: tuple[TestClient, Path],
        engine: Engine,
        filename: str,
    ) -> None:
        """A traversal attempt is rejected before any file is stored."""
        client, upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client, filename=filename, content=csv_payload(), user_id=user_id
        )

        assert response.status_code == 400
        assert load_logs(engine) == []
        assert not upload_dir.exists()

    def test_an_empty_file_is_refused_with_400(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """A part with no bytes cannot become a log."""
        client, upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(client, filename="events.csv", content=b"", user_id=user_id)

        assert response.status_code == 400
        assert load_logs(engine)[0].upload_status == UploadStatus.FAILED.value
        assert list(upload_dir.iterdir()) == []

    def test_an_oversized_file_is_refused_with_413(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """The configured ceiling is enforced against the received bytes."""
        client, upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client,
            filename="events.csv",
            content=b"x" * (TEST_MAX_UPLOAD_BYTES + 1),
            user_id=user_id,
        )

        assert response.status_code == 413
        assert load_logs(engine) == []
        assert not upload_dir.exists()

    @pytest.mark.parametrize(
        ("filename", "content", "content_type"),
        [
            ("events.csv", RAGGED_CSV.encode("utf-8"), "text/csv"),
            ("events.csv", b"a,b\n", "text/csv"),
            ("events.json", b"not json", "application/json"),
            ("events.json", b'[{"a": 1}, 2]', "application/json"),
            ("events.json", b"[]", "application/json"),
        ],
    )
    def test_contents_that_contradict_the_extension_are_refused_with_422(
        self,
        upload_client: tuple[TestClient, Path],
        engine: Engine,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> None:
        """Structural validation happens after storage and is reported as 422."""
        client, upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client,
            filename=filename,
            content=content,
            content_type=content_type,
            user_id=user_id,
        )

        assert response.status_code == 422
        assert [row.upload_status for row in load_logs(engine)] == [
            UploadStatus.FAILED.value
        ]
        assert list(upload_dir.iterdir()) == []

    def test_a_declared_media_type_of_octet_stream_is_still_accepted(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """Browsers label CSV as binary often enough that the header is ignored."""
        client, _upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client,
            filename="events.csv",
            content=csv_payload(),
            content_type="application/octet-stream",
            user_id=user_id,
        )

        assert response.status_code == 201

    def test_a_filename_needing_encoding_is_stored_verbatim(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """Non-ASCII names survive the round trip for display."""
        client, _upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        response = post_log(
            client,
            filename="journal réseau 2026.csv",
            content=csv_payload(),
            user_id=user_id,
        )

        assert response.status_code == 201
        assert response.json()["filename"] == "journal réseau 2026.csv"

    def test_each_upload_is_recorded_separately(
        self, upload_client: tuple[TestClient, Path], engine: Engine
    ) -> None:
        """Two submissions of the same file produce two rows and two stored files."""
        client, upload_dir = upload_client
        user_id = seed_user(engine, role=UserRole.ANALYST.value)

        for _ in range(2):
            post_log(
                client, filename="events.csv", content=csv_payload(), user_id=user_id
            )

        stored = load_logs(engine)
        assert len({row.id for row in stored}) == 2
        assert len({row.file_path for row in stored}) == 2
        assert len(list(upload_dir.iterdir())) == 2
