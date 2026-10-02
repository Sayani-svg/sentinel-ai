"""Parsing of uploaded log files into the internal record representation.

The output is deliberately format-neutral: every accepted file becomes a
:class:`ParsedLog` holding plain :class:`dict` records whose values are drawn
from a small closed set of scalars. That is the shape the prediction slice hands
to the ML pipeline, so no downstream module has to know whether the bytes
arrived as CSV, a JSON array, or JSON Lines.

Three layouts are accepted, all of which occur in real security log exports:

* CSV with a header row, using ``,``, ``;``, tab or ``|`` as the delimiter,
  chosen by inspecting the header line.
* A JSON array of objects.
* JSON Lines, one JSON object per non-blank line.

Numbers are recognised with explicit patterns rather than by calling
:func:`int` and :func:`float`, because those accept Python-specific spellings
such as ``1_0`` and ``nan``. A cell that does not match a numeric pattern is
kept as a string, so no input is ever silently discarded.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final, TextIO

from app.core.logger import get_logger
from app.schemas.upload import LogFileFormat, LogRecord, LogValue, LogValueType

logger = get_logger(__name__)

#: Upper bound on records held in memory for one file. A file below the byte
#: limit can still describe an enormous number of very short rows, and the
#: prediction slice works in batches, so the summary is capped rather than the
#: bytes. The stored file is always complete.
MAX_LOG_ROWS: int = 100_000

#: Records echoed back in an ingestion response so a client can eyeball the
#: shape of what was accepted.
PREVIEW_ROW_COUNT: int = 5

#: Delimiters considered for CSV, in preference order.
CSV_DELIMITERS: Final[tuple[str, ...]] = (",", ";", "\t", "|")

#: Encoding used for every accepted format. ``utf-8-sig`` transparently strips a
#: byte-order mark, which spreadsheet exports commonly add and which would
#: otherwise corrupt the first column name.
LOG_FILE_ENCODING: str = "utf-8-sig"

#: Numbers are matched explicitly so that digit separators and other Python
#: literal spellings are treated as strings rather than silently converted.
_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[+-]?[0-9]+")
_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?"
)

_BOOLEAN_TRUE: Final[frozenset[str]] = frozenset({"true"})
_BOOLEAN_FALSE: Final[frozenset[str]] = frozenset({"false"})


class LogParsingError(Exception):
    """Base class for failures that make a log file unusable."""


class EmptyLogError(LogParsingError):
    """Raised when a file carries no records at all."""


class MalformedLogError(LogParsingError):
    """Raised when a file does not follow the layout its extension promises."""


@dataclass(frozen=True, slots=True)
class LogColumn:
    """Inferred shape of one column across the parsed records.

    Attributes:
        name: Column name as written in the file.
        value_type: Single type observed across non-null values, or
            :attr:`~app.schemas.upload.LogValueType.MIXED` when the column is
            heterogeneous.
        null_count: Number of parsed records whose value for this column was
            absent or empty.
    """

    name: str
    value_type: LogValueType
    null_count: int


@dataclass(frozen=True, slots=True)
class ParsedLog:
    """Format-neutral view of an ingested log file.

    Attributes:
        file_format: Layout the records were read from.
        columns: Column summaries in first-seen order.
        rows: Parsed records, capped at :data:`MAX_LOG_ROWS`.
        truncated: Whether the file held more records than were retained.
    """

    file_format: LogFileFormat
    columns: tuple[LogColumn, ...]
    rows: tuple[LogRecord, ...]
    truncated: bool

    @property
    def row_count(self) -> int:
        """Return the number of retained records.

        Returns:
            int: Size of :attr:`rows`.
        """
        return len(self.rows)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the column names in first-seen order.

        Returns:
            tuple[str, ...]: One entry per column.
        """
        return tuple(column.name for column in self.columns)

    def sample_rows(self, limit: int = PREVIEW_ROW_COUNT) -> tuple[LogRecord, ...]:
        """Return the first ``limit`` records.

        Args:
            limit: Maximum number of records to return. Values below one yield an
                empty tuple rather than raising, so a caller cannot accidentally
                slice from the end.

        Returns:
            tuple[LogRecord, ...]: Up to ``limit`` records.
        """
        if limit < 1:
            return ()
        return self.rows[:limit]

    def to_records(self) -> list[LogRecord]:
        """Return every retained record as a plain list.

        Returns:
            list[LogRecord]: Row-major records ready to become a dataframe.
        """
        return [dict(record) for record in self.rows]


def coerce_scalar(raw: object) -> LogValue:
    """Normalise one raw cell to the closed set of parsed log values.

    Text arriving from CSV is untyped, so it is interpreted as a boolean, then an
    integer, then a float, and otherwise kept as a string. Values already typed
    by a JSON decoder pass through, except that non-finite floats become strings:
    JSON has no encoding for ``NaN`` or ``Infinity``, so keeping them numeric
    would produce a response body that strict clients cannot parse. Nested JSON
    structures are serialised back to text rather than dropped.

    Args:
        raw: Cell value as read from the file.

    Returns:
        LogValue: ``None``, :class:`bool`, :class:`int`, :class:`float` or
        :class:`str`.
    """
    if raw is None:
        return None

    if isinstance(raw, bool):
        return raw

    if isinstance(raw, int):
        return raw

    if isinstance(raw, float):
        return raw if math.isfinite(raw) else repr(raw)

    if isinstance(raw, (dict, list)):
        # Sorted keys so the same object always serialises identically.
        return json.dumps(raw, sort_keys=True, separators=(",", ":"))

    if not isinstance(raw, str):
        return str(raw)

    text = raw.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in _BOOLEAN_TRUE:
        return True
    if lowered in _BOOLEAN_FALSE:
        return False

    if _INTEGER_PATTERN.fullmatch(text):
        return int(text)

    if _FLOAT_PATTERN.fullmatch(text):
        number = float(text)
        if math.isfinite(number):
            return number

    # A spelling such as "nan" or "inf" matches the float pattern but is not
    # representable in JSON, so it is preserved verbatim as text instead.
    return text


def _value_type_of(value: LogValue) -> LogValueType:
    """Return the reported type of one non-null parsed value.

    Args:
        value: A value produced by :func:`coerce_scalar`.

    Returns:
        LogValueType: The matching member.
    """
    if isinstance(value, bool):
        return LogValueType.BOOLEAN
    if isinstance(value, int):
        return LogValueType.INTEGER
    if isinstance(value, float):
        return LogValueType.FLOAT
    return LogValueType.STRING


def _combine_types(observed: set[LogValueType]) -> LogValueType:
    """Reduce the types seen in one column to a single reported type.

    Args:
        observed: Distinct types of the non-null values in a column.

    Returns:
        LogValueType: ``NULL`` when nothing was observed, ``FLOAT`` when the
        column mixes integers and floats, otherwise the single observed type, or
        ``MIXED`` when the column is heterogeneous.
    """
    if not observed:
        return LogValueType.NULL
    if len(observed) == 1:
        return next(iter(observed))
    if observed == {LogValueType.INTEGER, LogValueType.FLOAT}:
        return LogValueType.FLOAT
    return LogValueType.MIXED


def collect_column_names(records: tuple[LogRecord, ...]) -> tuple[str, ...]:
    """Return the union of record keys in first-seen order.

    JSON records need not share a schema, so a key absent from some records is
    reported as null for those records rather than treated as a defect.

    Args:
        records: Parsed records.

    Returns:
        tuple[str, ...]: Every key seen, in first-seen order.
    """
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return tuple(names)


def build_columns(
    column_names: tuple[str, ...], rows: tuple[LogRecord, ...]
) -> tuple[LogColumn, ...]:
    """Summarise each column's inferred type and null count.

    Args:
        column_names: Column names in the order they should be reported.
        rows: Parsed records to inspect.

    Returns:
        tuple[LogColumn, ...]: One summary per name, in the given order.
    """
    summaries: list[LogColumn] = []
    for name in column_names:
        observed: set[LogValueType] = set()
        null_count = 0
        for record in rows:
            value = record.get(name)
            if value is None:
                null_count += 1
            else:
                observed.add(_value_type_of(value))
        summaries.append(
            LogColumn(
                name=name,
                value_type=_combine_types(observed),
                null_count=null_count,
            )
        )
    return tuple(summaries)


def detect_csv_delimiter(header_line: str) -> str:
    """Choose the delimiter used by a CSV header line.

    The candidate with the most occurrences wins, ties resolving to the order in
    :data:`CSV_DELIMITERS`. A header containing no candidate is treated as
    single-column, which the structural checks then accept or reject.

    Args:
        header_line: First non-blank line of the file, including its newline.

    Returns:
        str: A single-character delimiter.
    """
    best = CSV_DELIMITERS[0]
    best_count = header_line.count(best)
    for candidate in CSV_DELIMITERS[1:]:
        count = header_line.count(candidate)
        if count > best_count:
            best, best_count = candidate, count
    return best


def _first_content_line(stream: TextIO) -> str:
    """Return the first line holding non-whitespace content.

    Args:
        stream: Text stream positioned at the start of the file.

    Returns:
        str: The first meaningful line, including its newline.

    Raises:
        EmptyLogError: If the stream holds no such line.
    """
    for line in stream:
        if line.strip():
            return line
    raise EmptyLogError("The file does not contain any readable content.")


def _validate_header(header: list[str]) -> tuple[str, ...]:
    """Validate and normalise a CSV header row.

    Args:
        header: Raw field names as read by :mod:`csv`.

    Returns:
        tuple[str, ...]: Trimmed column names in file order.

    Raises:
        MalformedLogError: If a name is blank or a name repeats.
    """
    names: list[str] = []
    seen: set[str] = set()
    for index, raw_name in enumerate(header):
        name = raw_name.strip()
        if not name:
            raise MalformedLogError(
                f"Column {index + 1} of the header row has no name."
            )
        if name in seen:
            raise MalformedLogError(
                f"Column name {name!r} appears more than once in the header row."
            )
        seen.add(name)
        names.append(name)
    return tuple(names)


def parse_csv_log(stream: TextIO) -> ParsedLog:
    """Parse a CSV log stream that begins with a header row.

    Records are read incrementally, so a large file never has to be
    materialised as a single string. Fully blank lines are skipped, because a
    trailing newline is not a structural defect.

    Args:
        stream: Decoded text stream positioned at the start of the file.

    Returns:
        ParsedLog: The parsed records and their column summaries.

    Raises:
        EmptyLogError: If the file has no header row or no data rows.
        MalformedLogError: If the header is unnamed or duplicated, a row has the
            wrong number of fields, or a field exceeds the CSV parser's limit.
    """
    # The header line is read to choose the delimiter, then handed back to the
    # reader so it is still parsed as the header rather than discarded.
    header_line = _first_content_line(stream)
    reader = csv.reader(
        chain([header_line], stream), delimiter=detect_csv_delimiter(header_line)
    )

    try:
        header: list[str] | None = None
        for candidate in reader:
            if any(field.strip() for field in candidate):
                header = candidate
                break

        if header is None:
            raise EmptyLogError("The CSV file does not contain a header row.")

        column_names = _validate_header(header)
        expected_width = len(column_names)

        rows: list[LogRecord] = []
        truncated = False
        for line_number, raw_row in enumerate(reader, start=2):
            if not raw_row or not any(field.strip() for field in raw_row):
                continue
            if len(raw_row) != expected_width:
                raise MalformedLogError(
                    f"Row {line_number} has {len(raw_row)} fields but the header "
                    f"declares {expected_width}."
                )
            if len(rows) >= MAX_LOG_ROWS:
                truncated = True
                break
            rows.append(
                {
                    name: coerce_scalar(value)
                    for name, value in zip(column_names, raw_row, strict=True)
                }
            )
    except csv.Error as exc:
        raise MalformedLogError(f"The CSV structure could not be read: {exc}") from exc

    if not rows:
        raise EmptyLogError("The CSV file contains a header row but no data rows.")

    frozen = tuple(rows)
    return ParsedLog(
        file_format=LogFileFormat.CSV,
        columns=build_columns(column_names, frozen),
        rows=frozen,
        truncated=truncated,
    )


def _decode_json(
    text: str, *, allow_single_record: bool
) -> dict[str, object] | list[object]:
    """Decode JSON text into either a record array or a single record.

    Args:
        text: Decoded file contents.
        allow_single_record: Whether a lone top-level object may be treated as a
            one-record log.

    Returns:
        dict[str, object] | list[object]: The decoded payload.

    Raises:
        MalformedLogError: If the text is not valid JSON, or decodes to a scalar.
    """
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedLogError(f"The file is not valid JSON: {exc}") from exc

    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, dict):
        if not allow_single_record:
            raise MalformedLogError(
                "The JSON file must contain an array of records, not a single object."
            )
        return decoded
    raise MalformedLogError(
        "The JSON file must contain an array of records, not a bare "
        f"{type(decoded).__name__}."
    )


def _records_from_payload(
    payload: dict[str, object] | list[object],
) -> tuple[list[LogRecord], bool]:
    """Validate a decoded JSON payload and normalise it to records.

    Args:
        payload: Output of :func:`_decode_json`.

    Returns:
        tuple[list[LogRecord], bool]: The records, capped at
        :data:`MAX_LOG_ROWS`, and whether anything was dropped.

    Raises:
        MalformedLogError: If an array element is not a JSON object.
        EmptyLogError: If the payload holds no records.
    """
    candidates: list[object] = [payload] if isinstance(payload, dict) else payload

    if not candidates:
        raise EmptyLogError("The JSON file contains an empty array of records.")

    records: list[LogRecord] = []
    for index, element in enumerate(candidates):
        if not isinstance(element, dict):
            raise MalformedLogError(
                f"Record {index} is a {type(element).__name__}, but every record "
                "must be a JSON object."
            )
        records.append(
            {str(key): coerce_scalar(value) for key, value in element.items()}
        )

    return records[:MAX_LOG_ROWS], len(records) > MAX_LOG_ROWS


def _json_log(records: list[LogRecord] | tuple[LogRecord, ...], truncated: bool) -> ParsedLog:
    """Assemble a :class:`ParsedLog` from JSON-derived records.

    Args:
        records: Normalised records.
        truncated: Whether the record cap discarded anything.

    Returns:
        ParsedLog: The parsed records and their column summaries.
    """
    frozen = tuple(records)
    return ParsedLog(
        file_format=LogFileFormat.JSON,
        columns=build_columns(collect_column_names(frozen), frozen),
        rows=frozen,
        truncated=truncated,
    )


def _parse_json_array(text: str) -> ParsedLog:
    """Parse a JSON array of records.

    Args:
        text: Decoded file contents.

    Returns:
        ParsedLog: The parsed records and their column summaries.

    Raises:
        MalformedLogError: If the text is not valid JSON or not an array.
        EmptyLogError: If the array holds no records.
    """
    decoded = _decode_json(text, allow_single_record=False)
    records, truncated = _records_from_payload(decoded)
    return _json_log(records, truncated)


def _parse_json_lines(text: str) -> ParsedLog:
    """Parse a JSON Lines file of one object per line.

    A file holding exactly one object is accepted as a single-record log, which
    is why this reader tolerates braces spanning several lines.

    Args:
        text: Decoded file contents.

    Returns:
        ParsedLog: The parsed records and their column summaries.

    Raises:
        MalformedLogError: If a line is not valid JSON or not a JSON object.
        EmptyLogError: If no line holds a record.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise EmptyLogError("The file does not contain any readable content.")

    if len(lines) == 1:
        decoded = _decode_json(lines[0], allow_single_record=True)
        records, truncated = _records_from_payload(decoded)
        return _json_log(records, truncated)

    records: list[LogRecord] = []
    truncated = False
    for line_number, line in enumerate(lines, start=1):
        try:
            decoded_line = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MalformedLogError(
                f"Line {line_number} is not valid JSON: {exc.msg}."
            ) from exc
        if not isinstance(decoded_line, dict):
            raise MalformedLogError(
                f"Line {line_number} is a {type(decoded_line).__name__}, but every "
                "JSON Lines record must be a JSON object."
            )
        if len(records) >= MAX_LOG_ROWS:
            truncated = True
            break
        records.append(
            {str(key): coerce_scalar(value) for key, value in decoded_line.items()}
        )

    if not records:
        raise EmptyLogError("The file does not contain any readable content.")

    return _json_log(records, truncated)


def parse_json_log(text: str) -> ParsedLog:
    """Parse JSON log contents in either array or JSON Lines form.

    The layout is decided by inspecting the first meaningful character rather
    than by trying one reader and falling back, so the error a malformed file
    produces names the real problem instead of a speculative one.

    Args:
        text: Decoded file contents.

    Returns:
        ParsedLog: The parsed records and their column summaries.

    Raises:
        MalformedLogError: If the contents are not a JSON array of objects, a
            JSON Lines stream, or a single JSON object.
        EmptyLogError: If the file holds no records.
    """
    stripped = text.lstrip()
    if not stripped:
        raise EmptyLogError("The file does not contain any readable content.")

    if stripped.startswith("["):
        return _parse_json_array(stripped)

    if stripped.startswith("{"):
        return _parse_json_lines(stripped)

    raise MalformedLogError(
        "The file is neither a JSON array of records nor a JSON Lines stream."
    )


def parse_log_file(path: Path, file_format: LogFileFormat) -> ParsedLog:
    """Parse a stored log file from disk.

    Args:
        path: Location of the stored file.
        file_format: Layout to read it as.

    Returns:
        ParsedLog: The parsed records and their column summaries.

    Raises:
        LogParsingError: If the file cannot be decoded or does not match its
            declared layout.
        OSError: If the file cannot be read.
    """
    if file_format is LogFileFormat.CSV:
        with path.open("r", encoding=LOG_FILE_ENCODING, newline="") as handle:
            return parse_csv_log(handle)

    return parse_json_log(path.read_text(encoding=LOG_FILE_ENCODING))
