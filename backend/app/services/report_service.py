"""Report generation, retrieval, and the audit trail both of them write to.

A report is derived data. :func:`create_report` reads one stored
``predictions`` row, restates it as a :class:`~app.schemas.report.ReportDocument`,
and writes that document to :data:`~app.core.config.Settings.REPORT_DIR`. Nothing
is recomputed: the attack type, confidence and severity are copied exactly as
recorded, so regenerating a report after the detection rules change cannot rewrite
an earlier conclusion. The remediation guidance is the only derived content, and it
is a pure function of the recorded class and severity.

**One artefact, one source of truth.** The file named by ``reports.report_path``
is JSON, and it is the only file that is ever read back. :func:`write_report`
writes a sibling ``.csv`` export in the same call, from the same in-memory
document, so the two cannot drift; the CSV is served verbatim by the download
route rather than re-rendered. Storing two independently generated formats would
have meant two artefacts that could disagree about the same finding, and a
disagreement between a PDF and a CSV about what was detected is exactly the kind
of problem nobody notices until an incident review.

**Writes are atomic.** The document is written to a staging file and moved into
place, mirroring :func:`app.services.upload_service.write_upload`: a report that
is half-written must never be retrievable as though it were complete.

**The report and its audit entry commit together.** :func:`create_report` stages
the ``audit_logs`` row on the same session as the ``reports`` row and writes both
with a single commit, so a failure cannot leave a durable artefact with no record
of who asked for it. :func:`record_audit_action` still commits on its own by
default, which is what an action that stands alone -- reading a report -- needs.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logger import get_logger
from app.models.audit_log import AuditLog
from app.models.prediction import Prediction
from app.models.report import Report
from app.models.uploaded_log import UploadedLog
from app.schemas.audit import VALID_AUDIT_ACTION_VALUES, AuditAction
from app.schemas.prediction import AttackType, Severity
from app.schemas.report import ReportDocument, ReportEvidence, ReportFindings

logger = get_logger(__name__)

#: Encoding used for every file this module writes and reads. UTF-8 rather than
#: the platform default, so a report generated on one host is byte-identical to
#: one generated on another.
REPORT_ENCODING: Final[str] = "utf-8"

#: Suffixes for the two files written per report. The JSON suffix is what
#: ``reports.report_path`` points at; the CSV sits beside it.
REPORT_JSON_SUFFIX: Final[str] = ".json"
REPORT_CSV_SUFFIX: Final[str] = ".csv"

#: Severities that justify paging someone now. Stated once so the finding summary
#: and the escalation note in the recommendations cannot disagree.
IMMEDIATE_ACTION_SEVERITIES: Final[frozenset[Severity]] = frozenset(
    {Severity.CRITICAL, Severity.HIGH}
)

#: Remediation guidance per attack class, most urgent action first. Kept as a
#: static table rather than generated prose: guidance an analyst cannot verify is
#: worse than none, and this way every claim in a report can be checked against a
#: single source.
RECOMMENDATIONS_BY_ATTACK_TYPE: Final[dict[AttackType, tuple[str, ...]]] = {
    AttackType.DOS: (
        "Apply or verify rate limiting on the affected endpoint or uplink.",
        "Identify the source addresses contributing the bulk of the volume and "
        "block or challenge them at the edge.",
        "Confirm upstream capacity headroom, since a flood that stays up is an "
        "availability incident rather than a blocked attack.",
    ),
    AttackType.DDOS: (
        "Contact the upstream provider and request scrubbing at the edge.",
        "Confirm any provider mitigation actually took effect before standing "
        "down; the attack is usually still in flight when it is reported.",
        "Lower edge rate limits and review origin-agnostic exposure until the "
        "provider confirms mitigation.",
    ),
    AttackType.PORT_SCAN: (
        "Close the probed ports or restrict them to known peers at the firewall.",
        "Review the scanning source addresses for prior reconnaissance across "
        "other assets.",
        "Alert on scan volume as an early indicator of a follow-on exploit.",
    ),
    AttackType.BRUTE_FORCE: (
        "Enforce account lockout or exponential backoff on the targeted service.",
        "Require multi-factor authentication for the affected accounts.",
        "Review authentication logs for successful logins from the same sources, "
        "which would mean the attempt eventually succeeded.",
    ),
    AttackType.BOT: (
        "Identify and remove the infected hosts from the network segment.",
        "Rotate credentials and secrets reachable from the affected hosts.",
        "Block the command-and-control indicators on the egress path.",
    ),
    AttackType.WEB_ATTACK: (
        "Inspect the web server access and error logs for exploitation attempts "
        "against the targeted path.",
        "Recheck input validation and output encoding on the affected "
        "application, and confirm a WAF rule covers the pattern.",
        "Review the application's database layer for unexpected queries that "
        "would indicate the attempt succeeded.",
    ),
    AttackType.INFILTRATION: (
        "Isolate the affected hosts and preserve volatile evidence before "
        "reimaging.",
        "Hunt laterally from the affected hosts for persistence mechanisms and "
        "further compromised assets.",
        "Review privileged access on the segment for credentials an attacker "
        "would have needed.",
    ),
    AttackType.BENIGN: (
        "No action is required. Retain the report as evidence that the traffic "
        "was assessed and cleared.",
    ),
    AttackType.UNKNOWN: (
        "Review the source log manually: the detector could not classify this "
        "traffic from the features it recognised.",
        "Confirm the log carries the columns the detector needs, then re-run "
        "detection.",
        "Treat the finding as unassessed rather than benign until a human has "
        "classified it.",
    ),
}

#: Guidance for a class this build has no specific entry for. Every class in
#: :class:`~app.schemas.prediction.AttackType` is covered above, so this only
#: applies if a class is added without extending the table.
DEFAULT_RECOMMENDATIONS: Final[tuple[str, ...]] = (
    "No specific guidance is registered for this attack class. Review the "
    "prediction manually.",
)


def utcnow() -> datetime:
    """Return the current UTC time in the form this application stores.

    Every timestamp column in the schema is a timezone-naive ``DateTime``, so an
    aware value would be written with its offset discarded and come back naive. A
    report stamps the same instant into both the ``reports`` row and the document,
    and generation returns both in one payload, so an aware value would show the
    caller one timestamp twice in two different shapes: once with a ``Z`` from the
    document and once without from the row.

    Naive UTC is therefore the convention here, matching every other slice. It is
    the value written, not a truncation: the digits are UTC either way, and only
    the redundant offset is dropped.

    Returns:
        datetime: The current UTC time, without a timezone attached.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ReportServiceError(Exception):
    """Base class for report generation and retrieval failures."""


class ReportsDisabledError(ReportServiceError):
    """Raised when report generation is switched off by configuration."""

    def __init__(self) -> None:
        """Record that the refusal is a configuration decision.

        Args:
            None.
        """
        super().__init__("Report generation is disabled by configuration.")


class UnknownPredictionError(ReportServiceError):
    """Raised when no stored prediction matches the requested id."""

    def __init__(self, prediction_id: int) -> None:
        """Record the id that matched nothing, for logging.

        Args:
            prediction_id: The identifier that was requested.
        """
        super().__init__(prediction_id)
        self.prediction_id = prediction_id


class UnknownReportError(ReportServiceError):
    """Raised when no stored report matches the requested id."""

    def __init__(self, report_id: int) -> None:
        """Record the id that matched nothing, for logging.

        Args:
            report_id: The identifier that was requested.
        """
        super().__init__(report_id)
        self.report_id = report_id


class ReportStorageError(ReportServiceError):
    """Raised when a report cannot be written to the report directory."""


class ReportUnavailableError(ReportServiceError):
    """Raised when a stored report's file is missing or unreadable."""

    def __init__(self, report_id: int, report_path: Path) -> None:
        """Record the id and the location that could not be read.

        Args:
            report_id: The report the path belongs to.
            report_path: The path that could not be read.
        """
        super().__init__(report_id, str(report_path))
        self.report_id = report_id
        self.report_path = report_path


class MalformedReportError(ReportServiceError):
    """Raised when a stored report file is readable but is not a valid report."""

    def __init__(self, report_id: int, report_path: Path, reason: str) -> None:
        """Record the id, the location and why the file was refused.

        Args:
            report_id: The report the file belongs to.
            report_path: The path that was read.
            reason: Why the contents could not be validated.
        """
        super().__init__(report_id, str(report_path), reason)
        self.report_id = report_id
        self.report_path = report_path
        self.reason = reason


class MalformedPredictionError(ReportServiceError):
    """Raised when a stored prediction cannot be documented by this build.

    ``predictions.attack_type``, ``predictions.severity`` and
    ``predictions.confidence`` are plain columns with no constraint of their own,
    so a row written by another schema version can name a class or a severity
    this build does not recognise, or carry a confidence outside the range a
    report is allowed to quote. Left alone those surface as a bare
    :class:`ValueError` or :class:`pydantic.ValidationError` from deep inside
    assembly, which says nothing about which record is at fault. This is the
    report slice's counterpart to :class:`MalformedReportError`: the stored
    artefact does not match what this build can represent.
    """

    def __init__(
        self, prediction_id: int, field: str, value: object, reason: str
    ) -> None:
        """Record which prediction, which column and why it could not be used.

        Args:
            prediction_id: The prediction that could not be documented.
            field: Column of the ``predictions`` row that carried the value.
            value: The value that was found, kept for the log message.
            reason: Why the value cannot be represented in a report.
        """
        super().__init__(prediction_id, field, str(value), reason)
        self.prediction_id = prediction_id
        self.field = field
        self.value = value
        self.reason = reason


class InvalidAuditActionError(ReportServiceError):
    """Raised when an audit action outside the declared set is recorded."""


@dataclass(frozen=True, slots=True)
class AuditContext:
    """Who performed a security-relevant action, and where the request came from.

    Bundled into one value rather than passed as three separate arguments on
    purpose. Three independent parameters would let a caller supply some of them
    and omit the rest, and a partially specified context is precisely the state
    that atomicity exists to prevent: an action recorded with no account, or an
    account recorded with no action.

    Attributes:
        user_id: Account that performed the action.
        action: The action to record, as an enum member or stored string.
        ip_address: Client address the action came from.
    """

    user_id: int
    action: AuditAction | str
    ip_address: str


@dataclass(frozen=True, slots=True)
class GeneratedReport:
    """Outcome of generating a report.

    Attributes:
        report: The persisted ``reports`` row, marked generated.
        document: The document written to disk and referenced by the row.
        json_path: Location of the canonical JSON document.
        csv_path: Location of the derived CSV export.
    """

    report: Report
    document: ReportDocument
    json_path: Path
    csv_path: Path


def build_recommendations(
    attack_type: AttackType | str, severity: Severity | str
) -> list[str]:
    """Return the remediation guidance for a finding.

    Args:
        attack_type: Recorded attack class, as an enum member or stored string.
        severity: Recorded severity, as an enum member or stored string.

    Returns:
        list[str]: Guidance lines, most urgent first. The escalation line comes
        first when the severity justifies immediate action.

    Raises:
        ValueError: If either argument names a class or severity outside the
            declared sets. Raised rather than defaulted, because guidance for a
            class this build does not recognise would be invented rather than
            looked up.
    """
    resolved_type = (
        attack_type if isinstance(attack_type, AttackType) else AttackType(attack_type)
    )
    resolved_severity = (
        severity if isinstance(severity, Severity) else Severity(severity)
    )

    guidance = list(
        RECOMMENDATIONS_BY_ATTACK_TYPE.get(resolved_type, DEFAULT_RECOMMENDATIONS)
    )
    if resolved_severity in IMMEDIATE_ACTION_SEVERITIES:
        guidance.insert(
            0,
            f"Escalate now: the finding is recorded at {resolved_severity.value} "
            "severity.",
        )
    return guidance


def _resolve_stored_value(
    prediction: Prediction, field: str, enum_type: type[AttackType] | type[Severity]
) -> AttackType | Severity:
    """Read one plain-text column of a stored prediction as its declared type.

    The two columns are text by design, so this is the point where the
    application checks them. It refuses rather than falls back: inventing a
    report for a finding whose recorded class cannot be read would state
    something the database does not say.

    Args:
        prediction: The persisted prediction row.
        field: Name of the column to read.
        enum_type: Declared type the value is expected to belong to.

    Returns:
        AttackType | Severity: The resolved enum member.

    Raises:
        MalformedPredictionError: If the stored value is outside the declared set.
    """
    raw = getattr(prediction, field)
    try:
        return enum_type(raw)
    except ValueError as exc:
        permitted = ", ".join(sorted(member.value for member in enum_type))
        raise MalformedPredictionError(
            prediction.id,
            field,
            raw,
            f"{raw!r} is not a value this build recognises. Recorded values: "
            f"{permitted}.",
        ) from exc


def build_findings(prediction: Prediction) -> ReportFindings:
    """Summarise a stored prediction in the sentence an analyst reads first.

    Args:
        prediction: The persisted prediction row.

    Returns:
        ReportFindings: A one-sentence summary and whether the severity calls
        for immediate action.

    Raises:
        MalformedPredictionError: If the stored severity is outside the declared
            set.
    """
    severity = _resolve_stored_value(prediction, "severity", Severity)
    return ReportFindings(
        summary=(
            f"{prediction.attack_type} recorded with confidence "
            f"{prediction.confidence:.0%} against upload {prediction.upload_id}, "
            f"at {severity.value} severity."
        ),
        requires_immediate_action=severity in IMMEDIATE_ACTION_SEVERITIES,
    )


def build_report_document(
    prediction: Prediction,
    upload: UploadedLog | None,
    *,
    generated_at: datetime,
) -> ReportDocument:
    """Assemble the report document for a stored prediction.

    Args:
        prediction: The persisted prediction row.
        upload: The ingested log the prediction names, or ``None`` when that row
            has since been deleted. The finding outlives the file it came from,
            so the report is still generated and simply records no filename.
        generated_at: Timestamp to stamp the report with. Passed in rather than
            read from the clock so the document and the ``reports`` row can be
            given the same value.

    Returns:
        ReportDocument: The complete report.

    Raises:
        MalformedPredictionError: If the stored finding carries an attack type,
            a severity or a confidence this build cannot quote.
    """
    attack_type = _resolve_stored_value(prediction, "attack_type", AttackType)
    severity = _resolve_stored_value(prediction, "severity", Severity)

    try:
        evidence = ReportEvidence(
            prediction_id=prediction.id,
            upload_id=prediction.upload_id,
            source_filename=upload.filename if upload is not None else None,
            attack_type=attack_type,
            confidence=prediction.confidence,
            severity=severity,
            predicted_at=prediction.created_at,
        )
    except ValidationError as exc:
        # The only value left that a stored row can put outside the schema is
        # confidence, which ReportEvidence bounds to 0.0..1.0.
        raise MalformedPredictionError(
            prediction.id,
            "confidence",
            prediction.confidence,
            "the recorded confidence cannot be quoted in a report "
            f"({exc.error_count()} problem(s))",
        ) from exc

    return ReportDocument(
        generated_at=generated_at,
        evidence=evidence,
        findings=build_findings(prediction),
        recommendations=build_recommendations(attack_type, severity),
    )


def render_report_csv(document: ReportDocument) -> str:
    """Render a report document as a two-column CSV export.

    The CSV is a flattening, not a second document: every row is one field of the
    JSON, addressed by a dotted or indexed key, so a reader can reconstruct which
    field a value came from. Line endings are fixed so the rendering is
    byte-identical on every platform.

    Args:
        document: The report to render.

    Returns:
        str: The rendered CSV, terminated by a trailing newline.
    """
    rows: list[tuple[str, str]] = [("field", "value")]
    rows.append(("generated_at", document.generated_at.isoformat()))
    for key, value in document.evidence.model_dump().items():
        rows.append((f"evidence.{key}", "" if value is None else str(value)))
    for key, value in document.findings.model_dump().items():
        rows.append((f"findings.{key}", str(value)))
    for index, recommendation in enumerate(document.recommendations):
        rows.append((f"recommendations[{index}]", recommendation))

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def allocate_report_path(settings: Settings) -> Path:
    """Reserve the on-disk location for a report.

    The directory is created on demand because ``REPORT_DIR`` is a configured
    path that a fresh checkout does not have. The name is generated rather than
    derived from client input, so no request can steer a write outside the
    directory.

    Args:
        settings: Application settings naming the report directory.

    Returns:
        Path: An unused absolute path inside ``REPORT_DIR``.

    Raises:
        ReportStorageError: If the report directory cannot be created.
    """
    try:
        settings.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReportStorageError(
            f"The report directory {settings.REPORT_DIR} is not writable."
        ) from exc

    return settings.REPORT_DIR / f"{uuid4().hex}{REPORT_JSON_SUFFIX}"


def _write_atomically(destination: Path, payload: str) -> None:
    """Write text to a path via a staging file, so readers never see a partial file.

    The payload is encoded and written as bytes rather than through ``write_text``,
    because text mode translates ``\\n`` to the platform separator: on Windows the
    stored file would carry CRLF while the same report generated on Linux carried
    LF, so two hosts would disagree about the bytes of the same artefact. Encoding
    here also pins the encoding to the module constant.

    Args:
        destination: Final path for the file.
        payload: Text to write.

    Raises:
        ReportStorageError: If the bytes cannot be written.
    """
    staging = destination.with_name(f".partial-{uuid4().hex}{destination.suffix}")
    try:
        staging.write_bytes(payload.encode(REPORT_ENCODING))
        staging.replace(destination)
    except OSError as exc:
        staging.unlink(missing_ok=True)
        raise ReportStorageError(
            f"The report could not be stored at {destination}: {exc}"
        ) from exc


def write_report(json_path: Path, document: ReportDocument) -> Path:
    """Write a report document and its derived CSV export.

    Both files are produced from the same in-memory document in one call, so the
    CSV cannot describe a different finding from the JSON. If either write fails
    the other is removed: leaving a CSV with no report behind it would be an
    export of a document that does not exist.

    Args:
        json_path: Destination for the canonical JSON document. The stored
            ``reports.report_path`` points at this file.
        document: The report to write.

    Returns:
        Path: Location of the CSV export, which sits beside the JSON document.

    Raises:
        ReportStorageError: If either file cannot be written.
    """
    csv_path = json_path.with_suffix(REPORT_CSV_SUFFIX)
    _write_atomically(
        json_path,
        json.dumps(document.model_dump(mode="json"), indent=2, sort_keys=True),
    )
    try:
        _write_atomically(csv_path, render_report_csv(document))
    except ReportStorageError:
        json_path.unlink(missing_ok=True)
        raise
    return csv_path


def report_export_path(report: Report) -> Path:
    """Return the CSV export that sits beside a report's canonical document.

    The naming rule lives here rather than in the router so that the location a
    download reads from is decided in the same place that decides where the file
    is written. A router that rebuilt the name from ``report_path`` would be
    guessing at a convention, and would guess wrong the day the suffix changed.

    Args:
        report: The persisted ``reports`` row.

    Returns:
        Path: Location of the CSV export for this report.
    """
    return Path(report.report_path).with_suffix(REPORT_CSV_SUFFIX)


def read_report_export(report: Report) -> Path:
    """Locate a report's CSV export, refusing when it is not readable.

    Args:
        report: The persisted ``reports`` row.

    Returns:
        Path: Location of the CSV export.

    Raises:
        ReportUnavailableError: If the export is missing from disk.
    """
    path = report_export_path(report)
    if not path.is_file():
        raise ReportUnavailableError(report.id, path)
    return path


def read_report_document(report: Report) -> ReportDocument:
    """Read a stored report document back and validate it.

    Reading validates rather than trusting. A report file is the artefact an
    incident review quotes, so a file that no longer matches the schema is
    reported as damaged instead of being served as though it were the original.

    Args:
        report: The persisted ``reports`` row.

    Returns:
        ReportDocument: The validated document.

    Raises:
        ReportUnavailableError: If the file is missing or unreadable.
        MalformedReportError: If the file is readable but is not a valid report.
    """
    path = Path(report.report_path)
    try:
        payload = path.read_text(encoding=REPORT_ENCODING)
    except OSError as exc:
        raise ReportUnavailableError(report.id, path) from exc

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MalformedReportError(report.id, path, "it is not valid JSON") from exc

    try:
        return ReportDocument.model_validate(decoded)
    except ValidationError as exc:
        raise MalformedReportError(
            report.id,
            path,
            f"it does not match the report schema ({exc.error_count()} problem(s))",
        ) from exc


def create_report(
    db: Session,
    *,
    prediction_id: int,
    settings: Settings,
    audit: AuditContext | None = None,
) -> GeneratedReport:
    """Generate a report for a stored prediction and record it.

    The row is committed only after both files are on disk, and is removed again
    if the commit fails, so a failed generation cannot leave a ``reports`` row
    pointing at a file that was never written.

    **The report and its audit entry share one transaction.** When ``audit`` is
    supplied, the ``audit_logs`` row is staged on the same session and both rows
    are written by a single :meth:`~sqlalchemy.orm.Session.commit`. Generating a
    report and disclosing that it happened are one security action, and two
    commits would let a failure between them leave a durable, quotable artefact
    with no record of who asked for it -- the one gap an audit trail exists to
    close. Either both rows exist or neither does.

    The action is validated *before* any path is allocated, so a caller passing an
    action outside the declared set cannot leave files behind either.

    Args:
        db: Active database session.
        prediction_id: Identifier of the prediction to document.
        settings: Application settings, including the report directory and
            whether report generation is enabled.
        audit: Who is generating the report, for the audit entry. When ``None`` no
            audit row is staged, which is what a direct service call that is not
            itself a user request wants. Every route should pass one.

    Returns:
        GeneratedReport: The persisted row, the document and both file locations.

    Raises:
        ReportsDisabledError: If ``ENABLE_REPORTS`` is off.
        UnknownPredictionError: If no stored prediction matches the id.
        MalformedPredictionError: If the prediction names a class, a severity or
            a confidence this build cannot quote.
        InvalidAuditActionError: If ``audit`` names an action outside the declared
            set.
        ReportStorageError: If the files cannot be written, or the rows cannot be
            committed.
    """
    if not settings.ENABLE_REPORTS:
        logger.warning("Refused to generate a report: reporting is disabled.")
        raise ReportsDisabledError

    if audit is not None and audit.action not in VALID_AUDIT_ACTION_VALUES:
        # Raised before any path is allocated, so a rejected action writes nothing
        # at all rather than leaving files with no record of who asked for them.
        permitted = ", ".join(sorted(VALID_AUDIT_ACTION_VALUES))
        raise InvalidAuditActionError(
            f"{audit.action!r} is not a known audit action. "
            f"Permitted actions: {permitted}."
        )

    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise UnknownPredictionError(prediction_id)

    upload = db.get(UploadedLog, prediction.upload_id)
    generated_at = utcnow()
    document = build_report_document(prediction, upload, generated_at=generated_at)

    json_path = allocate_report_path(settings)
    csv_path = write_report(json_path, document)

    report = Report(
        prediction_id=prediction.id,
        report_path=str(json_path),
        generated_at=generated_at,
    )
    try:
        db.add(report)
        if audit is not None:
            record_audit_action(
                db,
                user_id=audit.user_id,
                action=audit.action,
                ip_address=audit.ip_address,
                settings=settings,
                commit=False,
            )
        db.commit()
        db.refresh(report)
    except SQLAlchemyError as exc:
        db.rollback()
        json_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise ReportStorageError(
            f"The report record could not be stored: {exc}"
        ) from exc

    logger.info(
        "Generated report %d for prediction %d (%s at %s severity).",
        report.id,
        prediction.id,
        prediction.attack_type,
        prediction.severity,
    )
    return GeneratedReport(
        report=report,
        document=document,
        json_path=json_path,
        csv_path=csv_path,
    )


def get_report(db: Session, report_id: int) -> Report:
    """Load a stored report record.

    Args:
        db: Active database session.
        report_id: Identifier of the report to load.

    Returns:
        Report: The persisted ``reports`` row.

    Raises:
        UnknownReportError: If no stored report matches the id.
    """
    report = db.get(Report, report_id)
    if report is None:
        raise UnknownReportError(report_id)
    return report


def retrieve_report(db: Session, report_id: int) -> tuple[Report, ReportDocument]:
    """Load a report record together with the document it points at.

    Args:
        db: Active database session.
        report_id: Identifier of the report to load.

    Returns:
        tuple[Report, ReportDocument]: The row and the validated document.

    Raises:
        UnknownReportError: If no stored report matches the id.
        ReportUnavailableError: If the stored file is missing or unreadable.
        MalformedReportError: If the stored file is not a valid report.
    """
    report = get_report(db, report_id)
    return report, read_report_document(report)


def record_audit_action(
    db: Session,
    *,
    user_id: int,
    action: AuditAction | str,
    ip_address: str,
    settings: Settings,
    commit: bool = True,
) -> AuditLog | None:
    """Record one security-relevant action in the audit trail.

    This is the reusable primitive for every slice that needs an audit entry; the
    reports slice is simply its first caller.

    ``commit=True`` is the right default for an action that stands on its own,
    such as reading a report: the entry is committed immediately so it survives
    whatever the caller does next. Pass ``commit=False`` when the entry must land
    in the *same* transaction as the work it describes, which is how
    :func:`create_report` keeps a generated report and its audit entry from
    existing independently of one another.

    The write is skipped, rather than failed, when ``ENABLE_AUDIT_LOGGING`` is
    off. That is a deliberate difference from the failure mode described above:
    the flag exists to let an operator run without an audit trail, whereas a
    database error means the trail was meant to be written and could not be.

    Args:
        db: Active database session.
        user_id: Account that performed the action.
        action: The action recorded, as an enum member or stored string.
        ip_address: Client address the action came from.
        settings: Application settings, including whether auditing is enabled.
        commit: Whether to commit and refresh here. When ``False`` the row is
            only staged on the session and the caller owns the transaction, so
            the entry is returned without a database-assigned id.

    Returns:
        AuditLog | None: The persisted entry, or ``None`` when auditing is
        disabled by configuration. With ``commit=False`` the entry is staged but
        not yet persisted.

    Raises:
        InvalidAuditActionError: If the action is outside the declared set. It is
            checked here because ``action_performed`` is a plain text column that
            the database would otherwise accept anything into.
    """
    if not settings.ENABLE_AUDIT_LOGGING:
        logger.info(
            "Audit logging is disabled; not recording %s by account %d.",
            action,
            user_id,
        )
        return None

    if action not in VALID_AUDIT_ACTION_VALUES:
        permitted = ", ".join(sorted(VALID_AUDIT_ACTION_VALUES))
        raise InvalidAuditActionError(
            f"{action!r} is not a known audit action. Permitted actions: {permitted}."
        )

    entry = AuditLog(
        user_id=user_id,
        action_performed=AuditAction(action).value,
        timestamp=utcnow(),
        ip_address=ip_address,
    )
    db.add(entry)

    if not commit:
        # Staged only. No refresh: the row has no id until the caller's commit.
        return entry

    db.commit()
    db.refresh(entry)

    logger.info(
        "Recorded audit action %s by account %d from %s.",
        entry.action_performed,
        user_id,
        ip_address,
    )
    return entry
