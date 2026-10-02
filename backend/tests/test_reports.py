"""Tests for the reports and audit trail slice.

The suite is layered to match the module layout, and the halves are weighted
differently on purpose.

The service is exercised directly, because the interesting behaviour is invisible
from the outside: that a report restates a stored prediction rather than
reclassifying it, that the CSV export is derived from the same document as the
JSON, that a readable-but-invalid report file is refused rather than served, and
that the audit trail writes or does not write on purpose. These tests need no
client and, for most of them, no database.

The endpoint tests then cover only what a client can observe: the response body,
the status code for each rejection, the role gate, and the rows left behind in
``reports`` and ``audit_logs``.

Two conventions carry over from the upload suite. ``REPORT_DIR`` is redirected
into ``tmp_path`` for every test, because the configured default points at the
repository root and a suite that wrote there would litter untracked files on every
run. And no test needs a live database: the fixtures build a private in-memory
SQLite database per test, and the audit primitive is additionally proved against
a stub session with no database at all.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.dependencies import get_db
from app.core.security import create_access_token, hash_password
from app.main import create_app
from app.models.audit_log import AuditLog
from app.models.prediction import Prediction
from app.models.report import Report
from app.models.uploaded_log import UploadedLog
from app.models.user import User
from app.schemas.audit import VALID_AUDIT_ACTION_VALUES, AuditAction, AuditLogRead
from app.schemas.prediction import AttackType, Severity
from app.schemas.report import (
    ReportDetail,
    ReportDocument,
    ReportEvidence,
    ReportFindings,
    ReportRequest,
)
from app.schemas.upload import UploadStatus
from app.schemas.user import UserRole
from app.services.report_service import (
    DEFAULT_RECOMMENDATIONS,
    IMMEDIATE_ACTION_SEVERITIES,
    AuditContext,
    GeneratedReport,
    InvalidAuditActionError,
    MalformedPredictionError,
    MalformedReportError,
    ReportsDisabledError,
    ReportStorageError,
    ReportUnavailableError,
    UnknownPredictionError,
    UnknownReportError,
    allocate_report_path,
    build_findings,
    build_recommendations,
    build_report_document,
    create_report,
    get_report,
    read_report_document,
    read_report_export,
    record_audit_action,
    render_report_csv,
    report_export_path,
    retrieve_report,
    write_report,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.responses import Response

REPORTS_URL = "/api/v1/reports"

VALID_PASSWORD = "correct-horse-battery-staple"

#: Address a test presents through the proxy header, so an asserted audit value is
#: pinned rather than left to whatever ``TestClient`` uses as its peer.
CLIENT_IP = "203.0.113.7"

#: A high-severity finding: the case that escalates, and that an incident review
#: actually quotes.
FLOOD_FINDING: dict[str, object] = {
    "attack_type": "DoS",
    "confidence": 0.95,
    "severity": "Critical",
}

#: A cleared finding, which must not read like an incident.
BENIGN_FINDING: dict[str, object] = {
    "attack_type": "Benign",
    "confidence": 0.7,
    "severity": "Low",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def settings_for(report_dir: Path, **overrides: object) -> Settings:
    """Build settings that store generated reports under ``report_dir``.

    Args:
        report_dir: Directory that receives generated reports.
        **overrides: Additional settings fields to override.

    Returns:
        Settings: A copy of the cached settings, so the suite never mutates the
        instance the rest of the application shares.
    """
    return get_settings().model_copy(
        update={"REPORT_DIR": report_dir, **overrides}
    )


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


def bearer(user_id: int, ip_address: str = CLIENT_IP) -> dict[str, str]:
    """Build request headers for an authenticated account behind a proxy.

    The forwarded header is sent so the address the audit trail records is the one
    this test chose, rather than the address ``TestClient`` happens to present.

    Args:
        user_id: Account id to embed as the token subject.
        ip_address: Address the request appears to come from.

    Returns:
        dict[str, str]: Authorization and forwarded-for headers.
    """
    return {
        "Authorization": f"Bearer {create_access_token(subject=str(user_id))}",
        "X-Forwarded-For": ip_address,
    }


def seed_finding(
    engine: Engine,
    *,
    filename: str = "flood.csv",
    author: int | None = None,
    **finding: object,
) -> tuple[int, int]:
    """Record an ingestion and a detection result for one finding.

    Args:
        engine: Test database engine.
        filename: Display name recorded on the ingestion.
        author: Account credited with the upload, or ``None`` to insert one.
        **finding: Prediction column values overriding the default flood finding.

    Returns:
        tuple[int, int]: The upload id and the prediction id.
    """
    uploaded_by = author if author is not None else seed_user(
        engine, role=UserRole.ANALYST.value
    )
    values: dict[str, object] = {**FLOOD_FINDING, **finding}
    with Session(bind=engine) as db:
        upload = UploadedLog(
            filename=filename,
            file_path=f"unused/{filename}",
            upload_status=UploadStatus.COMPLETED.value,
            uploaded_by=uploaded_by,
            created_at=datetime.now(timezone.utc),
        )
        db.add(upload)
        db.commit()
        db.refresh(upload)

        prediction = Prediction(
            upload_id=upload.id,
            attack_type=values["attack_type"],
            confidence=values["confidence"],
            severity=values["severity"],
            created_at=datetime.now(timezone.utc),
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return int(upload.id), int(prediction.id)


def stored_prediction(
    upload_id: int,
    *,
    attack_type: str = "DoS",
    confidence: float = 0.95,
    severity: str = "Critical",
) -> Prediction:
    """Build a transient prediction row for pure-function tests.

    The row is never added to a session, so the document assembly tests need no
    database at all.

    Args:
        upload_id: Ingested log the prediction belongs to.
        attack_type: Recorded class.
        confidence: Recorded score.
        severity: Recorded urgency.

    Returns:
        Prediction: A transient ORM object.
    """
    return Prediction(
        upload_id=upload_id,
        attack_type=attack_type,
        confidence=confidence,
        severity=severity,
        created_at=datetime.now(timezone.utc),
    )


def sample_document() -> ReportDocument:
    """Return a minimal valid report for the storage tests.

    Returns:
        ReportDocument: A report describing one critical finding.
    """
    predicted_at = datetime.now(timezone.utc)
    return ReportDocument(
        generated_at=predicted_at,
        evidence=ReportEvidence(
            prediction_id=1,
            upload_id=1,
            source_filename="flood.csv",
            attack_type=AttackType.DOS,
            confidence=0.95,
            severity=Severity.CRITICAL,
            predicted_at=predicted_at,
        ),
        findings=ReportFindings(
            summary="DoS recorded with confidence 95% against upload 1.",
            requires_immediate_action=True,
        ),
        recommendations=["Escalate now."],
    )


def load_reports(engine: Engine) -> list[Report]:
    """Read every ``reports`` row in insertion order.

    Args:
        engine: Test database engine.

    Returns:
        list[Report]: Persisted report records, detached from a session.
    """
    with Session(bind=engine) as db:
        return list(db.scalars(select(Report).order_by(Report.id)))


def load_audit_entries(engine: Engine) -> list[AuditLog]:
    """Read every ``audit_logs`` row in insertion order.

    Args:
        engine: Test database engine.

    Returns:
        list[AuditLog]: Persisted audit entries, detached from a session.
    """
    with Session(bind=engine) as db:
        return list(db.scalars(select(AuditLog).order_by(AuditLog.id)))


def report_url(report_id: int | str, *, report_format: str | None = None) -> str:
    """Build the retrieval URL for one report.

    Args:
        report_id: Identifier to address.
        report_format: Optional ``format`` query value.

    Returns:
        str: The absolute path to request.
    """
    url = f"{REPORTS_URL}/{report_id}"
    return url if report_format is None else f"{url}?format={report_format}"


def build_app(
    engine: Engine, settings: Settings, *, raise_server_errors: bool = True
) -> FastAPI:
    """Create an application wired to the test database and given settings.

    Args:
        engine: Test database engine every request should reach.
        settings: Settings for the report directory and feature flags.
        raise_server_errors: Kept for symmetry; handled by ``TestClient``.

    Returns:
        FastAPI: An application with both dependencies overridden.
    """

    def override_get_db() -> Iterator[Session]:
        """Serve each request from the per-test database.

        Yields:
            Session: A session bound to the test engine.
        """
        db = Session(bind=engine, expire_on_commit=False)
        try:
            yield db
        finally:
            db.close()

    application = create_app(settings)
    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[get_settings] = lambda: settings
    return application


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    """Return the directory generated reports are written into for a test.

    The directory is deliberately left uncreated, so the service is exercised on
    having to create it.

    Args:
        tmp_path: Per-test temporary directory.

    Returns:
        Path: A path that does not exist yet.
    """
    return tmp_path / "reports"


@pytest.fixture()
def reports_client(engine: Engine, report_dir: Path) -> Iterator[TestClient]:
    """Yield a client whose reports land in a throwaway directory.

    Args:
        engine: Test database engine.
        report_dir: Directory receiving generated reports.

    Yields:
        TestClient: The client.
    """
    application = build_app(engine, settings_for(report_dir))
    with TestClient(application) as client:
        yield client


@pytest.fixture()
def reports_disabled_client(engine: Engine, report_dir: Path) -> Iterator[TestClient]:
    """Yield a client for a deployment with report generation switched off.

    Args:
        engine: Test database engine.
        report_dir: Directory that must stay untouched.

    Yields:
        TestClient: A client whose settings have ``ENABLE_REPORTS`` off.
    """
    application = build_app(engine, settings_for(report_dir, ENABLE_REPORTS=False))
    with TestClient(application) as client:
        yield client


@pytest.fixture()
def fault_tolerant_client(engine: Engine, report_dir: Path) -> Iterator[TestClient]:
    """Yield a client that reports a server fault as a response.

    ``TestClient`` re-raises an unhandled server exception to its caller by
    default, which is the right default but makes the application's own 500
    handler unobservable. This client turns that off so the generic 500 body can
    be asserted the way a real client would see it.

    Args:
        engine: Test database engine.
        report_dir: Directory receiving generated reports.

    Yields:
        TestClient: A client that surfaces the 500 response instead of raising.
    """
    application = build_app(engine, settings_for(report_dir))
    with TestClient(application, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture()
def analyst(engine: Engine) -> dict[str, str]:
    """Return request headers for a seeded Analyst account.

    Retrieval requires authentication on every request, so the headers are
    produced once by a fixture rather than repeated at each call site.

    Args:
        engine: Test database engine.

    Returns:
        dict[str, str]: Headers for a newly inserted Analyst.
    """
    return bearer(seed_user(engine, role=UserRole.ANALYST.value))


@pytest.fixture()
def stored_report_id(
    reports_client: TestClient, engine: Engine, analyst: dict[str, str]
) -> int:
    """Generate one report through the endpoint and return its id.

    Args:
        reports_client: Client bound to a test report directory.
        engine: Test database engine.
        analyst: Headers for the Analyst that generates the report.

    Returns:
        int: The identifier of the generated report.
    """
    user_id = seed_user(engine, role=UserRole.ANALYST.value, email="bob@example.com")
    _, prediction_id = seed_finding(engine, author=user_id)
    response = generate(reports_client, prediction_id=prediction_id, user_id=user_id)
    return int(response.json()["id"])


def generate(
    client: TestClient, *, prediction_id: int, user_id: int | None = None
) -> Response:
    """Submit one report generation request.

    Args:
        client: Client bound to an application with a test report directory.
        prediction_id: Finding to document.
        user_id: Account to authenticate as, or ``None`` to send no credentials.

    Returns:
        Response: The client's response.
    """
    headers = {} if user_id is None else bearer(user_id)
    return client.post(
        REPORTS_URL, json={"prediction_id": prediction_id}, headers=headers
    )


# ---------------------------------------------------------------------------
# Declared value sets
# ---------------------------------------------------------------------------


def test_audit_actions_are_dotted_and_stable() -> None:
    """The values written to ``action_performed`` are greppable and closed."""
    assert VALID_AUDIT_ACTION_VALUES == {
        "report.generated",
        "report.retrieved",
        "report.downloaded",
    }


def test_the_audit_schema_reads_actions_written_by_other_slices() -> None:
    """The shared table is read open, so an unfamiliar action stays readable.

    ``audit_logs`` is written by every slice, so narrowing the read model to this
    build's vocabulary would make entries written elsewhere unserialisable.
    """
    entry = AuditLogRead.model_validate(
        {
            "id": 1,
            "user_id": 2,
            "action_performed": "login.succeeded",
            "timestamp": datetime.now(timezone.utc),
            "ip_address": "10.0.0.1",
        }
    )

    assert entry.action_performed == "login.succeeded"


def test_immediate_action_covers_exactly_the_escalating_severities() -> None:
    """The two views of urgency are stated once, so they cannot disagree."""
    assert IMMEDIATE_ACTION_SEVERITIES == frozenset({Severity.CRITICAL, Severity.HIGH})


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attack_type", list(AttackType), ids=[member.value for member in AttackType]
)
def test_every_class_has_guidance_registered(attack_type: AttackType) -> None:
    """No class in the enumeration falls through to the generic placeholder."""
    guidance = build_recommendations(attack_type, Severity.MEDIUM)

    assert guidance
    assert guidance != list(DEFAULT_RECOMMENDATIONS)


def test_an_urgent_finding_is_told_to_escalate_first() -> None:
    """The escalation line leads the guidance rather than trailing it."""
    guidance = build_recommendations(AttackType.DOS, Severity.CRITICAL)

    assert guidance[0].startswith("Escalate now:")
    assert "Critical" in guidance[0]


def test_a_low_severity_finding_is_not_told_to_escalate() -> None:
    """Escalation guidance is reserved for severities that justify it."""
    guidance = build_recommendations(AttackType.BENIGN, Severity.LOW)

    assert not any(line.startswith("Escalate now:") for line in guidance)


def test_guidance_accepts_the_stored_plain_strings() -> None:
    """Values read straight out of the column resolve the same way."""
    assert build_recommendations("PortScan", Severity.HIGH)[0].startswith(
        "Escalate now:"
    )


def test_guidance_refuses_a_class_this_build_does_not_know() -> None:
    """Guidance is looked up, never invented for an unrecognised class."""
    with pytest.raises(ValueError):
        build_recommendations("Catastrophe", Severity.LOW)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def test_a_finding_summary_states_what_was_found_and_how_strongly(
    engine: Engine,
) -> None:
    """The first line an analyst reads carries class, confidence and severity."""
    upload_id, _ = seed_finding(engine)

    findings = build_findings(stored_prediction(upload_id))

    assert findings.summary == (
        "DoS recorded with confidence 95% against upload "
        f"{upload_id}, at Critical severity."
    )
    assert findings.requires_immediate_action is True


def test_a_cleared_finding_does_not_ask_for_immediate_action(engine: Engine) -> None:
    """A benign report reads like a clearance, not like an incident."""
    upload_id, _ = seed_finding(engine, **BENIGN_FINDING)

    findings = build_findings(
        stored_prediction(
            upload_id,
            attack_type="Benign",
            confidence=0.7,
            severity="Low",
        )
    )

    assert findings.requires_immediate_action is False
    assert "Benign" in findings.summary


def test_a_report_copies_the_prediction_rather_than_reclassifying_it(
    engine: Engine,
) -> None:
    """The report's evidence is the stored row, restated exactly."""
    upload_id, prediction_id = seed_finding(engine, filename="flood.csv")
    generated_at = datetime.now(timezone.utc)

    with Session(bind=engine) as db:
        prediction = db.get(Prediction, prediction_id)
        assert prediction is not None
        document = build_report_document(prediction, None, generated_at=generated_at)

    assert document.evidence == ReportEvidence(
        prediction_id=prediction_id,
        upload_id=upload_id,
        source_filename=None,
        attack_type=AttackType.DOS,
        confidence=0.95,
        severity=Severity.CRITICAL,
        predicted_at=prediction.created_at,
    )
    assert document.generated_at == generated_at


def test_a_report_records_the_filename_of_its_source(engine: Engine) -> None:
    """The ingestion row supplies the display name the report quotes."""
    upload_id, prediction_id = seed_finding(engine, filename="traffic, evening.csv")
    generated_at = datetime.now(timezone.utc)

    with Session(bind=engine) as db:
        prediction = db.get(Prediction, prediction_id)
        upload = db.get(UploadedLog, upload_id)
        assert prediction is not None and upload is not None
        document = build_report_document(prediction, upload, generated_at=generated_at)

    assert document.evidence.source_filename == "traffic, evening.csv"


def test_a_report_survives_the_deletion_of_its_upload(engine: Engine) -> None:
    """A finding outlives the file it came from, so no filename is invented."""
    _, prediction_id = seed_finding(engine)
    generated_at = datetime.now(timezone.utc)

    with Session(bind=engine) as db:
        prediction = db.get(Prediction, prediction_id)
        assert prediction is not None
        document = build_report_document(prediction, None, generated_at=generated_at)

    assert document.evidence.source_filename is None
    assert document.evidence.attack_type is AttackType.DOS


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def test_the_csv_export_addresses_every_field_by_key() -> None:
    """A reader can tell which field of the JSON each row came from."""
    rows = list(csv.reader(io.StringIO(render_report_csv(sample_document()))))

    assert rows[0] == ["field", "value"]
    keys = [row[0] for row in rows[1:]]
    assert "generated_at" in keys
    assert "evidence.attack_type" in keys
    assert "evidence.severity" in keys
    assert "findings.requires_immediate_action" in keys
    assert keys.count("recommendations[0]") == 1


def test_the_csv_export_carries_the_recorded_values() -> None:
    """Values are quoted and separated correctly, so commas inside data survive."""
    document = sample_document()
    document.evidence.source_filename = "traffic, evening.csv"

    rows = dict(
        (row[0], row[1]) for row in csv.reader(io.StringIO(render_report_csv(document)))
    )

    assert rows["evidence.attack_type"] == "DoS"
    assert rows["evidence.severity"] == "Critical"
    assert rows["evidence.source_filename"] == "traffic, evening.csv"
    assert rows["findings.requires_immediate_action"] == "True"


def test_the_csv_rendering_is_deterministic() -> None:
    """The same document renders identically.

    The document is built once and rendered twice. Rendering two separately built
    documents would assert that the clock does not advance between them, which is
    not what this test is about and fails intermittently.
    """
    document = sample_document()
    rendered = render_report_csv(document)

    assert rendered == render_report_csv(document)
    assert rendered.endswith("\n")


def test_the_written_csv_is_identical_on_every_platform(tmp_path: Path) -> None:
    """Line endings in the stored file are the ones that were rendered.

    Text mode would translate ``\\n`` to the platform separator on write, so a
    report generated on Windows would store CRLF while the same document
    generated on Linux stored LF. The bytes are compared directly, and read as
    bytes, because ``read_text`` would translate the difference away again and
    hide exactly the defect this test exists to catch.
    """
    document = sample_document()
    csv_path = write_report(tmp_path / "report.json", document)

    stored = csv_path.read_bytes()

    assert b"\r\n" not in stored
    assert stored == render_report_csv(document).encode("utf-8")


def test_the_written_json_is_identical_on_every_platform(tmp_path: Path) -> None:
    """The canonical document keeps its rendered newlines too."""
    document = sample_document()
    json_path = tmp_path / "report.json"

    write_report(json_path, document)

    assert b"\r\n" not in json_path.read_bytes()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_the_report_directory_is_created_on_demand(report_dir: Path) -> None:
    """``REPORT_DIR`` is a configured path a fresh checkout does not have."""
    assert not report_dir.exists()

    path = allocate_report_path(settings_for(report_dir))

    assert report_dir.is_dir()
    assert path.parent == report_dir
    assert path.suffix == ".json"


def test_a_report_writes_a_canonical_document_and_a_csv_export(
    tmp_path: Path,
) -> None:
    """Both files come from one call, so they cannot describe different findings."""
    document = sample_document()
    json_path = tmp_path / "report.json"

    csv_path = write_report(json_path, document)

    assert json.loads(json_path.read_text(encoding="utf-8")) == document.model_dump(
        mode="json"
    )
    assert csv_path == tmp_path / "report.csv"
    assert "evidence.attack_type" in csv_path.read_text(encoding="utf-8")


def test_no_staging_file_is_left_behind_by_a_successful_write(tmp_path: Path) -> None:
    """Files are moved into place, so no partial artefact survives the call."""
    write_report(tmp_path / "report.json", sample_document())

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "report.csv",
        "report.json",
    ]


def test_an_unwritable_report_directory_is_reported(tmp_path: Path) -> None:
    """A path that cannot be created is a storage failure, not a crash."""
    blocker = tmp_path / "blocked"
    blocker.write_text("this is a file, not a directory", encoding="utf-8")

    with pytest.raises(ReportStorageError):
        allocate_report_path(settings_for(blocker / "reports"))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generating_a_report_records_it_and_writes_both_files(
    engine: Engine, report_dir: Path
) -> None:
    """The row, the document and the export all describe the same finding."""
    _, prediction_id = seed_finding(engine)

    with Session(bind=engine) as db:
        generated = create_report(
            db, prediction_id=prediction_id, settings=settings_for(report_dir)
        )

        assert generated.report.id is not None
        assert generated.report.prediction_id == prediction_id
        assert generated.report.generated_at == generated.document.generated_at
        assert generated.report.report_path == str(generated.json_path)
        assert generated.json_path.is_file()
        assert generated.csv_path.is_file()
        assert read_report_document(generated.report) == generated.document


def test_a_report_is_refused_when_no_prediction_matches(engine: Engine) -> None:
    """The caller learns which identifier matched nothing."""
    with Session(bind=engine) as db:
        with pytest.raises(UnknownPredictionError) as caught:
            create_report(db, prediction_id=987, settings=settings_for(Path("unused")))

    assert caught.value.prediction_id == 987


def test_generation_is_refused_when_reports_are_disabled(
    engine: Engine, report_dir: Path
) -> None:
    """``ENABLE_REPORTS`` is a configuration contract, honoured by the service."""
    _, prediction_id = seed_finding(engine)

    with Session(bind=engine) as db:
        with pytest.raises(ReportsDisabledError):
            create_report(
                db,
                prediction_id=prediction_id,
                settings=settings_for(report_dir, ENABLE_REPORTS=False),
            )

    assert not report_dir.exists()
    assert load_reports(engine) == []


def test_a_failed_row_write_leaves_no_files_behind(
    engine: Engine, report_dir: Path
) -> None:
    """A ``reports`` row is never left pointing at a file that was removed."""
    _, prediction_id = seed_finding(engine)

    with Session(bind=engine) as db:
        db.commit = _refuse_commits  # type: ignore[method-assign]
        with pytest.raises(ReportStorageError):
            create_report(
                db, prediction_id=prediction_id, settings=settings_for(report_dir)
            )

    assert list(report_dir.iterdir()) == []
    assert load_reports(engine) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("attack_type", "Ransomware", id="unrecognised-class"),
        pytest.param("severity", "Catastrophic", id="unrecognised-severity"),
        pytest.param("confidence", 1.5, id="confidence-outside-range"),
    ],
)
def test_a_stored_finding_this_build_cannot_quote_is_refused(
    field: str, value: object
) -> None:
    """A value outside the declared sets is refused as a typed service failure.

    The two value-set columns are plain text by design, so a row written by
    another schema version can hold something this build has no name for. The
    refusal has to arrive as a service error naming the column rather than as the
    bare ``ValueError`` the enum constructor would otherwise raise.
    """
    prediction = stored_prediction(1, **{field: value})

    with pytest.raises(MalformedPredictionError) as caught:
        build_report_document(prediction, None, generated_at=datetime.now(timezone.utc))

    assert caught.value.field == field
    assert caught.value.value == value
    assert caught.value.reason


def test_a_stored_finding_refused_by_assembly_leaves_nothing_behind(
    engine: Engine, report_dir: Path
) -> None:
    """The document is assembled before any path is allocated or file written."""
    _, prediction_id = seed_finding(engine, attack_type="Ransomware")

    with Session(bind=engine) as db:
        with pytest.raises(MalformedPredictionError) as caught:
            create_report(
                db, prediction_id=prediction_id, settings=settings_for(report_dir)
            )

    assert caught.value.prediction_id == prediction_id
    assert not report_dir.exists()
    assert load_reports(engine) == []


def _refuse_commits() -> None:
    """Refuse a commit, standing in for the ``reports`` table being unavailable.

    Raises:
        SQLAlchemyError: Always.
    """
    raise SQLAlchemyError("the reports table is unavailable")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_a_report_round_trips_through_the_stored_file(
    engine: Engine, report_dir: Path
) -> None:
    """What was generated is what comes back, validated on the way."""
    _, prediction_id = seed_finding(engine)

    with Session(bind=engine) as db:
        generated = create_report(
            db, prediction_id=prediction_id, settings=settings_for(report_dir)
        )
        db.expire_all()

        report, document = retrieve_report(db, generated.report.id)

        assert report.id == generated.report.id
        assert document == generated.document


def test_retrieval_refuses_an_unknown_id(engine: Engine) -> None:
    """A missing report is named in the error, not guessed at."""
    with Session(bind=engine) as db:
        with pytest.raises(UnknownReportError) as caught:
            get_report(db, 55)

    assert caught.value.report_id == 55


def test_a_report_whose_file_was_deleted_is_reported(
    engine: Engine, report_dir: Path
) -> None:
    """The row survives its file, and retrieval says so instead of crashing."""
    _, prediction_id = seed_finding(engine)

    with Session(bind=engine) as db:
        generated = create_report(
            db, prediction_id=prediction_id, settings=settings_for(report_dir)
        )
        Path(generated.report.report_path).unlink()

        with pytest.raises(ReportUnavailableError) as caught:
            read_report_document(generated.report)

    assert caught.value.report_id == generated.report.id


def test_a_report_file_that_is_not_json_is_refused(
    engine: Engine, report_dir: Path
) -> None:
    """A damaged artefact is reported, not served as though it were original."""
    generated = generate_and_return(engine, report_dir)
    Path(generated.report.report_path).write_text("not json at all", encoding="utf-8")

    with pytest.raises(MalformedReportError) as caught:
        read_report_document(generated.report)

    assert caught.value.reason == "it is not valid JSON"


def test_a_report_file_that_is_not_a_report_is_refused(
    engine: Engine, report_dir: Path
) -> None:
    """Valid JSON in the wrong shape is still a damaged report."""
    generated = generate_and_return(engine, report_dir)
    Path(generated.report.report_path).write_text(
        json.dumps({"generated_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8"
    )

    with pytest.raises(MalformedReportError) as caught:
        read_report_document(generated.report)

    assert "does not match the report schema" in caught.value.reason


def test_a_report_file_carrying_extra_fields_is_refused(
    engine: Engine, report_dir: Path
) -> None:
    """An unrecognised field means the file was not written by this build."""
    generated = generate_and_return(engine, report_dir)
    path = Path(generated.report.report_path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["approved_by"] = "someone"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(MalformedReportError):
        read_report_document(generated.report)


def test_a_missing_csv_export_is_reported(engine: Engine, report_dir: Path) -> None:
    """A download of an export that is not there is a gone resource."""
    generated = generate_and_return(engine, report_dir)
    generated.csv_path.unlink()

    with pytest.raises(ReportUnavailableError):
        read_report_export(generated.report)


def test_the_export_path_sits_beside_the_canonical_document(tmp_path: Path) -> None:
    """The naming rule lives in the service, so the router cannot guess wrong."""
    report = Report(
        id=1,
        prediction_id=1,
        report_path=str(tmp_path / "abc123.json"),
        generated_at=datetime.now(timezone.utc),
    )

    assert report_export_path(report) == tmp_path / "abc123.csv"


def generate_and_return(engine: Engine, report_dir: Path) -> GeneratedReport:
    """Generate one report for a flood finding and return the result.

    Args:
        engine: Test database engine.
        report_dir: Directory to receive the generated files.

    Returns:
        GeneratedReport: The detached result of :func:`create_report`.
    """
    _, prediction_id = seed_finding(engine)
    with Session(bind=engine) as db:
        return create_report(
            db, prediction_id=prediction_id, settings=settings_for(report_dir)
        )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def test_an_audit_entry_records_who_did_what_from_where(engine: Engine) -> None:
    """The row carries all four facts a later review needs."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)

    with Session(bind=engine) as db:
        entry = record_audit_action(
            db,
            user_id=user_id,
            action=AuditAction.REPORT_GENERATED,
            ip_address=CLIENT_IP,
            settings=get_settings(),
        )

        assert entry is not None
        assert entry.id is not None
        assert entry.user_id == user_id
        assert entry.action_performed == "report.generated"
        assert entry.ip_address == CLIENT_IP
        assert entry.timestamp is not None
        assert AuditLogRead.model_validate(entry).action_performed == "report.generated"


def test_an_action_outside_the_declared_set_is_refused(engine: Engine) -> None:
    """The text column is checked in code, because the database would not."""
    with Session(bind=engine) as db:
        with pytest.raises(InvalidAuditActionError):
            record_audit_action(
                db,
                user_id=1,
                action="something.unexpected",
                ip_address="10.0.0.1",
                settings=get_settings(),
            )

    assert load_audit_entries(engine) == []


def test_no_entry_is_written_when_auditing_is_disabled(engine: Engine) -> None:
    """The flag exists so an operator can run without a trail, so it is honoured."""
    with Session(bind=engine) as db:
        entry = record_audit_action(
            db,
            user_id=1,
            action=AuditAction.REPORT_GENERATED,
            ip_address="10.0.0.1",
            settings=settings_for(Path("unused"), ENABLE_AUDIT_LOGGING=False),
        )

    assert entry is None
    assert load_audit_entries(engine) == []


def test_the_audit_primitive_needs_no_database() -> None:
    """The primitive is a plain function over a session, so a stub suffices.

    Nothing in the audit layer reads, and the write is a single ``add``/``commit``,
    which is why this test needs no fixture at all.
    """
    stub = StubSession()

    entry = record_audit_action(
        stub,
        user_id=42,
        action=AuditAction.REPORT_DOWNLOADED,
        ip_address="198.51.100.9",
        settings=get_settings(),
    )

    assert entry is not None
    assert stub.committed == 1
    assert entry.action_performed == "report.downloaded"
    assert entry.ip_address == "198.51.100.9"
    assert entry.user_id == 42


def test_the_audit_primitive_can_join_a_callers_transaction() -> None:
    """``commit=False`` stages the row instead of writing it on its own.

    This is the mode :func:`create_report` uses so that a report and its audit
    entry land in one transaction.
    """
    stub = StubSession()

    entry = record_audit_action(
        stub,
        user_id=42,
        action=AuditAction.REPORT_GENERATED,
        ip_address="198.51.100.9",
        settings=get_settings(),
        commit=False,
    )

    assert entry is not None
    assert len(stub.added) == 1
    assert stub.committed == 0


def test_a_report_and_its_audit_entry_are_written_by_one_commit(
    engine: Engine, report_dir: Path
) -> None:
    """Generation commits once, not once per row.

    Two commits would let a failure between them leave a durable report with no
    record of who generated it. Counting the commits is the direct statement of
    that invariant; asserting both rows exist would not catch the regression,
    because a successful two-commit path also produces both rows.
    """
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)
    counting = CommitCountingSession(bind=engine)

    with counting as db:
        create_report(
            db,
            prediction_id=prediction_id,
            settings=settings_for(report_dir),
            audit=AuditContext(
                user_id=user_id,
                action=AuditAction.REPORT_GENERATED,
                ip_address=CLIENT_IP,
            ),
        )

    assert counting.commits == 1
    assert len(load_reports(engine)) == 1
    entries = load_audit_entries(engine)
    assert [entry.action_performed for entry in entries] == ["report.generated"]
    assert entries[0].user_id == user_id
    assert entries[0].ip_address == CLIENT_IP


def test_a_failed_commit_leaves_neither_the_report_nor_the_audit_entry(
    engine: Engine, report_dir: Path
) -> None:
    """The transaction is all-or-nothing, and the files go with it.

    Before this was one transaction the report row committed first and the audit
    row second, so a failure in between left a retrievable artefact that no audit
    trail mentioned.
    """
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)

    with Session(bind=engine) as db:
        db.commit = _refuse_commits  # type: ignore[method-assign]
        with pytest.raises(ReportStorageError):
            create_report(
                db,
                prediction_id=prediction_id,
                settings=settings_for(report_dir),
                audit=AuditContext(
                    user_id=user_id,
                    action=AuditAction.REPORT_GENERATED,
                    ip_address=CLIENT_IP,
                ),
            )

    assert load_reports(engine) == []
    assert load_audit_entries(engine) == []
    assert list(report_dir.iterdir()) == []


def test_an_action_outside_the_declared_set_is_refused_before_anything_is_written(
    engine: Engine, report_dir: Path
) -> None:
    """A rejected action costs nothing: no directory, no files, no rows.

    Validating after the document was written would leave a report on disk that
    the audit trail refused to record, which is the exact state this change
    exists to prevent.
    """
    _, prediction_id = seed_finding(engine)

    with Session(bind=engine) as db:
        with pytest.raises(InvalidAuditActionError):
            create_report(
                db,
                prediction_id=prediction_id,
                settings=settings_for(report_dir),
                audit=AuditContext(
                    user_id=1,
                    action="report.teleported",
                    ip_address=CLIENT_IP,
                ),
            )

    assert not report_dir.exists()
    assert load_reports(engine) == []
    assert load_audit_entries(engine) == []


def test_generation_without_an_audit_context_still_works(
    engine: Engine, report_dir: Path
) -> None:
    """A direct service call is not a user request, so it need not be audited."""
    _, prediction_id = seed_finding(engine)

    with Session(bind=engine) as db:
        generated = create_report(
            db, prediction_id=prediction_id, settings=settings_for(report_dir)
        )

    assert generated.report.id is not None
    assert load_audit_entries(engine) == []


class CommitCountingSession(Session):
    """Session that counts commits, so a transaction boundary can be asserted."""

    def __init__(self, **kwargs: object) -> None:
        """Start the commit count at zero.

        Args:
            **kwargs: Passed to :class:`~sqlalchemy.orm.Session`.
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.commits = 0

    def commit(self) -> None:
        """Record the commit, then perform it.

        Raises:
            Exception: Whatever the real commit raises.
        """
        self.commits += 1
        super().commit()


class StubSession:
    """Session stand-in recording the calls the audit primitive makes."""

    def __init__(self) -> None:
        """Start with nothing added and nothing committed.

        Args:
            None.
        """
        self.added: list[Any] = []
        self.committed = 0

    def add(self, instance: Any) -> None:
        """Record an added instance.

        Args:
            instance: The object added to the session.

        Returns:
            None.
        """
        self.added.append(instance)

    def commit(self) -> None:
        """Count a commit.

        Returns:
            None.
        """
        self.committed += 1

    def refresh(self, instance: Any) -> None:
        """Stand in for a refresh that assigns a primary key.

        Args:
            instance: The object being refreshed.

        Returns:
            None.
        """
        if getattr(instance, "id", None) is None:
            instance.id = 1


# ---------------------------------------------------------------------------
# Endpoint: generation
# ---------------------------------------------------------------------------


def test_generation_returns_the_stored_report_and_its_document(
    reports_client: TestClient, engine: Engine
) -> None:
    """A successful run answers 201 with everything an incident review quotes."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)

    response = generate(
        reports_client, prediction_id=prediction_id, user_id=user_id
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] > 0
    assert body["prediction_id"] == prediction_id
    assert body["generated_at"]
    assert body["document"]["evidence"]["attack_type"] == "DoS"
    assert body["document"]["evidence"]["severity"] == "Critical"
    assert body["document"]["evidence"]["source_filename"] == "flood.csv"
    assert body["document"]["evidence"]["prediction_id"] == prediction_id
    assert body["document"]["findings"]["requires_immediate_action"] is True
    assert body["document"]["recommendations"][0].startswith("Escalate now:")
    assert len(load_reports(engine)) == 1


def test_the_row_and_the_document_report_one_timestamp_shape(
    reports_client: TestClient, engine: Engine
) -> None:
    """Generation returns one instant, not the same instant in two shapes.

    The row's ``generated_at`` is a timezone-naive column while the document is
    built in memory, so an aware value would surface once with a ``Z`` and once
    without, leaving the caller unable to compare them.
    """
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)

    body = generate(
        reports_client, prediction_id=prediction_id, user_id=user_id
    ).json()

    assert body["generated_at"] == body["document"]["generated_at"]
    assert not body["generated_at"].endswith(("Z", "+00:00"))


def test_the_report_path_is_never_disclosed(
    reports_client: TestClient, engine: Engine, report_dir: Path
) -> None:
    """The on-disk location stays server-side, exactly as uploads do."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)

    response = generate(
        reports_client, prediction_id=prediction_id, user_id=user_id
    )

    assert response.status_code == 201
    assert "report_path" not in response.text
    assert str(report_dir) not in response.text


def test_generation_is_audited(reports_client: TestClient, engine: Engine) -> None:
    """Creating an artefact names the account and the address it came from."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)

    generate(reports_client, prediction_id=prediction_id, user_id=user_id)

    entries = load_audit_entries(engine)
    assert [entry.action_performed for entry in entries] == ["report.generated"]
    assert entries[0].user_id == user_id
    assert entries[0].ip_address == CLIENT_IP


def test_generation_for_an_unknown_prediction_is_reported_as_not_found(
    reports_client: TestClient, engine: Engine
) -> None:
    """A missing finding is a client error about the reference, and writes nothing."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)

    response = generate(reports_client, prediction_id=4242, user_id=user_id)

    assert response.status_code == 404
    assert "4242" in response.json()["detail"]
    assert load_reports(engine) == []
    assert load_audit_entries(engine) == []


def test_a_stored_finding_this_build_cannot_quote_is_a_server_error(
    fault_tolerant_client: TestClient, engine: Engine, report_dir: Path
) -> None:
    """A corrupt stored row is the server's fault, so it reports 500 and writes nothing.

    The finding the client named exists, so this is not a 404, and nothing about
    the request was wrong, so it is not a 422. The generic detail is returned
    because the specific reason belongs in the log, not in the response.
    """
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id, severity="Catastrophic")

    response = generate(
        fault_tolerant_client, prediction_id=prediction_id, user_id=user_id
    )

    assert response.status_code == 500
    assert "Catastrophic" not in response.text
    assert load_reports(engine) == []
    assert load_audit_entries(engine) == []
    assert not report_dir.exists()


def test_generation_is_unavailable_when_reports_are_disabled(
    reports_disabled_client: TestClient, engine: Engine, report_dir: Path
) -> None:
    """A switched-off feature reports itself as unavailable, not as a fault."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)

    response = generate(
        reports_disabled_client, prediction_id=prediction_id, user_id=user_id
    )

    assert response.status_code == 503
    assert load_reports(engine) == []
    assert load_audit_entries(engine) == []
    assert not report_dir.exists()


def test_generation_through_the_endpoint_commits_once(
    engine: Engine, report_dir: Path
) -> None:
    """The route adds no second commit of its own.

    The defect this guards against lived in the route rather than the service:
    the route generated the report and then recorded the audit entry as a
    separate call, which is two transactions and therefore a window in which a
    report exists with nothing in the trail. Counting commits across the whole
    request -- authentication included -- is what proves the route no longer
    commits separately.
    """
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    _, prediction_id = seed_finding(engine, author=user_id)
    settings = settings_for(report_dir)
    opened: list[CommitCountingSession] = []

    def override_get_db() -> Iterator[Session]:
        """Serve each request from the per-test database, counting commits.

        Yields:
            Session: A commit-counting session bound to the test engine.
        """
        db = CommitCountingSession(bind=engine, expire_on_commit=False)
        opened.append(db)
        try:
            yield db
        finally:
            db.close()

    application = create_app(settings)
    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[get_settings] = lambda: settings

    with TestClient(application) as client:
        response = generate(client, prediction_id=prediction_id, user_id=user_id)

    assert response.status_code == 201
    assert sum(session.commits for session in opened) == 1
    assert len(load_reports(engine)) == 1
    assert [e.action_performed for e in load_audit_entries(engine)] == [
        "report.generated"
    ]


# ---------------------------------------------------------------------------
# Endpoint: retrieval
# ---------------------------------------------------------------------------


def test_a_report_is_retrieved_as_the_document_that_was_stored(
    reports_client: TestClient,
    stored_report_id: int,
    report_dir: Path,
    analyst: dict[str, str],
) -> None:
    """Retrieval reads the file back, so a tampered file cannot pass unnoticed."""
    response = reports_client.get(report_url(stored_report_id), headers=analyst)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == stored_report_id
    assert body["document"]["evidence"]["attack_type"] == "DoS"
    assert body["document"]["evidence"]["prediction_id"] > 0

    stored = next(path for path in report_dir.iterdir() if path.suffix == ".json")
    assert json.loads(stored.read_text(encoding="utf-8")) == body["document"]


def test_a_report_can_be_downloaded_as_the_stored_csv_export(
    reports_client: TestClient,
    stored_report_id: int,
    report_dir: Path,
    analyst: dict[str, str],
) -> None:
    """The download is the file that was generated, not a re-rendering."""
    response = reports_client.get(
        report_url(stored_report_id, report_format="csv"), headers=analyst
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert f"sentinel-report-{stored_report_id}.csv" in response.headers[
        "content-disposition"
    ]

    export = next(path for path in report_dir.iterdir() if path.suffix == ".csv")
    assert response.content == export.read_bytes()

    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0] == ["field", "value"]
    assert ["evidence.attack_type", "DoS"] in rows


def test_retrieval_and_download_are_audited_separately(
    reports_client: TestClient,
    engine: Engine,
    stored_report_id: int,
    analyst: dict[str, str],
) -> None:
    """Reading a finding and exporting a file are different auditable acts."""
    reports_client.get(report_url(stored_report_id), headers=analyst)
    reports_client.get(
        report_url(stored_report_id, report_format="csv"), headers=analyst
    )

    actions = [entry.action_performed for entry in load_audit_entries(engine)]
    assert actions == ["report.generated", "report.retrieved", "report.downloaded"]


def test_an_unknown_report_is_reported_as_not_found(
    reports_client: TestClient, analyst: dict[str, str]
) -> None:
    """A missing report is named in the error."""
    response = reports_client.get(report_url(999), headers=analyst)

    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_a_report_whose_file_vanished_is_reported_as_gone(
    reports_client: TestClient,
    stored_report_id: int,
    report_dir: Path,
    analyst: dict[str, str],
) -> None:
    """The row exists but the artefact does not, which is a gone resource."""
    for path in report_dir.iterdir():
        path.unlink()

    response = reports_client.get(report_url(stored_report_id), headers=analyst)

    assert response.status_code == 410
    assert "no longer available" in response.json()["detail"]


def test_a_missing_csv_export_is_reported_as_gone(
    reports_client: TestClient,
    stored_report_id: int,
    report_dir: Path,
    analyst: dict[str, str],
) -> None:
    """A download of an export that was removed is also a gone resource."""
    for path in report_dir.iterdir():
        if path.suffix == ".csv":
            path.unlink()

    response = reports_client.get(
        report_url(stored_report_id, report_format="csv"), headers=analyst
    )

    assert response.status_code == 410


def test_a_tampered_report_file_is_not_served(
    reports_client: TestClient,
    stored_report_id: int,
    report_dir: Path,
    analyst: dict[str, str],
) -> None:
    """A readable file that is not a report is a server fault, not a 404."""
    stored = next(path for path in report_dir.iterdir() if path.suffix == ".json")
    stored.write_text('{"approved_by": "someone"}', encoding="utf-8")

    response = reports_client.get(report_url(stored_report_id), headers=analyst)

    assert response.status_code == 500
    assert response.json()["detail"] == "The stored report could not be read."


# ---------------------------------------------------------------------------
# Endpoint: malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"prediction_id": 0},
        {"prediction_id": -1},
        {"prediction_id": "one"},
        {"prediction_id": None},
        {"prediction_id": 1, "format": "pdf"},
        {},
    ],
)
def test_a_malformed_generation_body_is_refused_by_the_schema(
    reports_client: TestClient, engine: Engine, body: dict[str, object]
) -> None:
    """The body is closed and the identifier must be a positive integer."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)

    response = reports_client.post(REPORTS_URL, json=body, headers=bearer(user_id))

    assert response.status_code == 422
    assert load_reports(engine) == []


@pytest.mark.parametrize("report_id", [0, -2, "abc", "1.5"])
def test_a_malformed_report_id_is_refused(
    reports_client: TestClient, analyst: dict[str, str], report_id: object
) -> None:
    """The path parameter must be a positive integer."""
    assert reports_client.get(report_url(report_id), headers=analyst).status_code == 422


def test_an_unknown_response_format_is_refused(
    reports_client: TestClient, stored_report_id: int, analyst: dict[str, str]
) -> None:
    """``format`` is a closed set; an unsupported value is not a server fault."""
    response = reports_client.get(
        report_url(stored_report_id, report_format="pdf"), headers=analyst
    )

    assert response.status_code == 422


def test_the_request_schema_refuses_unknown_fields() -> None:
    """A typo in the body is reported rather than silently ignored."""
    with pytest.raises(ValidationError):
        ReportRequest.model_validate({"prediction_id": 1, "predictionId": 2})


def test_the_detail_schema_round_trips_a_stored_report() -> None:
    """The response model accepts what the service builds."""
    detail = ReportDetail(
        id=1,
        prediction_id=1,
        generated_at=datetime.now(timezone.utc),
        document=sample_document(),
    )

    payload = detail.model_dump(mode="json")

    assert payload["document"]["evidence"]["attack_type"] == "DoS"
    assert "report_path" not in payload


# ---------------------------------------------------------------------------
# Endpoint: authorization
# ---------------------------------------------------------------------------


def test_credentials_are_required_to_generate(reports_client: TestClient) -> None:
    """An unauthenticated request is challenged, not processed."""
    response = reports_client.post(REPORTS_URL, json={"prediction_id": 1})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_credentials_are_required_to_retrieve(reports_client: TestClient) -> None:
    """Retrieval is guarded exactly as generation is."""
    response = reports_client.get(report_url(1))

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_a_viewer_may_not_generate_a_report(
    reports_client: TestClient, engine: Engine
) -> None:
    """Authoring a durable artefact is reserved for analysts and admins."""
    viewer_id = seed_user(engine, role=UserRole.VIEWER.value, email="vic@example.com")
    _, prediction_id = seed_finding(engine, author=viewer_id)

    response = generate(reports_client, prediction_id=prediction_id, user_id=viewer_id)

    assert response.status_code == 403
    assert load_reports(engine) == []
    assert load_audit_entries(engine) == []


def test_an_admin_may_generate_a_report(
    reports_client: TestClient, engine: Engine
) -> None:
    """The gate admits Admin as well as Analyst."""
    admin_id = seed_user(engine, role=UserRole.ADMIN.value, email="root@example.com")
    _, prediction_id = seed_finding(engine, author=admin_id)

    response = generate(reports_client, prediction_id=prediction_id, user_id=admin_id)

    assert response.status_code == 201


def test_a_viewer_may_retrieve_a_report(
    reports_client: TestClient, engine: Engine, stored_report_id: int
) -> None:
    """Reading threat evidence is open to every authenticated role."""
    viewer_id = seed_user(engine, role=UserRole.VIEWER.value, email="vic@example.com")

    response = reports_client.get(
        report_url(stored_report_id), headers=bearer(viewer_id)
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Endpoint: documentation
# ---------------------------------------------------------------------------


def test_the_report_endpoints_are_documented(reports_client: TestClient) -> None:
    """Both routes and their failure modes appear on the OpenAPI page."""
    schema = reports_client.get("/openapi.json").json()

    generation = schema["paths"][REPORTS_URL]["post"]
    retrieval = schema["paths"][f"{REPORTS_URL}/{{report_id}}"]["get"]

    assert generation["tags"] == ["Reports"]
    assert "201" in generation["responses"]
    for code in ("401", "403", "404", "422", "503"):
        assert code in generation["responses"]

    assert retrieval["tags"] == ["Reports"]
    for code in ("200", "401", "403", "404", "410", "422"):
        assert code in retrieval["responses"]

    parameters = {item["name"]: item for item in retrieval["parameters"]}
    assert parameters["format"]["schema"]["default"] == "json"
