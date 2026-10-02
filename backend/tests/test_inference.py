"""Tests for the detection slice: transformation, classification and persistence.

The suite is layered to match the module layout. The pipeline stages in
:mod:`app.services.prediction_service` are exercised directly, because the
interesting behaviour -- alias resolution, label precedence, rule thresholds,
confidence aggregation, severity demotion -- is invisible from the outside. The
endpoint tests then cover what a client actually observes: the response body, the
status code for each rejection, and the row left in ``predictions`` afterwards.

Three decisions are pinned by tests below, because each one is a judgement call
rather than a consequence of the code:

* A label stated by the file outranks the heuristics, even when the features
  contradict it (:func:`test_a_stated_label_outranks_contradicting_features`).
* A file whose features are readable but trip nothing is ``Benign``; a file with
  no readable feature at all is ``Unknown`` rather than a guess.
* An unrecognised label is reported as ``Unknown`` at low confidence, never
  coerced into a neighbouring class.

Every fixture writes to ``tmp_path``. The detector is deterministic, so all
assertions are exact equalities rather than tolerances.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.dependencies import get_db
from app.core.security import create_access_token, hash_password
from app.main import create_app
from app.models.prediction import Prediction
from app.models.uploaded_log import UploadedLog
from app.models.user import User
from app.schemas.prediction import (
    SEVERITY_ORDER,
    VALID_ATTACK_TYPE_VALUES,
    VALID_SEVERITY_VALUES,
    AttackType,
    PredictionRequest,
    Severity,
)
from app.schemas.upload import LogFileFormat, UploadStatus
from app.schemas.user import UserRole
from app.services.log_parser import ParsedLog, parse_log_file
from app.services.prediction_service import (
    BENIGN_CONFIDENCE,
    DOS_MEAN_BYTES_THRESHOLD,
    LABEL_CONFIDENCE,
    NO_SIGNAL_CONFIDENCE,
    UNRECOGNISED_LABEL_CONFIDENCE,
    AttackDetection,
    DetectionFeature,
    DetectionInput,
    EmptyDetectionInputError,
    InvalidAttackTypeError,
    InvalidConfidenceError,
    InvalidSeverityError,
    MalformedDetectionInputError,
    StoredLogUnavailableError,
    UnknownUploadError,
    UploadNotReadyError,
    attack_type_for_label,
    build_detection_input,
    calculate_confidence,
    create_prediction,
    determine_severity,
    detect_attack,
    map_columns,
    normalize_column_name,
    normalize_label,
    run_detection,
    summarize_features,
    to_detection_input,
    to_label,
    to_number,
)

if TYPE_CHECKING:
    from fastapi.responses import Response
    from fastapi.testclient import TestClient as TestClientType

PREDICT_URL = "/api/v1/predict"

VALID_PASSWORD = "correct-horse-battery-staple"

#: A flood: heavy volumes, few packets, no probes. The mean byte count clears
#: :data:`~app.services.prediction_service.DOS_MEAN_BYTES_THRESHOLD`.
DOS_CSV = "\n".join(
    [
        "Flow Duration,Total Length of Fwd Packets,Total Packets,Avg Packet Size",
        *(f"12000,{int(DOS_MEAN_BYTES_THRESHOLD) + 500_000},4,1200" for _ in range(6)),
        "",
    ]
)

#: A scan: many packets, every payload tiny. ``Flow Duration`` sits far above the
#: brute-force ceiling so exactly one rule can fire.
PORT_SCAN_CSV = "\n".join(
    [
        "Flow Duration,Total Length of Fwd Packets,Total Packets,Avg Packet Size",
        *("9000000,600,26,32" for _ in range(12)),
        "",
    ]
)

#: Credential guessing: a dozen rows, every flow short.
BRUTE_FORCE_CSV = "\n".join(
    [
        "Flow Duration,Total Length of Fwd Packets,Total Packets,Avg Packet Size",
        *("8000,300,2,150" for _ in range(12)),
        "",
    ]
)

#: Unremarkable traffic: small, slow, no flood, no scan, and too few rows to look
#: like credential guessing.
BENIGN_CSV = "\n".join(
    [
        "Flow Duration,Total Length of Fwd Packets,Total Packets,Avg Packet Size",
        *("40000,2048,6,340" for _ in range(4)),
        "",
    ]
)

#: Both signals at once, so corroboration can be observed.
FLOOD_AND_SCAN_CSV = "\n".join(
    [
        "Total Length of Fwd Packets,Total Packets,Avg Packet Size",
        *(
            f"{int(DOS_MEAN_BYTES_THRESHOLD) + 500_000},26,32"
            for _ in range(12)
        ),
        "",
    ]
)

#: Carries a label, and volumes that would otherwise read as a flood.
LABELLED_DOS_CSV = (
    "Label,Total Length of Fwd Packets\nDoS Hulk,9000000\nDoS Hulk,9000000\n"
)

#: States ``BENIGN`` while carrying flood volumes.
LABELLED_BENIGN_CSV = (
    "Label,Total Length of Fwd Packets\nBENIGN,9000000\nBENIGN,9000000\n"
)

#: Two different classes in one file, and no numeric feature to fall back on.
CONFLICTING_LABELS_CSV = "Label\nPortScan\nDDoS\n"

#: A label no version of this build knows.
UNKNOWN_LABEL_CSV = "Label\nWorm Propagation\nWorm Propagation\n"

#: Well-formed headers the detector recognises nothing in.
UNMAPPED_CSV = "src_ip,dst_port,protocol\n10.0.0.1,80,tcp\n10.0.0.2,443,tcp\n"


@pytest.fixture()
def csv_file(tmp_path: Path) -> Iterator[Callable[[str], Path]]:
    """Yield a factory that writes CSV text into the per-test directory.

    :func:`app.services.log_parser.parse_log_file` reads from a path, so tests
    write real files rather than mocking the reader.

    Args:
        tmp_path: Per-test temporary directory.

    Yields:
        Callable[[str], Path]: Writes the given text and returns its path.
    """
    counter = itertools.count(1)

    def write(csv_text: str) -> Path:
        """Write ``csv_text`` to a uniquely named file.

        Args:
            csv_text: Decoded CSV file contents.

        Returns:
            Path: Location of the written file.
        """
        path = tmp_path / f"case{next(counter)}.csv"
        path.write_text(csv_text, encoding="utf-8")
        return path

    yield write


def parsed(path: Path) -> ParsedLog:
    """Parse a CSV file the way the pipeline does.

    Args:
        path: Location of the CSV file.

    Returns:
        ParsedLog: The parsed file.
    """
    return parse_log_file(path, LogFileFormat.CSV)


def detect_csv(csv_file: Callable[[str], Path], csv_text: str) -> AttackDetection:
    """Classify CSV text through the whole transformation.

    Args:
        csv_file: Factory writing a file for the parser to read.
        csv_text: Decoded CSV file contents.

    Returns:
        AttackDetection: The classification.
    """
    return detect_attack(to_detection_input(parsed(csv_file(csv_text))))


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


def detection_settings(upload_dir: Path) -> Settings:
    """Build settings that resolve stored uploads under ``upload_dir``.

    Args:
        upload_dir: Directory holding the ingested files.

    Returns:
        Settings: A copy of the cached settings with the upload field changed.
    """
    return get_settings().model_copy(update={"UPLOAD_DIR": upload_dir})


def store_upload(
    db: Session,
    *,
    filename: str,
    csv_text: str,
    upload_dir: Path,
    uploaded_by: int,
    upload_status: str = UploadStatus.COMPLETED.value,
) -> UploadedLog:
    """Write a log file to disk and record it as an ingested upload.

    Args:
        db: Session to insert the row through.
        filename: Name to store the file under.
        csv_text: Decoded CSV file contents.
        upload_dir: Directory the file is written into.
        uploaded_by: Account credited with the upload.
        upload_status: Status to record on the row.

    Returns:
        UploadedLog: The persisted upload, refreshed so its id is available.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored = upload_dir / filename
    stored.write_text(csv_text, encoding="utf-8")

    upload = UploadedLog(
        filename=filename,
        file_path=str(stored),
        upload_status=upload_status,
        uploaded_by=uploaded_by,
        created_at=datetime.now(timezone.utc),
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def load_predictions(engine: Engine) -> list[Prediction]:
    """Read every ``predictions`` row in insertion order.

    Args:
        engine: Test database engine.

    Returns:
        list[Prediction]: Persisted predictions, detached from a session.
    """
    with Session(bind=engine) as db:
        return list(db.scalars(select(Prediction).order_by(Prediction.id)))


def post_predict(
    client: TestClientType,
    *,
    body: dict[str, object],
    user_id: int | None = None,
) -> Response:
    """Submit one classification request.

    Args:
        client: Client bound to an application with a test upload directory.
        body: JSON request body.
        user_id: Account to authenticate as, or ``None`` to send no credentials.

    Returns:
        Response: The client's response.
    """
    headers = {} if user_id is None else bearer(user_id)
    return client.post(PREDICT_URL, json=body, headers=headers)


@pytest.fixture()
def predict_client(tmp_path: Path, engine: Engine) -> Iterator[TestClient]:
    """Yield a client whose detection reads uploads from a temporary directory.

    Args:
        tmp_path: Per-test temporary directory.
        engine: Test database engine.

    Yields:
        TestClient: The client.
    """
    upload_dir = tmp_path / "uploads"
    settings = detection_settings(upload_dir)
    application = create_app(settings)

    def override_get_db() -> Iterator[Session]:
        """Serve route handlers from the per-test database.

        Yields:
            Session: A session bound to the test engine.
        """
        db = Session(bind=engine, expire_on_commit=False)
        try:
            yield db
        finally:
            db.close()

    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Declared value sets
# ---------------------------------------------------------------------------


def test_severity_values_are_the_column_contract() -> None:
    """The four stored severities are named exactly as the column declares."""
    assert VALID_SEVERITY_VALUES == {"Critical", "High", "Medium", "Low"}
    assert SEVERITY_ORDER == (
        Severity.LOW,
        Severity.MEDIUM,
        Severity.HIGH,
        Severity.CRITICAL,
    )


def test_attack_types_cover_the_benchmark_vocabulary() -> None:
    """Every class the project reports is a member of the declared set."""
    assert VALID_ATTACK_TYPE_VALUES == {
        "Benign",
        "DoS",
        "DDoS",
        "PortScan",
        "Bot",
        "BruteForce",
        "WebAttack",
        "Infiltration",
        "Unknown",
    }


def test_every_reported_class_carries_a_storable_severity() -> None:
    """No classification can produce a severity the column does not accept."""
    for attack_type in AttackType:
        assert determine_severity(attack_type, 0.9).value in VALID_SEVERITY_VALUES


# ---------------------------------------------------------------------------
# Name normalisation and alias resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Total Length of Fwd Packets", "totallengthoffwdpackets"),
        ("  total_packets  ", "totalpackets"),
        ("Avg. Packet Size", "avgpacketsize"),
    ],
)
def test_column_names_reduce_to_a_comparable_form(raw: str, expected: str) -> None:
    """Punctuation and spacing cannot hide an otherwise exact header."""
    assert normalize_column_name(raw) == expected


def test_benchmark_headers_map_onto_the_canonical_features() -> None:
    """The CICIDS2017 header set resolves onto every feature the rules use."""
    mapping = map_columns(
        (
            "Flow Duration",
            "Total Length of Fwd Packets",
            "Total Packets",
            "Avg Packet Size",
            "Label",
        )
    )

    assert mapping[DetectionFeature.FLOW_DURATION] == "Flow Duration"
    assert mapping[DetectionFeature.TOTAL_BYTES] == "Total Length of Fwd Packets"
    assert mapping[DetectionFeature.TOTAL_PACKETS] == "Total Packets"
    assert mapping[DetectionFeature.AVG_PACKET_SIZE] == "Avg Packet Size"
    assert mapping[DetectionFeature.LABEL] == "Label"


def test_an_exact_header_wins_over_a_containing_one() -> None:
    """``bytes`` is claimed outright rather than left to a compound alias."""
    mapping = map_columns(["Flow Bytes Transferred", "bytes"])

    assert mapping[DetectionFeature.TOTAL_BYTES] == "bytes"


def test_the_longest_containing_alias_wins() -> None:
    """A compound header is read by its most specific interpretation."""
    mapping = map_columns(["Total Length of Fwd Packets"])

    assert mapping[DetectionFeature.TOTAL_BYTES] == "Total Length of Fwd Packets"


def test_a_column_is_never_claimed_by_two_features() -> None:
    """Two headers matching one feature leave the second unmapped.

    ``Total Length of Bwd Packets`` also contains the word ``Packets``. Claiming
    it would report a byte count as a packet count, so it is left alone.
    """
    mapping = map_columns(
        ["Total Length of Fwd Packets", "Total Length of Bwd Packets"]
    )

    assert mapping == {DetectionFeature.TOTAL_BYTES: "Total Length of Fwd Packets"}


def test_unrecognised_headers_map_to_nothing() -> None:
    """A file the detector understands nothing about is still processable."""
    assert map_columns(["src_ip", "dst_port", "protocol"]) == {}


# ---------------------------------------------------------------------------
# Cell coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1024, 1024.0),
        (10.5, 10.5),
        ("2048", 2048.0),
        (" 3.5 ", 3.5),
        ("", None),
        ("n/a", None),
        (None, None),
        (True, None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_only_numeric_cells_become_measurements(
    raw: object, expected: float | None
) -> None:
    """A flag or a word is not a quantity, and cannot satisfy a threshold."""
    assert to_number(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DoS Hulk", "DoS Hulk"),
        ("  benign ", "benign"),
        (7, "7"),
        ("", None),
        ("   ", None),
        (None, None),
        ({"a": 1}, None),
    ],
)
def test_labels_are_trimmed_and_never_invented(
    raw: object, expected: str | None
) -> None:
    """Only a scalar cell can name a class; empty and structured cells cannot."""
    assert to_label(raw) == expected


# ---------------------------------------------------------------------------
# Label interpretation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("BENIGN", AttackType.BENIGN),
        ("Normal", AttackType.BENIGN),
        ("DoS Hulk", AttackType.DOS),
        ("DOS-GoldenEye", AttackType.DOS),
        ("DDoS", AttackType.DDOS),
        ("PortScan", AttackType.PORT_SCAN),
        ("PATATOR", AttackType.BRUTE_FORCE),
        ("Brute Force", AttackType.BRUTE_FORCE),
        ("SQL Injection", AttackType.WEB_ATTACK),
        ("Bot", AttackType.BOT),
    ],
)
def test_known_labels_resolve_to_their_class(label: str, expected: AttackType) -> None:
    """Benchmark spellings of a class all resolve to the same attack type."""
    assert attack_type_for_label(label) is expected


def test_a_multi_word_label_is_matched_whole_before_its_words() -> None:
    """``DoS Hulk`` is denial of service, not the sum of its words."""
    assert attack_type_for_label("dos hulk") is AttackType.DOS


def test_words_that_name_different_classes_are_not_guessed_at() -> None:
    """A label naming two classes is unknown, not whichever came first."""
    assert attack_type_for_label("PortScan DDoS") is AttackType.UNKNOWN


def test_an_unknown_label_is_unknown() -> None:
    """A class this build has never heard of is not coerced into a neighbour."""
    assert attack_type_for_label("Worm Propagation") is AttackType.UNKNOWN


def test_labels_normalise_the_way_benchmarks_vary_them() -> None:
    """Separators and casing in a label cannot change its meaning."""
    assert normalize_label("  DoS__Hulk ") == "dos hulk"


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------


def test_a_file_with_no_records_is_refused() -> None:
    """Classifying nothing is a defect in the data, not an empty result."""
    with pytest.raises(EmptyDetectionInputError):
        build_detection_input([])


def test_a_record_that_is_not_a_mapping_is_refused() -> None:
    """The transformation states what went wrong and where."""
    with pytest.raises(MalformedDetectionInputError, match="Record 1"):
        build_detection_input([{"bytes": 1}, "not-a-record"], ["bytes"])


def test_measurements_and_labels_are_split_into_their_own_slots() -> None:
    """A record keeps its class name and its numbers, in separate fields."""
    detection_input = build_detection_input(
        [{"Label": "DoS Hulk", "Total Length of Fwd Packets": "2048"}],
        ["Label", "Total Length of Fwd Packets"],
    )

    record = detection_input.records[0]
    assert record.label == "DoS Hulk"
    assert record.features[DetectionFeature.TOTAL_BYTES] == 2048.0


def test_a_cell_that_is_not_numeric_is_absent_rather_than_zero() -> None:
    """A missing measurement must not look like a measured absence."""
    detection_input = build_detection_input([{"bytes": "n/a"}], ["bytes"])

    assert detection_input.records[0].features == {}
    assert detection_input.has_signal_features is False


def test_the_column_map_is_reported_for_operators() -> None:
    """The caller can see which headers were understood."""
    detection_input = build_detection_input([{"bytes": 1}], ["bytes"])

    assert detection_input.mapped_columns == {DetectionFeature.TOTAL_BYTES: "bytes"}
    assert detection_input.row_count == 1


def test_a_parsed_file_is_transformed(csv_file: Callable[[str], Path]) -> None:
    """The parsed view is the only input the detector accepts."""
    detection_input = to_detection_input(parsed(csv_file(LABELLED_DOS_CSV)))

    assert detection_input.row_count == 2
    assert detection_input.has_label is True
    assert detection_input.labels == ("DoS Hulk",)


def test_labels_are_reported_once_each_in_first_seen_order() -> None:
    """Repeated labels collapse, and the order stays the file's."""
    detection_input = build_detection_input(
        [{"Label": "benign"}, {"Label": "DoS"}, {"Label": "benign"}],
        ["Label"],
    )

    assert detection_input.labels == ("benign", "DoS")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_features_are_summarised_over_the_records_that_carried_them() -> None:
    """A blank cell does not drag an average towards zero."""
    summary = summarize_features(
        build_detection_input(
            [{"bytes": 100, "packets": 4}, {"bytes": 300}, {"packets": 6}],
            ["bytes", "packets"],
        )
    )

    assert summary.row_count == 3
    assert summary.maxima[DetectionFeature.TOTAL_BYTES] == 300.0
    assert summary.means[DetectionFeature.TOTAL_BYTES] == 200.0
    assert summary.means[DetectionFeature.TOTAL_PACKETS] == 5.0


def test_an_empty_file_summarises_to_nothing() -> None:
    """A summary of no measurements reports no features."""
    summary = summarize_features(DetectionInput(records=()))

    assert summary.row_count == 0
    assert summary.present_features == frozenset()


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_no_matching_rule_yields_no_confidence() -> None:
    """An empty match set is zero rather than a default score."""
    assert calculate_confidence([]) == 0.0


def test_a_lone_rule_supplies_the_whole_score() -> None:
    """One matching rule contributes exactly its own weight."""
    assert calculate_confidence([0.9]) == 0.9


def test_agreement_between_rules_adds_a_bounded_bonus() -> None:
    """Corroboration helps, but cannot manufacture certainty."""
    assert calculate_confidence([0.9, 0.8]) == 0.95
    assert calculate_confidence([0.9, 0.8, 0.7, 0.7]) == 1.0


def test_a_score_can_never_leave_the_unit_interval() -> None:
    """Even two maximal rules cannot exceed certainty."""
    assert calculate_confidence([1.0, 1.0]) == 1.0


def test_a_weight_outside_the_unit_interval_is_refused() -> None:
    """The aggregation refuses to score something it cannot express."""
    with pytest.raises(InvalidConfidenceError):
        calculate_confidence([1.4])


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attack_type", "severity"),
    [
        (AttackType.DOS, Severity.CRITICAL),
        (AttackType.DDOS, Severity.CRITICAL),
        (AttackType.PORT_SCAN, Severity.HIGH),
        (AttackType.BRUTE_FORCE, Severity.HIGH),
        (AttackType.BOT, Severity.HIGH),
        (AttackType.WEB_ATTACK, Severity.MEDIUM),
        (AttackType.INFILTRATION, Severity.MEDIUM),
        (AttackType.BENIGN, Severity.LOW),
        (AttackType.UNKNOWN, Severity.LOW),
    ],
)
def test_each_class_carries_its_operational_severity(
    attack_type: AttackType, severity: Severity
) -> None:
    """A confident classification keeps the severity its class implies."""
    assert determine_severity(attack_type, 0.95) is severity


def test_an_uncertain_finding_is_stepped_down_one_severity() -> None:
    """An uncertain flood should not reach the top of an analyst's queue."""
    assert determine_severity(AttackType.DOS, 0.4) is Severity.HIGH


def test_demotion_stops_at_low() -> None:
    """Demotion has a floor and never wraps around to Critical."""
    assert determine_severity(AttackType.BENIGN, 0.1) is Severity.LOW


def test_severity_accepts_the_stored_plain_strings() -> None:
    """A value read back out of the column maps back to its class."""
    assert determine_severity("DoS", 0.9) is Severity.CRITICAL


def test_an_undeclared_attack_type_is_refused() -> None:
    """A class outside the declared set never reaches the column."""
    with pytest.raises(InvalidAttackTypeError):
        determine_severity("Catastrophe", 0.9)


def test_a_confidence_outside_the_unit_interval_is_refused() -> None:
    """A score that cannot be stored is refused before it is written."""
    with pytest.raises(InvalidConfidenceError):
        determine_severity(AttackType.DOS, 1.2)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_a_stated_label_is_taken_as_ground_truth(
    csv_file: Callable[[str], Path]
) -> None:
    """An export that names its own class is believed at near certainty."""
    detection = detect_csv(csv_file, LABELLED_DOS_CSV)

    assert detection.attack_type is AttackType.DOS
    assert detection.confidence == LABEL_CONFIDENCE
    assert detection.severity is Severity.CRITICAL
    assert detection.matched_rules == ()


def test_a_stated_label_outranks_contradicting_features(
    csv_file: Callable[[str], Path]
) -> None:
    """A file labelled BENIGN stays benign however large its volumes look."""
    detection = detect_csv(csv_file, LABELLED_BENIGN_CSV)

    assert detection.attack_type is AttackType.BENIGN
    assert detection.severity is Severity.LOW


def test_an_unrecognised_label_is_reported_as_unknown(
    csv_file: Callable[[str], Path]
) -> None:
    """A class this build cannot name is not guessed at from its features."""
    detection = detect_csv(csv_file, UNKNOWN_LABEL_CSV)

    assert detection.attack_type is AttackType.UNKNOWN
    assert detection.confidence == UNRECOGNISED_LABEL_CONFIDENCE
    assert detection.severity is Severity.LOW


def test_conflicting_labels_fall_through_to_the_features(
    csv_file: Callable[[str], Path]
) -> None:
    """A file disagreeing with itself is judged on its numbers instead."""
    detection = detect_csv(csv_file, CONFLICTING_LABELS_CSV)

    assert detection.attack_type is AttackType.UNKNOWN
    assert detection.confidence == UNRECOGNISED_LABEL_CONFIDENCE


def test_volumetric_traffic_is_detected_as_a_flood(
    csv_file: Callable[[str], Path]
) -> None:
    """Heavy mean volume is the flood signal, at high confidence."""
    detection = detect_csv(csv_file, DOS_CSV)

    assert detection.attack_type is AttackType.DOS
    assert detection.severity is Severity.CRITICAL
    assert len(detection.matched_rules) == 1


def test_many_small_probe_flows_are_detected_as_a_scan(
    csv_file: Callable[[str], Path]
) -> None:
    """High packet counts with tiny payloads are a scan, not a flood."""
    detection = detect_csv(csv_file, PORT_SCAN_CSV)

    assert detection.attack_type is AttackType.PORT_SCAN
    assert detection.severity is Severity.HIGH


def test_many_short_flows_are_detected_as_credential_guessing(
    csv_file: Callable[[str], Path]
) -> None:
    """A dozen rows sharing a short flow duration is a brute force."""
    detection = detect_csv(csv_file, BRUTE_FORCE_CSV)

    assert detection.attack_type is AttackType.BRUTE_FORCE
    assert detection.severity is Severity.HIGH


def test_agreement_between_two_rules_raises_confidence(
    csv_file: Callable[[str], Path]
) -> None:
    """Corroboration raises the score but not the class."""
    detection = detect_csv(csv_file, FLOOD_AND_SCAN_CSV)

    assert detection.attack_type is AttackType.DOS
    assert len(detection.matched_rules) == 2
    assert detection.confidence > 0.9


def test_readable_features_tripping_nothing_are_benign(
    csv_file: Callable[[str], Path]
) -> None:
    """An unremarkable file is a positive conclusion, not an absence of one."""
    detection = detect_csv(csv_file, BENIGN_CSV)

    assert detection.attack_type is AttackType.BENIGN
    assert detection.confidence == BENIGN_CONFIDENCE
    assert detection.severity is Severity.LOW


def test_a_file_with_no_readable_feature_is_unknown_rather_than_benign(
    csv_file: Callable[[str], Path]
) -> None:
    """Silence is not evidence of normality."""
    detection = detect_csv(csv_file, UNMAPPED_CSV)

    assert detection.attack_type is AttackType.UNKNOWN
    assert detection.confidence == NO_SIGNAL_CONFIDENCE
    assert detection.severity is Severity.LOW


def test_an_empty_input_cannot_be_classified() -> None:
    """The pipeline refuses nothing-to-read rather than reporting Benign."""
    with pytest.raises(EmptyDetectionInputError):
        detect_attack(DetectionInput(records=()))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_classification_is_stored_against_its_upload(
    db_session: Session, tmp_path: Path, csv_file: Callable[[str], Path]
) -> None:
    """The stored row carries the class, score, severity and timestamp."""
    user_id = seed_user(db_session.get_bind().engine, role=UserRole.ANALYST.value)
    upload = store_upload(
        db_session,
        filename="flood.csv",
        csv_text=DOS_CSV,
        upload_dir=tmp_path / "uploads",
        uploaded_by=user_id,
    )

    prediction = create_prediction(
        db_session, upload_id=upload.id, detection=detect_csv(csv_file, DOS_CSV)
    )

    assert prediction.id is not None
    assert prediction.upload_id == upload.id
    assert prediction.attack_type == "DoS"
    assert prediction.severity == "Critical"
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.created_at is not None


def test_a_severity_outside_the_column_contract_is_refused(db_session: Session) -> None:
    """The text column is checked here rather than by the database."""
    detection = AttackDetection(
        attack_type=AttackType.DOS,
        confidence=0.9,
        severity="Catastrophic",  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidSeverityError):
        create_prediction(db_session, upload_id=1, detection=detection)


def test_an_unstoreable_confidence_is_refused(db_session: Session) -> None:
    """A score the response schema would reject is refused before storage."""
    detection = AttackDetection(
        attack_type=AttackType.DOS,
        confidence=1.5,
        severity=Severity.CRITICAL,
    )

    with pytest.raises(InvalidConfidenceError):
        create_prediction(db_session, upload_id=1, detection=detection)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_the_pipeline_classifies_a_stored_file(
    db_session: Session, tmp_path: Path
) -> None:
    """A completed upload is read from disk, classified and recorded."""
    upload_dir = tmp_path / "uploads"
    user_id = seed_user(db_session.get_bind().engine, role=UserRole.ANALYST.value)
    upload = store_upload(
        db_session,
        filename="scan.csv",
        csv_text=PORT_SCAN_CSV,
        upload_dir=upload_dir,
        uploaded_by=user_id,
    )

    outcome = run_detection(
        db_session, upload_id=upload.id, settings=detection_settings(upload_dir)
    )

    assert outcome.prediction.attack_type == "PortScan"
    assert outcome.prediction.severity == "High"
    assert outcome.records_analysed == 12
    assert outcome.attack_detection.attack_type is AttackType.PORT_SCAN


def test_an_unknown_upload_is_named_in_the_error(db_session: Session) -> None:
    """The caller learns which identifier matched nothing."""
    with pytest.raises(UnknownUploadError) as caught:
        run_detection(db_session, upload_id=404, settings=get_settings())

    assert caught.value.upload_id == 404


def test_an_upload_that_is_not_completed_is_refused(
    db_session: Session, tmp_path: Path
) -> None:
    """A rejected upload keeps its row for audit but has nothing to classify."""
    upload_dir = tmp_path / "uploads"
    user_id = seed_user(db_session.get_bind().engine, role=UserRole.ANALYST.value)
    upload = store_upload(
        db_session,
        filename="rejected.csv",
        csv_text=BENIGN_CSV,
        upload_dir=upload_dir,
        uploaded_by=user_id,
        upload_status=UploadStatus.FAILED.value,
    )

    with pytest.raises(UploadNotReadyError) as caught:
        run_detection(
            db_session, upload_id=upload.id, settings=detection_settings(upload_dir)
        )

    assert caught.value.upload_status == UploadStatus.FAILED.value


def test_a_file_deleted_after_ingestion_is_reported(
    db_session: Session, tmp_path: Path
) -> None:
    """A completed row whose file has since been removed is not a crash."""
    upload_dir = tmp_path / "uploads"
    user_id = seed_user(db_session.get_bind().engine, role=UserRole.ANALYST.value)
    upload = store_upload(
        db_session,
        filename="gone.csv",
        csv_text=BENIGN_CSV,
        upload_dir=upload_dir,
        uploaded_by=user_id,
    )
    (upload_dir / "gone.csv").unlink()

    with pytest.raises(StoredLogUnavailableError):
        run_detection(
            db_session, upload_id=upload.id, settings=detection_settings(upload_dir)
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_a_classification_is_returned_as_a_created_prediction(
    predict_client: TestClient, engine: Engine, tmp_path: Path
) -> None:
    """A successful run answers 201 with the stored row and its record count."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    with Session(bind=engine) as db:
        upload = store_upload(
            db,
            filename="flood.csv",
            csv_text=DOS_CSV,
            upload_dir=tmp_path / "uploads",
            uploaded_by=user_id,
        )

    response = post_predict(
        predict_client, body={"upload_id": upload.id}, user_id=user_id
    )

    assert response.status_code == 201
    body = response.json()
    assert body["attack_type"] == "DoS"
    assert body["severity"] == "Critical"
    assert body["upload_id"] == upload.id
    assert body["records_analysed"] == 6
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["id"] > 0
    assert body["created_at"]

    stored = load_predictions(engine)
    assert len(stored) == 1
    assert stored[0].attack_type == "DoS"
    assert stored[0].severity in VALID_SEVERITY_VALUES


def test_an_admin_may_run_detection(predict_client: TestClient, engine: Engine) -> None:
    """The guard admits Admin as well as Analyst, so the request reaches the service."""
    admin_id = seed_user(engine, role=UserRole.ADMIN.value, email="root@example.com")

    response = post_predict(predict_client, body={"upload_id": 1}, user_id=admin_id)

    assert response.status_code == 404


def test_an_unknown_upload_is_reported_as_not_found(
    predict_client: TestClient, engine: Engine
) -> None:
    """A missing upload is a client error about the reference, not the file."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)

    response = post_predict(predict_client, body={"upload_id": 999}, user_id=user_id)

    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_an_upload_that_is_not_completed_is_reported_as_a_conflict(
    predict_client: TestClient, engine: Engine, tmp_path: Path
) -> None:
    """The client is told the upload is not ready rather than that it failed."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    with Session(bind=engine) as db:
        upload = store_upload(
            db,
            filename="rejected.csv",
            csv_text=BENIGN_CSV,
            upload_dir=tmp_path / "uploads",
            uploaded_by=user_id,
            upload_status=UploadStatus.FAILED.value,
        )

    response = post_predict(
        predict_client, body={"upload_id": upload.id}, user_id=user_id
    )

    assert response.status_code == 409
    assert UploadStatus.FAILED.value in response.json()["detail"]


def test_a_file_deleted_after_ingestion_is_reported_as_gone(
    predict_client: TestClient, engine: Engine, tmp_path: Path
) -> None:
    """The row exists but the bytes do not, which is a gone resource."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    upload_dir = tmp_path / "uploads"
    with Session(bind=engine) as db:
        upload = store_upload(
            db,
            filename="gone.csv",
            csv_text=BENIGN_CSV,
            upload_dir=upload_dir,
            uploaded_by=user_id,
        )
    (upload_dir / "gone.csv").unlink()

    response = post_predict(
        predict_client, body={"upload_id": upload.id}, user_id=user_id
    )

    assert response.status_code == 410


def test_a_file_the_parser_refuses_is_reported_as_unprocessable(
    predict_client: TestClient, engine: Engine, tmp_path: Path
) -> None:
    """A file with no records is the client's 422, not a server error."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)
    with Session(bind=engine) as db:
        upload = store_upload(
            db,
            filename="empty.csv",
            csv_text="",
            upload_dir=tmp_path / "uploads",
            uploaded_by=user_id,
        )

    response = post_predict(
        predict_client, body={"upload_id": upload.id}, user_id=user_id
    )

    assert response.status_code == 422
    assert load_predictions(engine) == []


def test_credentials_are_required(predict_client: TestClient) -> None:
    """An unauthenticated request is challenged, not processed."""
    response = post_predict(predict_client, body={"upload_id": 1})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_a_viewer_may_not_run_detection(
    predict_client: TestClient, engine: Engine
) -> None:
    """Detection is reserved for accounts cleared to analyse logs."""
    viewer_id = seed_user(engine, role=UserRole.VIEWER.value, email="vic@example.com")

    response = post_predict(predict_client, body={"upload_id": 1}, user_id=viewer_id)

    assert response.status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        {"upload_id": 0},
        {"upload_id": -3},
        {"upload_id": "seven"},
        {"upload_id": 1, "notes": "extra"},
        {},
    ],
)
def test_a_malformed_body_is_refused_by_the_schema(
    predict_client: TestClient, engine: Engine, body: dict[str, object]
) -> None:
    """The identifier must be a positive integer and the body nothing else."""
    user_id = seed_user(engine, role=UserRole.ANALYST.value)

    response = post_predict(predict_client, body=body, user_id=user_id)

    assert response.status_code == 422


def test_the_request_schema_refuses_unknown_fields() -> None:
    """The body is closed, so a typo is reported rather than ignored."""
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({"upload_id": 1, "upload_Id": 2})


def test_the_endpoint_is_documented_in_the_openapi_schema(
    predict_client: TestClient,
) -> None:
    """The route and its failure modes appear on the OpenAPI page."""
    schema = predict_client.get("/openapi.json").json()
    operation = schema["paths"][PREDICT_URL]["post"]

    assert operation["tags"] == ["Detection"]
    assert operation["responses"]["201"]
    for code in ("401", "403", "404", "409", "410", "415", "422"):
        assert code in operation["responses"]
