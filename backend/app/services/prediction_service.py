"""Detection pipeline: parsed log files become persisted predictions.

The pipeline is four stages, each a public function so a slice that needs only
one of them does not have to run the others:

1. :func:`to_detection_input` transforms the format-neutral records produced by
   :mod:`app.services.log_parser` into :class:`DetectionInput`. Column names are
   canonicalised against an alias table, so ``"Total Length of Fwd Packets"`` and
   ``total_bytes`` both become :attr:`~DetectionFeature.TOTAL_BYTES`.
2. :func:`detect_attack` decides the attack type and the confidence behind it.
3. :func:`determine_severity` maps that decision onto one of the four severities
   the ``predictions.severity`` column accepts.
4. :func:`create_prediction` writes the row, and :func:`run_detection` chains the
   four together from a stored upload.

The detector is deterministic and rule based rather than learned. This build has
no model artifact on disk and no scientific Python stack installed, and a
classifier that silently degrades to a constant when its weights are missing
would be worse than one whose rules are readable. Each rule names the features it
needs and the thresholds it applies, so swapping in a trained model later means
replacing :func:`detect_attack` and nothing else.

Labels present in the source file are treated as authoritative. A log exported
from a labelled benchmark states its own ground truth, and a heuristic that
overrode it would contradict the operator's own data. Heuristics run only when
the file carries no usable label.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Final

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logger import get_logger
from app.models.prediction import Prediction
from app.models.uploaded_log import UploadedLog
from app.schemas.prediction import (
    SEVERITY_ORDER,
    VALID_ATTACK_TYPE_VALUES,
    VALID_SEVERITY_VALUES,
    AttackType,
    Severity,
)
from app.schemas.upload import LogRecord, LogValue, UploadStatus
from app.services.log_parser import ParsedLog, collect_column_names, parse_log_file
from app.services.upload_service import resolve_log_format

logger = get_logger(__name__)

#: Confidence rounding. Four places is more than a score of this kind carries and
#: keeps the stored float readable in a response body.
CONFIDENCE_PRECISION: int = 4

#: Confidence applied when a file states a label this build recognises. Ground
#: truth in the export outranks any heuristic, so it scores near certainty.
LABEL_CONFIDENCE: Final[float] = 0.99

#: Confidence for a label the build does not recognise, and for a file carrying no
#: usable signal at all. Deliberately low: an honest "I do not know" must never
#: be mistaken for a finding.
UNRECOGNISED_LABEL_CONFIDENCE: Final[float] = 0.3
NO_SIGNAL_CONFIDENCE: Final[float] = 0.3

#: Confidence for a file whose features were all readable but which tripped no
#: rule. Higher than :data:`NO_SIGNAL_CONFIDENCE` because benign is a positive
#: conclusion from the data rather than an admission of ignorance.
BENIGN_CONFIDENCE: Final[float] = 0.7

#: Added for each independent rule that agreed with the strongest one, capped so
#: that agreement cannot manufacture certainty the thresholds do not support.
CORROBORATION_BONUS: Final[float] = 0.05
MAX_CORROBORATIONS: Final[int] = 2

#: Rule weights, each the confidence contributed when that rule is the sole
#: match. Ordered strongest claim first.
DOS_RULE_WEIGHT: Final[float] = 0.9
PORT_SCAN_RULE_WEIGHT: Final[float] = 0.8
BRUTE_FORCE_RULE_WEIGHT: Final[float] = 0.7

#: Thresholds for the heuristic rules. They are conservative defaults sized for
#: CICIDS2017-style network exports and are named so they can be tuned against a
#: trained model later rather than being buried in the predicates.
DOS_MEAN_BYTES_THRESHOLD: Final[float] = 1_000_000.0
DOS_PACKET_RATE_THRESHOLD: Final[float] = 1_000.0
PORT_SCAN_MIN_PACKETS: Final[float] = 20.0
PORT_SCAN_MAX_PACKET_SIZE: Final[float] = 64.0
BRUTE_FORCE_MIN_ROWS: Final[int] = 10
BRUTE_FORCE_MAX_FLOW_DURATION: Final[float] = 1_000_000.0

#: Below this confidence a classification is demoted one severity step, because
#: an uncertain finding should not page anyone at the top level.
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.5

#: Characters that are not alphanumeric collapse to nothing when a column name is
#: canonicalised, so ``"Total Length of Fwd Packets"`` and ``total_bytes`` agree.
_NON_ALPHANUMERIC: Final[re.Pattern[str]] = re.compile(r"[^0-9a-z]+")

#: Characters that collapse to a single space when a label is canonicalised, so
#: ``"DoS Hulk"``, ``"dos-hulk"`` and ``"DOS  HULK"`` all agree.
_NON_LABEL_CHARACTER: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


class PredictionServiceError(Exception):
    """Base class for detection pipeline failures."""


class EmptyDetectionInputError(PredictionServiceError):
    """Raised when there are no records to classify."""


class MalformedDetectionInputError(PredictionServiceError):
    """Raised when a record is not a mapping of column names to values."""


class InvalidAttackTypeError(PredictionServiceError):
    """Raised when an attack type outside the declared set is supplied."""


class InvalidSeverityError(PredictionServiceError):
    """Raised when a severity outside the declared set is supplied."""


class InvalidConfidenceError(PredictionServiceError):
    """Raised when a confidence outside ``0.0..1.0`` is supplied."""


class UnknownUploadError(PredictionServiceError):
    """Raised when no ingested log matches the requested id."""

    def __init__(self, upload_id: int) -> None:
        """Record the id that matched nothing, for logging.

        Args:
            upload_id: The identifier that was requested.
        """
        super().__init__(upload_id)
        self.upload_id = upload_id


class UploadNotReadyError(PredictionServiceError):
    """Raised when the upload exists but was not completed.

    A rejected upload keeps its row for audit purposes and has no file on disk,
    so there is nothing to classify.
    """

    def __init__(self, upload_id: int, upload_status: str) -> None:
        """Record the id and its state, for logging.

        Args:
            upload_id: The identifier that was requested.
            upload_status: The status the row actually holds.
        """
        super().__init__(upload_id, upload_status)
        self.upload_id = upload_id
        self.upload_status = upload_status


class StoredLogUnavailableError(PredictionServiceError):
    """Raised when a completed upload's file is no longer on disk."""

    def __init__(self, upload_id: int, file_path: Path) -> None:
        """Record the id and the missing location, for logging.

        Args:
            upload_id: The upload the file belonged to.
            file_path: The path that was expected to hold the log.
        """
        super().__init__(upload_id, str(file_path))
        self.upload_id = upload_id
        self.file_path = file_path


class DetectionFeature(StrEnum):
    """Canonical names for the columns the detector understands.

    A ``StrEnum`` rather than a bare string constant so the names are discoverable
    and compare equal to the plain strings a column map is keyed by.
    """

    LABEL = "label"
    FLOW_DURATION = "flow_duration"
    TOTAL_BYTES = "total_bytes"
    TOTAL_PACKETS = "total_packets"
    AVG_PACKET_SIZE = "avg_packet_size"
    PACKET_RATE = "packet_rate"


#: Canonical feature -> accepted source headers, matched only when the whole
#: header reduces to one of them. Declaration order breaks ties when two columns
#: are exact matches for the same feature.
COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    DetectionFeature.LABEL: (
        "label",
        "class",
        "classname",
        "attack",
        "attacktype",
        "category",
    ),
    DetectionFeature.FLOW_DURATION: (
        "flowduration",
        "flowdurations",
        "flowdurationus",
        "flowdurationmicroseconds",
        "duration",
    ),
    DetectionFeature.TOTAL_BYTES: (
        "bytes",
        "totalbytes",
        "totallength",
        "flowbytes",
        "totalbytestransferred",
    ),
    DetectionFeature.TOTAL_PACKETS: ("totalpackets", "packets", "packetcount"),
    DetectionFeature.AVG_PACKET_SIZE: (
        "avgpacketsize",
        "avgpacketlen",
        "meanpacketsize",
        "packetsize",
        "avglen",
    ),
    DetectionFeature.PACKET_RATE: (
        "packetrate",
        "packetspersecond",
        "flowrate",
        "flowpacketspersecond",
    ),
}

#: Canonical feature -> headers trusted when they appear *inside* a longer header
#: such as ``Total Length of Fwd Packets``. Only compound names qualify. A bare
#: generic word is deliberately excluded: matching ``packets`` inside
#: ``Total Length of Bwd Packets`` would report a byte count as a packet count,
#: and a rule reading it would be confidently wrong rather than merely absent.
COLUMN_CONTAINMENT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    DetectionFeature.LABEL: ("attacktype", "category"),
    DetectionFeature.FLOW_DURATION: (
        "flowduration",
        "flowdurations",
        "flowdurationus",
        "flowdurationmicroseconds",
    ),
    DetectionFeature.TOTAL_BYTES: ("totallength", "flowbytes", "totalbytestransferred"),
    DetectionFeature.TOTAL_PACKETS: ("totalpackets",),
    DetectionFeature.AVG_PACKET_SIZE: (
        "avgpacketsize",
        "avgpacketlen",
        "meanpacketsize",
    ),
    DetectionFeature.PACKET_RATE: ("packetspersecond", "flowpacketspersecond"),
}

#: Normalised label -> attack type. Benchmarks spell the same class several ways,
#: so the common spellings are listed rather than guessed at runtime.
LABEL_ATTACK_TYPES: Final[dict[str, AttackType]] = {
    "benign": AttackType.BENIGN,
    "normal": AttackType.BENIGN,
    "clean": AttackType.BENIGN,
    "background": AttackType.BENIGN,
    "dos": AttackType.DOS,
    "dos hulk": AttackType.DOS,
    "dosgoldeneye": AttackType.DOS,
    "dos hulk goldeneye": AttackType.DOS,
    "ddos": AttackType.DDOS,
    "distributed denial of service": AttackType.DDOS,
    "portscan": AttackType.PORT_SCAN,
    "port scan": AttackType.PORT_SCAN,
    "scan": AttackType.PORT_SCAN,
    "bot": AttackType.BOT,
    "botnet": AttackType.BOT,
    "bruteforce": AttackType.BRUTE_FORCE,
    "brute force": AttackType.BRUTE_FORCE,
    "bf": AttackType.BRUTE_FORCE,
    "patator": AttackType.BRUTE_FORCE,
    "webattack": AttackType.WEB_ATTACK,
    "web attack": AttackType.WEB_ATTACK,
    "xss": AttackType.WEB_ATTACK,
    "sqli": AttackType.WEB_ATTACK,
    "sql injection": AttackType.WEB_ATTACK,
    "infiltration": AttackType.INFILTRATION,
}

#: The severity each attack class carries before confidence is considered. Only
#: the four values the ``predictions.severity`` column accepts appear here.
SEVERITY_BY_ATTACK_TYPE: Final[dict[AttackType, Severity]] = {
    AttackType.DOS: Severity.CRITICAL,
    AttackType.DDOS: Severity.CRITICAL,
    AttackType.PORT_SCAN: Severity.HIGH,
    AttackType.BRUTE_FORCE: Severity.HIGH,
    AttackType.BOT: Severity.HIGH,
    AttackType.WEB_ATTACK: Severity.MEDIUM,
    AttackType.INFILTRATION: Severity.MEDIUM,
    AttackType.BENIGN: Severity.LOW,
    AttackType.UNKNOWN: Severity.LOW,
}


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """One parsed record reduced to the features the detector can use.

    Attributes:
        features: Canonical feature name to numeric value. A column that was
            present but held nothing numeric is absent rather than zero, so a
            missing measurement is never mistaken for a measured absence.
        label: The record's own class label, when the file carries one.
    """

    features: Mapping[str, float]
    label: str | None = None


@dataclass(frozen=True, slots=True)
class DetectionInput:
    """A parsed log file prepared for classification.

    Attributes:
        records: The records, in file order.
        mapped_columns: Canonical feature name to the source column it came
            from. Reported so an operator can see which headers were understood.
    """

    records: tuple[DetectionRecord, ...]
    mapped_columns: Mapping[str, str] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        """Return the number of records available to classify.

        Returns:
            int: Size of :attr:`records`.
        """
        return len(self.records)

    @property
    def has_label(self) -> bool:
        """Return whether any record carries a label.

        Returns:
            bool: True when at least one record has a non-empty label.
        """
        return any(record.label is not None for record in self.records)

    @property
    def has_signal_features(self) -> bool:
        """Return whether any record carried a numeric measurement.

        Returns:
            bool: True when at least one feature could be read as a number.
        """
        return any(record.features for record in self.records)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the distinct labels present, in first-seen order.

        Returns:
            tuple[str, ...]: One entry per distinct non-empty label.
        """
        seen: dict[str, None] = {}
        for record in self.records:
            if record.label is not None:
                seen.setdefault(record.label, None)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    """Aggregate view of the numeric features across every record.

    Attributes:
        row_count: Number of records summarised.
        maxima: Largest value seen per feature.
        means: Arithmetic mean per feature, over the records that carried it.
    """

    row_count: int
    maxima: Mapping[str, float]
    means: Mapping[str, float]

    @property
    def present_features(self) -> frozenset[str]:
        """Return the features that were readable in at least one record.

        Returns:
            frozenset[str]: Canonical feature names with at least one value.
        """
        return frozenset(self.means)


@dataclass(frozen=True, slots=True)
class DetectionRule:
    """One heuristic the detector applies.

    Attributes:
        attack_type: Class this rule reports when it matches.
        weight: Confidence this rule contributes when it is the strongest match.
        requires: Features that must be readable for the rule to be applied at
            all, so a rule is never evaluated on absent evidence.
        predicate: Test over the aggregate features.
        description: Human-readable statement of what the rule looks for.
    """

    attack_type: AttackType
    weight: float
    requires: frozenset[str]
    predicate: Callable[[FeatureSummary], bool]
    description: str

    def is_evaluable(self, summary: FeatureSummary) -> bool:
        """Return whether every feature this rule needs was readable.

        Args:
            summary: Aggregate features for the file.

        Returns:
            bool: True when all required features are present.
        """
        return self.requires <= summary.present_features

    def matches(self, summary: FeatureSummary) -> bool:
        """Return whether the rule's thresholds are met.

        Args:
            summary: Aggregate features for the file.

        Returns:
            bool: True when the rule fires.
        """
        return self.predicate(summary)


@dataclass(frozen=True, slots=True)
class AttackDetection:
    """The classification of one log file, before it is persisted.

    Attributes:
        attack_type: The detected class.
        confidence: How strongly it was detected, in ``0.0..1.0``.
        severity: Operational urgency, derived from the class and confidence.
        matched_rules: Descriptions of the rules that fired, empty when the
            classification came from a label rather than a heuristic.
    """

    attack_type: AttackType
    confidence: float
    severity: Severity
    matched_rules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectionOutcome:
    """A stored prediction together with the size of the input behind it.

    Attributes:
        prediction: The persisted row.
        records_analysed: Number of records the classification was derived from.
        attack_detection: The classification, retained for logging and callers
            that want the matched rules.
    """

    prediction: Prediction
    records_analysed: int
    attack_detection: AttackDetection


def normalize_column_name(name: str) -> str:
    """Reduce a source header to its canonical comparable form.

    Args:
        name: Column name exactly as the file spelled it.

    Returns:
        str: Lowercase letters and digits only, ready to match an alias.
    """
    return _NON_ALPHANUMERIC.sub("", name.strip().lower())


def normalize_label(label: str) -> str:
    """Reduce a class label to its canonical comparable form.

    Args:
        label: Label text as the file spelled it.

    Returns:
        str: Lowercase words separated by single spaces.
    """
    return _NON_LABEL_CHARACTER.sub(" ", label.strip().lower()).strip()


def map_columns(column_names: Sequence[str]) -> dict[str, str]:
    """Match source headers against the detector's canonical features.

    Resolution runs in two passes. Exact matches come first, so an unambiguous
    header such as ``bytes`` is never left to a longer compound interpretation.
    Containment matches follow, ordered longest alias first and then in the
    file's own column order, so the most specific reading of ``Total Length of
    Fwd Packets`` wins over a looser one later in the header row. Each column is
    claimed by at most one feature and each feature by at most one column.

    Args:
        column_names: Every column present in the file.

    Returns:
        dict[str, str]: Canonical feature name to the source column it maps to.
    """
    normalised = {name: normalize_column_name(name) for name in column_names}
    mapping: dict[str, str] = {}
    claimed: set[str] = set()

    for feature, aliases in COLUMN_ALIASES.items():
        if feature in mapping:
            continue
        for name in column_names:
            if name not in claimed and normalised[name] in aliases:
                mapping[feature] = name
                claimed.add(name)
                break

    containment = sorted(
        (
            (-len(alias), feature, position)
            for feature, aliases in COLUMN_CONTAINMENT_ALIASES.items()
            for position, name in enumerate(column_names)
            for alias in aliases
            if alias in normalised[name]
        )
    )
    for _specificity, feature, position in containment:
        name = column_names[position]
        if feature in mapping or name in claimed:
            continue
        mapping[feature] = name
        claimed.add(name)

    return mapping


def to_number(value: LogValue) -> float | None:
    """Interpret one raw cell as a measurement.

    Booleans are rejected deliberately: ``True`` is a flag, not a quantity, and
    treating it as ``1.0`` would let a status column satisfy a byte threshold.

    Args:
        value: Cell value as read from the file.

    Returns:
        float | None: The measurement, or ``None`` when the cell is not numeric.
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    return None


def to_label(value: LogValue) -> str | None:
    """Interpret one raw cell as a class label.

    Args:
        value: Cell value as read from the file.

    Returns:
        str | None: The trimmed label, or ``None`` when the cell is empty.
    """
    if value is None or isinstance(value, (dict, list)):
        return None

    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def attack_type_for_label(label: str) -> AttackType:
    """Return the attack type a label denotes.

    The whole label is matched first so that ``DoS Hulk`` resolves to
    :attr:`~AttackType.DOS`. Only when that fails are the individual words
    consulted, and only when they agree unanimously: ``PortScan DDoS`` names two
    classes and is therefore
    :attr:`~AttackType.UNKNOWN` rather than a guess.

    Args:
        label: Label text as the file spelled it.

    Returns:
        AttackType: The class named by the label, or
            :attr:`~AttackType.UNKNOWN`.
    """
    key = normalize_label(label)
    if not key:
        return AttackType.UNKNOWN

    direct = LABEL_ATTACK_TYPES.get(key)
    if direct is not None:
        return direct

    found = {
        LABEL_ATTACK_TYPES[word]
        for word in key.split()
        if word in LABEL_ATTACK_TYPES
    }
    if len(found) == 1:
        return next(iter(found))

    return AttackType.UNKNOWN


def build_detection_input(
    records: Sequence[LogRecord],
    column_names: Sequence[str] | None = None,
) -> DetectionInput:
    """Transform parsed records into the detector's input shape.

    Args:
        records: Parsed records, as :func:`app.services.log_parser.to_records`
            produces them.
        column_names: Column names to map, or ``None`` to take them from the
            records themselves.

    Returns:
        DetectionInput: The records reduced to recognised features.

    Raises:
        EmptyDetectionInputError: If ``records`` is empty. Classifying nothing
            is a defect in the caller's data, not an empty result.
        MalformedDetectionInputError: If an entry is not a mapping.
    """
    if not records:
        raise EmptyDetectionInputError("There are no records to classify.")

    names = tuple(column_names) if column_names is not None else collect_column_names(
        tuple(records)
    )
    mapping = map_columns(names)

    prepared: list[DetectionRecord] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MalformedDetectionInputError(
                f"Record {index} is a {type(record).__name__}, but every record "
                "must be a mapping of column names to values."
            )
        features: dict[str, float] = {}
        label: str | None = None
        for feature, column in mapping.items():
            value = record.get(column)
            if feature == DetectionFeature.LABEL:
                label = to_label(value)
                continue
            number = to_number(value)
            if number is not None:
                features[feature] = number
        prepared.append(DetectionRecord(features=features, label=label))

    return DetectionInput(records=tuple(prepared), mapped_columns=mapping)


def to_detection_input(parsed: ParsedLog) -> DetectionInput:
    """Transform a parsed log file into the detector's input shape.

    Args:
        parsed: A parsed log file.

    Returns:
        DetectionInput: The records reduced to recognised features.

    Raises:
        EmptyDetectionInputError: If the file yielded no records.
    """
    return build_detection_input(parsed.to_records(), parsed.column_names)


def summarize_features(detection_input: DetectionInput) -> FeatureSummary:
    """Aggregate the numeric features across every record.

    Args:
        detection_input: The prepared records.

    Returns:
        FeatureSummary: Per-feature maxima and means, over the records that
            carried each feature.
    """
    maxima: dict[str, float] = {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for record in detection_input.records:
        for feature, value in record.features.items():
            maxima[feature] = max(maxima.get(feature, value), value)
            totals[feature] = totals.get(feature, 0.0) + value
            counts[feature] = counts.get(feature, 0) + 1

    means = {feature: totals[feature] / counts[feature] for feature in totals}

    return FeatureSummary(
        row_count=detection_input.row_count,
        maxima=maxima,
        means=means,
    )


def calculate_confidence(matched_weights: Iterable[float]) -> float:
    """Turn the weights of the rules that fired into one confidence score.

    The strongest rule sets the base score and each additional agreeing rule adds
    a small bonus, capped so that agreement cannot manufacture certainty the
    thresholds do not support.

    Args:
        matched_weights: Weight of every rule that matched.

    Returns:
        float: A score in ``0.0..1.0``; ``0.0`` when nothing matched.

    Raises:
        InvalidConfidenceError: If a weight is outside ``0.0..1.0``.
    """
    weights = list(matched_weights)
    if not weights:
        return 0.0

    for weight in weights:
        if not 0.0 <= weight <= 1.0:
            raise InvalidConfidenceError(
                f"Rule weight {weight} is outside the range 0.0 to 1.0."
            )

    corroborations = min(len(weights) - 1, MAX_CORROBORATIONS)
    score = max(weights) + CORROBORATION_BONUS * corroborations
    return round(min(1.0, score), CONFIDENCE_PRECISION)


def determine_severity(attack_type: AttackType | str, confidence: float) -> Severity:
    """Map a classification onto one of the four stored severities.

    A classification the detector is not confident about is demoted one step: an
    uncertain finding should not reach the top of an analyst's queue, and
    :attr:`~Severity.LOW` is the floor rather than a wrap.

    Args:
        attack_type: The detected class.
        confidence: How strongly it was detected, in ``0.0..1.0``.

    Returns:
        Severity: One of ``Critical``, ``High``, ``Medium`` or ``Low``.

    Raises:
        InvalidAttackTypeError: If the attack type is outside the declared set.
        InvalidConfidenceError: If the confidence is outside ``0.0..1.0``.
    """
    if isinstance(attack_type, AttackType):
        resolved = attack_type
    elif attack_type in VALID_ATTACK_TYPE_VALUES:
        resolved = AttackType(attack_type)
    else:
        raise InvalidAttackTypeError(
            f"{attack_type!r} is not one of: "
            f"{', '.join(sorted(VALID_ATTACK_TYPE_VALUES))}."
        )

    if not 0.0 <= confidence <= 1.0:
        raise InvalidConfidenceError(
            f"Confidence {confidence} is outside the range 0.0 to 1.0."
        )

    base = SEVERITY_BY_ATTACK_TYPE[resolved]
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return base

    index = SEVERITY_ORDER.index(base)
    return SEVERITY_ORDER[max(0, index - 1)]


def _looks_like_flood(summary: FeatureSummary) -> bool:
    """Return whether the file describes volumetric traffic.

    Args:
        summary: Aggregate features for the file.

    Returns:
        bool: True when mean volume or peak rate reaches flood thresholds.
    """
    mean_bytes = summary.means.get(DetectionFeature.TOTAL_BYTES)
    peak_rate = summary.maxima.get(DetectionFeature.PACKET_RATE)
    return (mean_bytes is not None and mean_bytes >= DOS_MEAN_BYTES_THRESHOLD) or (
        peak_rate is not None and peak_rate >= DOS_PACKET_RATE_THRESHOLD
    )


def _looks_like_scan(summary: FeatureSummary) -> bool:
    """Return whether the file describes many small probe flows.

    Args:
        summary: Aggregate features for the file.

    Returns:
        bool: True when packet counts are high and payloads stay small.
    """
    peak_packets = summary.maxima.get(DetectionFeature.TOTAL_PACKETS)
    if peak_packets is None or peak_packets < PORT_SCAN_MIN_PACKETS:
        return False

    peak_size = summary.maxima.get(DetectionFeature.AVG_PACKET_SIZE)
    return peak_size is not None and peak_size <= PORT_SCAN_MAX_PACKET_SIZE


def _looks_like_guessing(summary: FeatureSummary) -> bool:
    """Return whether the file describes many short repeated flows.

    Args:
        summary: Aggregate features for the file.

    Returns:
        bool: True when enough rows share a flow duration below the ceiling.
    """
    peak_duration = summary.maxima.get(DetectionFeature.FLOW_DURATION)
    if peak_duration is None:
        return False

    return (
        summary.row_count >= BRUTE_FORCE_MIN_ROWS
        and peak_duration <= BRUTE_FORCE_MAX_FLOW_DURATION
    )


#: Ordered strongest claim first, which is also the tie-break order when two
#: rules match with equal weight.
DETECTION_RULES: Final[tuple[DetectionRule, ...]] = (
    DetectionRule(
        attack_type=AttackType.DOS,
        weight=DOS_RULE_WEIGHT,
        requires=frozenset({DetectionFeature.TOTAL_BYTES}),
        predicate=_looks_like_flood,
        description="Mean volume or peak packet rate reaches flood thresholds.",
    ),
    DetectionRule(
        attack_type=AttackType.PORT_SCAN,
        weight=PORT_SCAN_RULE_WEIGHT,
        requires=frozenset(
            {DetectionFeature.TOTAL_PACKETS, DetectionFeature.AVG_PACKET_SIZE}
        ),
        predicate=_looks_like_scan,
        description="High packet counts with consistently small payloads.",
    ),
    DetectionRule(
        attack_type=AttackType.BRUTE_FORCE,
        weight=BRUTE_FORCE_RULE_WEIGHT,
        requires=frozenset({DetectionFeature.FLOW_DURATION}),
        predicate=_looks_like_guessing,
        description="Many rows sharing a short flow duration.",
    ),
)


def detect_attack(detection_input: DetectionInput) -> AttackDetection:
    """Classify a prepared log file.

    A label stated by the file wins outright when every labelled record agrees,
    because the export states its own ground truth. Otherwise the heuristic rules
    are applied to the aggregate features; when none fire but the features were
    readable the file is benign, and when no feature was readable at all the
    answer is :attr:`~AttackType.UNKNOWN` rather than a guess.

    Args:
        detection_input: The prepared records.

    Returns:
        AttackDetection: The class, its confidence and its severity.

    Raises:
        EmptyDetectionInputError: If there are no records.
    """
    if detection_input.row_count == 0:
        raise EmptyDetectionInputError("There are no records to classify.")

    labels = detection_input.labels
    if labels:
        verdicts = {attack_type_for_label(label) for label in labels}
        if len(verdicts) == 1:
            verdict = next(iter(verdicts))
            confidence = (
                UNRECOGNISED_LABEL_CONFIDENCE
                if verdict is AttackType.UNKNOWN
                else LABEL_CONFIDENCE
            )
            return AttackDetection(
                attack_type=verdict,
                confidence=confidence,
                severity=determine_severity(verdict, confidence),
            )

    if not detection_input.has_signal_features:
        confidence = UNRECOGNISED_LABEL_CONFIDENCE if labels else NO_SIGNAL_CONFIDENCE
        return AttackDetection(
            attack_type=AttackType.UNKNOWN,
            confidence=confidence,
            severity=determine_severity(AttackType.UNKNOWN, confidence),
        )

    summary = summarize_features(detection_input)
    matched = [
        rule
        for rule in DETECTION_RULES
        if rule.is_evaluable(summary) and rule.matches(summary)
    ]

    if matched:
        strongest = max(matched, key=lambda rule: rule.weight)
        confidence = calculate_confidence(rule.weight for rule in matched)
        return AttackDetection(
            attack_type=strongest.attack_type,
            confidence=confidence,
            severity=determine_severity(strongest.attack_type, confidence),
            matched_rules=tuple(rule.description for rule in matched),
        )

    return AttackDetection(
        attack_type=AttackType.BENIGN,
        confidence=BENIGN_CONFIDENCE,
        severity=determine_severity(AttackType.BENIGN, BENIGN_CONFIDENCE),
    )


def create_prediction(
    db: Session,
    *,
    upload_id: int,
    detection: AttackDetection,
) -> Prediction:
    """Persist a classification against the upload it was derived from.

    The column is plain text, so the enum membership is checked here rather than
    left to the database: a value outside the declared set would otherwise be
    stored and later fail to serialise.

    Args:
        db: Active database session.
        upload_id: Identifier of the upload the prediction belongs to.
        detection: The classification to store.

    Returns:
        Prediction: The persisted row, refreshed so its generated id is
            available.

    Raises:
        InvalidSeverityError: If the severity is outside the declared set.
        InvalidConfidenceError: If the confidence is outside ``0.0..1.0``.
    """
    severity = detection.severity
    if severity not in VALID_SEVERITY_VALUES:
        raise InvalidSeverityError(
            f"{severity!r} is not one of: {', '.join(sorted(VALID_SEVERITY_VALUES))}."
        )

    if not 0.0 <= detection.confidence <= 1.0:
        raise InvalidConfidenceError(
            f"Confidence {detection.confidence} is outside the range 0.0 to 1.0."
        )

    prediction = Prediction(
        upload_id=upload_id,
        attack_type=detection.attack_type.value,
        confidence=detection.confidence,
        severity=severity.value,
        created_at=datetime.now(timezone.utc),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    logger.info(
        "Recorded prediction %d for upload %d: %s at confidence %.4f (%s).",
        prediction.id,
        upload_id,
        prediction.attack_type,
        prediction.confidence,
        prediction.severity,
    )
    return prediction


def load_ready_upload(db: Session, upload_id: int) -> UploadedLog:
    """Load an ingested log that can be classified.

    Args:
        db: Active database session.
        upload_id: Identifier of the upload to load.

    Returns:
        UploadedLog: The completed upload.

    Raises:
        UnknownUploadError: If no row matches the id.
        UploadNotReadyError: If the row is still processing or was rejected.
    """
    upload = db.get(UploadedLog, upload_id)
    if upload is None:
        raise UnknownUploadError(upload_id)

    if upload.upload_status != UploadStatus.COMPLETED.value:
        raise UploadNotReadyError(upload_id, upload.upload_status)

    return upload


def parse_stored_log(upload: UploadedLog, settings: Settings) -> ParsedLog:
    """Re-read and parse the file stored for a completed upload.

    Args:
        upload: The completed upload.
        settings: Application settings naming the permitted extensions.

    Returns:
        ParsedLog: The parsed file.

    Raises:
        StoredLogUnavailableError: If the file is no longer on disk.
        UnsupportedFileTypeError: If the configured extensions no longer permit
            the stored file's type.
    """
    stored = Path(upload.file_path)
    if not stored.is_file():
        raise StoredLogUnavailableError(upload.id, stored)

    return parse_log_file(stored, resolve_log_format(stored.name, settings))


def run_detection(
    db: Session, *, upload_id: int, settings: Settings
) -> DetectionOutcome:
    """Classify an ingested log file and persist the outcome.

    Args:
        db: Active database session.
        upload_id: Identifier of the completed upload to classify.
        settings: Application settings naming the permitted extensions.

    Returns:
        DetectionOutcome: The stored prediction, the number of records it was
            derived from, and the classification that produced it.

    Raises:
        UnknownUploadError: If no ingested log matches the id.
        UploadNotReadyError: If the upload was rejected or is still processing.
        StoredLogUnavailableError: If the stored file is gone.
        EmptyDetectionInputError: If the stored file yielded no records.
        InvalidSeverityError: If the classification carries an undeclared
            severity.
    """
    upload = load_ready_upload(db, upload_id)
    parsed = parse_stored_log(upload, settings)
    detection_input = to_detection_input(parsed)
    detection = detect_attack(detection_input)
    prediction = create_prediction(db, upload_id=upload_id, detection=detection)

    logger.info(
        "Analysed upload %d: %d record(s) classified as %s at confidence %.4f.",
        upload_id,
        detection_input.row_count,
        detection.attack_type.value,
        detection.confidence,
    )

    return DetectionOutcome(
        prediction=prediction,
        records_analysed=detection_input.row_count,
        attack_detection=detection,
    )