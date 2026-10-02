"""Pydantic schemas for the detection pipeline.

``predictions.severity`` and ``predictions.attack_type`` are plain text columns.
:mod:`app.models.prediction` explains why: an annotated ``Literal`` would be
resolved by SQLAlchemy into a real enum type, dragging the value set into the
table metadata and into Alembic autogenerate output. The permitted values are
therefore declared here, exactly as :mod:`app.schemas.user` declares the user
roles and :mod:`app.schemas.upload` declares the upload lifecycle, and
:mod:`app.services.prediction_service` checks them when a prediction is written.

The ``severity`` literals are the column's contract and are never renamed or
re-cased: ``Critical``, ``High``, ``Medium`` and ``Low``.

This module is free of :mod:`app.core` imports so that it stays importable
without a configured environment.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    """Operational urgency of a detected threat.

    The literal values are stored in the ``severity`` column of the
    ``predictions`` table, so they are never renamed or re-cased.
    """

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


#: Canonical severity values, kept as plain strings so they can be compared
#: against values loaded straight out of the database.
VALID_SEVERITY_VALUES: frozenset[str] = frozenset(
    severity.value for severity in Severity
)

#: Severities ordered from least to most urgent. Exposed so the service can step
#: down a classification without hard-coding an ordering of its own, and so the
#: ordering is stated once.
SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
)


class AttackType(StrEnum):
    """Threat classes the detector can report.

    The names follow the CICIDS2017 label vocabulary the project trains against.
    :attr:`UNKNOWN` is the honest answer for a log that carries no signal the
    detector recognises; it is never a synonym for benign.
    """

    BENIGN = "Benign"
    DOS = "DoS"
    DDOS = "DDoS"
    PORT_SCAN = "PortScan"
    BOT = "Bot"
    BRUTE_FORCE = "BruteForce"
    WEB_ATTACK = "WebAttack"
    INFILTRATION = "Infiltration"
    UNKNOWN = "Unknown"


#: Canonical attack type values, kept as plain strings for database comparisons.
VALID_ATTACK_TYPE_VALUES: frozenset[str] = frozenset(
    attack_type.value for attack_type in AttackType
)


class PredictionRequest(BaseModel):
    """Body of a request to classify an ingested log file.

    The upload is referenced by id rather than resubmitted: the bytes were
    already validated and stored by the upload slice, and re-accepting them
    would let two records disagree about the same file.
    """

    model_config = ConfigDict(extra="forbid")

    upload_id: int = Field(
        gt=0,
        description="Identifier of the completed upload to classify.",
    )


class PredictionRead(BaseModel):
    """Public representation of a persisted prediction."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    upload_id: int
    attack_type: AttackType
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the detector matched this attack type.",
    )
    severity: Severity
    created_at: datetime


class PredictionRunResponse(PredictionRead):
    """A stored prediction together with the size of the input behind it.

    ``records_analysed`` is reported because the parser caps the records it
    retains. A client comparing it against the upload's preview row count can
    tell that the verdict was reached from a capped view of the file.
    """

    records_analysed: int = Field(
        ge=0,
        description="Number of parsed records the classification was derived from.",
    )