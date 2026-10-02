"""Pydantic schemas for the security report resource.

A report is derived data: it restates a stored ``predictions`` row in the shape a
human reads during an incident. The ``reports`` table holds only a pointer and a
timestamp -- ``prediction_id``, ``report_path`` and ``generated_at`` -- so the
substance of a report lives in the file named by ``report_path``.

Two consequences shape this module.

First, :class:`ReportDocument` is the contract for that file. It is declared
separately from :class:`ReportRead` because the row and the document are
validated for different reasons: the row describes storage, the document is the
artefact itself. Retrieval re-reads the document from disk and validates it, so a
truncated or hand-edited file is reported rather than served as if it were the
original.

Second, :class:`ReportRead` omits ``report_path``. It is a server-side absolute
path, and disclosing it would leak the deployment's directory layout to any
authenticated caller. This mirrors how
:class:`~app.schemas.upload.UploadedLogRead` omits ``file_path`` and how
:class:`~app.schemas.user.UserRead` omits ``password_hash``: the column exists,
but no route can return it. Clients reach the contents through the retrieval
endpoint instead.

``attack_type`` and ``severity`` are declared by :mod:`app.schemas.prediction` and
are never restated here, so a report cannot disagree with the prediction it
describes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.prediction import AttackType, Severity


class ReportEvidence(BaseModel):
    """The facts a report is built from, copied from a stored prediction.

    Every field here is read from ``predictions`` (or from the ``uploaded_logs``
    row it names) at generation time. Nothing is recomputed: a report restates a
    finding, it does not reclassify one, so regenerating a report after the
    detection rules change cannot silently rewrite an earlier conclusion.
    """

    model_config = ConfigDict(extra="forbid")

    prediction_id: int = Field(
        gt=0, description="Prediction this report documents."
    )
    upload_id: int = Field(
        gt=0, description="Ingested log file the prediction was derived from."
    )
    source_filename: str | None = Field(
        default=None,
        description=(
            "Display name of the ingested file. Absent when the ``uploaded_logs`` "
            "row has since been deleted, since the prediction outlives it."
        ),
    )
    attack_type: AttackType = Field(description="Class recorded on the prediction.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the prediction recorded this class.",
    )
    severity: Severity = Field(
        description="Operational urgency recorded on the prediction."
    )
    predicted_at: datetime = Field(description="When the prediction was recorded.")


class ReportFindings(BaseModel):
    """The conclusion an analyst reads first, in prose rather than as columns."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        min_length=1,
        description="One sentence stating what was found and how strongly.",
    )
    requires_immediate_action: bool = Field(
        description=(
            "True for the severities that justify paging someone now. Derived from "
            "the recorded severity rather than stated independently."
        )
    )


class ReportDocument(BaseModel):
    """The canonical on-disk report, and the contract for reading one back.

    ``extra="forbid"`` is load bearing. A file carrying fields this build does not
    recognise was written by something else or hand-edited, and serving it
    unvalidated would present a tampered artefact as a stored one.
    """

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(
        description=(
            "When the report was generated. Equal to the ``generated_at`` column "
            "of the ``reports`` row, so the file and the record cannot disagree."
        )
    )
    evidence: ReportEvidence = Field(description="Facts copied from the prediction.")
    findings: ReportFindings = Field(description="The conclusion drawn from them.")
    recommendations: list[str] = Field(
        description=(
            "Remediation guidance for this attack class. Ordered most urgent "
            "first; may be empty when the class carries no specific guidance."
        )
    )


class ReportRequest(BaseModel):
    """Body of a request to generate a report for a stored prediction.

    The prediction is referenced by id rather than resubmitted: the finding
    already exists, and accepting a restatement of it would let a client request
    a report for evidence the server never classified.
    """

    model_config = ConfigDict(extra="forbid")

    prediction_id: int = Field(
        gt=0,
        description="Identifier of the prediction to document.",
    )


class ReportRead(BaseModel):
    """Public representation of a stored report record.

    ``report_path`` is intentionally absent so that no route can disclose the
    server-side location where generated reports are stored.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    prediction_id: int
    generated_at: datetime


class ReportDetail(ReportRead):
    """A stored report record together with the document it points at.

    Returned by retrieval rather than by generation, so a client can confirm the
    stored artefact round-trips through validation before relying on it.
    """

    document: ReportDocument = Field(
        description="The report as read back from disk and validated."
    )
