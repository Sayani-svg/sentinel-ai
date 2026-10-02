"""Pydantic schemas for the audit trail resource.

``audit_logs.action_performed`` is a plain text column, so the values that may be
written to it are declared here as a :class:`enum.StrEnum` and checked by
:mod:`app.services.report_service` rather than by the database. This is the same
pattern :mod:`app.schemas.user` uses for ``users.role`` and
:mod:`app.schemas.upload` uses for ``uploaded_logs.upload_status``.

The audit trail is shared: every slice that records a security-relevant action
writes into this one table. :class:`AuditLogRead` therefore types
``action_performed`` as :class:`str` rather than as :class:`AuditAction`, because
narrowing it would make the schema reject entries written by a slice that has its
own vocabulary. :data:`VALID_AUDIT_ACTION_VALUES` is the closed set this build
can *write*; the schema stays open so it can *read* everything the table holds.

``ip_address`` is the client address resolved by
:func:`app.middleware.rate_limit.resolve_client_key`, the same derivation the rate
limiter uses. One meaning gets one policy: if this module resolved the address
differently from the limiter, an operator comparing a rate-limit decision with an
audit entry would be comparing two different answers to the same question.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AuditAction(StrEnum):
    """Security-relevant actions this build records.

    The values are dotted ``resource.action`` strings so the column stays
    greppable as the table grows: an operator asking "who exported evidence"
    filters on ``report.generated`` without needing to know the row id.
    """

    #: A report was generated from a stored prediction.
    REPORT_GENERATED = "report.generated"
    #: A report was read back as a structured document.
    REPORT_RETRIEVED = "report.retrieved"
    #: A report was downloaded as a file.
    REPORT_DOWNLOADED = "report.downloaded"


#: Canonical audit action values, kept as plain strings so they can be compared
#: against values loaded straight out of the database.
VALID_AUDIT_ACTION_VALUES: frozenset[str] = frozenset(
    action.value for action in AuditAction
)


class AuditLogRead(BaseModel):
    """Public representation of one audit trail entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    action_performed: str = Field(description="Action recorded, as stored.")
    timestamp: datetime
    ip_address: str
