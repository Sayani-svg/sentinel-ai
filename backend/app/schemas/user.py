"""Pydantic schemas for the user resource.

This module is intentionally free of :mod:`app.core` imports so that
:mod:`app.core.dependencies` may depend on it without creating an import cycle.
Policy that requires settings (password length, bcrypt input limits) lives in
:mod:`app.schemas.auth` instead.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class UserRole(StrEnum):
    """Access roles recognised by the authorization layer.

    The literal values must match the ``role`` column of the ``users`` table
    exactly, so they are never renamed or re-cased.
    """

    VIEWER = "Viewer"
    ANALYST = "Analyst"
    ADMIN = "Admin"


#: Canonical role values, kept as plain strings so they can be compared against
#: values loaded straight out of the database.
VALID_ROLE_VALUES: frozenset[str] = frozenset(role.value for role in UserRole)

#: Conservative syntactic email check. The RFC 5322 grammar cannot be expressed
#: as a regular expression, so this validates the shape of an address rather
# than proving that it is deliverable.
EMAIL_PATTERN: str = (
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+$"
)

MAX_EMAIL_LENGTH: int = 255

#: Email field that trims surrounding whitespace and lower-cases the address, so
#: that ``Alice@Example.com`` and ``alice@example.com`` resolve to one account.
NormalizedEmail = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        pattern=EMAIL_PATTERN,
        max_length=MAX_EMAIL_LENGTH,
    ),
]


class UserRead(BaseModel):
    """Public representation of a user account.

    ``password_hash`` is deliberately absent so that no route can leak it by
    returning this model.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: NormalizedEmail
    role: UserRole
    created_at: datetime
