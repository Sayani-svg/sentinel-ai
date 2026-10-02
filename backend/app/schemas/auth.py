"""Pydantic schemas for the authentication endpoints.

Request bodies are configured with ``extra="forbid"`` so that unexpected fields
are rejected instead of ignored. That is what stops a caller from smuggling a
``role`` into a registration payload: role assignment is an administrative
action, not something a new account may choose for itself.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.core.config import get_settings
from app.core.security import BCRYPT_MAX_PASSWORD_BYTES
from app.schemas.user import NormalizedEmail

MAX_NAME_LENGTH: int = 255


class AuthRequest(BaseModel):
    """Base class for authentication request bodies."""

    model_config = ConfigDict(extra="forbid")


def _validate_encoded_length(password: str) -> None:
    """Reject passwords whose UTF-8 encoding exceeds the bcrypt input limit.

    Args:
        password: The candidate plaintext password.

    Raises:
        ValueError: If the encoded password is longer than bcrypt accepts.
    """
    encoded_length = len(password.encode("utf-8"))
    if encoded_length > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must not exceed {BCRYPT_MAX_PASSWORD_BYTES} bytes "
            "when UTF-8 encoded."
        )


class UserRegisterRequest(AuthRequest):
    """Payload for self-service account registration."""

    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NAME_LENGTH),
    ]
    email: NormalizedEmail
    password: str

    @field_validator("password")
    @classmethod
    def validate_password_policy(cls, password: str) -> str:
        """Enforce the configured password policy.

        The minimum is read from settings at validation time rather than baked
        into the field, so raising ``PASSWORD_MIN_LENGTH`` takes effect without
        a code change.

        Args:
            password: The candidate plaintext password.

        Returns:
            str: The validated password, unchanged.

        Raises:
            ValueError: If the password is too short or too long for bcrypt.
        """
        minimum_length = get_settings().PASSWORD_MIN_LENGTH
        if len(password) < minimum_length:
            raise ValueError(
                f"Password must be at least {minimum_length} characters long."
            )
        _validate_encoded_length(password)
        return password


class UserLoginRequest(AuthRequest):
    """Payload for exchanging credentials for an access token.

    Length policy is deliberately not applied here: a password that no longer
    satisfies the current policy must still be verifiable so the account owner
    receives an authentication failure instead of a validation error that would
    imply the account does not exist.
    """

    email: NormalizedEmail
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, password: str) -> str:
        """Reject passwords that can never match a stored hash.

        Args:
            password: The candidate plaintext password.

        Returns:
            str: The validated password, unchanged.

        Raises:
            ValueError: If the password is empty or exceeds the bcrypt limit.
        """
        if not password:
            raise ValueError("Password must not be empty.")
        _validate_encoded_length(password)
        return password


class TokenResponse(BaseModel):
    """Bearer token envelope returned by the login endpoint."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(
        description="Access token lifetime in seconds, counted from issue time."
    )
