"""Centralized security module for Sentinel AI application."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.logger import get_logger

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Constants - Configuration values
JWT_ISSUER = "Sentinel AI"
PWD_CONTEXT = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=getattr(settings, "BCRYPT_ROUNDS", 12),
)


# Custom Exceptions
class AuthenticationError(Exception):
    """Base class for authentication errors."""

class InvalidTokenError(AuthenticationError):
    """Raised when the token is invalid or missing required claims."""


class ExpiredTokenError(AuthenticationError):
    """Raised when the token has expired."""


class AuthorizationError(AuthenticationError):
    """Raised when the user is not authorized."""


# Password Hashing
def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        str: The hashed password.
    """
    return PWD_CONTEXT.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    Args:
        password: The plaintext password.
        hashed_password: The hashed password.

    Returns:
        bool: True if password matches, False otherwise.
    """
    return PWD_CONTEXT.verify(password, hashed_password)


# JWT Management
def create_access_token(
    subject: str,
    additional_claims: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a new JWT access token.

    Args:
        subject: The subject of the token (e.g., user ID).
        additional_claims: Optional custom claims to include in the payload.
        expires_delta: Optional custom expiration duration.

    Returns:
        str: The encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))

    payload = {
        "sub": subject,
        "iat": now,
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "iss": JWT_ISSUER,
    }

    if additional_claims:
        payload.update(additional_claims)

    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token.

    Args:
        token: The encoded JWT string.

    Returns:
        dict[str, Any]: The decoded token payload.

    Raises:
        ExpiredTokenError: If the token has expired.
        InvalidTokenError: If the token is invalid, malformed, or missing required claims.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            issuer=JWT_ISSUER,
        )

        # Validate required claims
        REQUIRED_CLAIMS = frozenset(
        {
            "sub",
            "jti",
            "iss",
        }
        )
        for claim in REQUIRED_CLAIMS:
            if claim not in payload:
                logger.error("Token missing required claim.")
                raise InvalidTokenError(f"Missing required claim: {claim}")

        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Attempted to use an expired JWT.")
        raise ExpiredTokenError("Token has expired.")
    except (jwt.InvalidTokenError, KeyError):
        logger.warning("Invalid JWT received.")
        raise InvalidTokenError("Invalid token.")


def get_current_subject(token: str) -> str:
    """Extract the subject from a JWT token.

    Args:
        token: The encoded JWT string.

    Returns:
        str: The subject identifier.
    """
    payload = decode_access_token(token)
    return str(payload["sub"])


# Sensitive Data Masking
def mask_sensitive(value: str) -> str:
    """Mask sensitive strings for logging.

    Args:
        value: The sensitive string value.

    Returns:
        str: A masked version of the value.
    """
    if len(value) <= 8:
        return "********"
    return f"{value[:4]}****{value[-4:]}"
