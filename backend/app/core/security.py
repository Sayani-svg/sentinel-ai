"""Centralized security module for Sentinel AI application."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.core.config import get_settings
from app.core.logger import get_logger

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Constants - Configuration values
JWT_ISSUER = "Sentinel AI"
BCRYPT_ROUNDS = getattr(settings, "BCRYPT_ROUNDS", 12)

# bcrypt ignores input past 72 *bytes* rather than characters, so a multi-byte
# password can cross the boundary at fewer than 72 characters. Request schemas
# check the encoded length against this limit so such passwords are rejected
# with a 422 instead of raising from :func:`hash_password` at request time.
BCRYPT_MAX_PASSWORD_BYTES = 72


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

    Each call generates a fresh random salt, so the same password never yields
    the same digest twice. The ``bcrypt`` package is called directly because
    ``passlib`` 1.7.4 cannot load ``bcrypt`` 5.x: it introspects the removed
    ``bcrypt.__about__`` attribute and then fails every hash operation.

    Args:
        password: The plaintext password.

    Returns:
        str: The hashed password, including its algorithm id and salt.

    Raises:
        ValueError: If the password exceeds the 72-byte limit that bcrypt
            imposes on its input. The error is raised rather than truncating
            silently, so two passwords sharing a 72-byte prefix cannot collide.
            Length policy is enforced upstream by
            :attr:`~app.core.config.Settings.PASSWORD_MIN_LENGTH`.
    """
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    Args:
        password: The plaintext password.
        hashed_password: The hashed password.

    Returns:
        bool: True if password matches, False otherwise. An unreadable stored
            hash is reported as a non-match rather than raised, so a corrupt
            record cannot turn into a server error on the login path.
    """
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except ValueError as exc:
        logger.warning("Password verification failed: %s", exc)
        return False


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

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


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
            settings.SECRET_KEY,
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
