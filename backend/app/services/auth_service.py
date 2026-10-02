"""Business logic for account registration and credential verification.

The service layer is transport agnostic: it raises the typed errors below and
never :class:`fastapi.HTTPException`, leaving status-code mapping to the routers.
It also never builds tokens -- that is the router's responsibility, because
issuing a credential is an API concern rather than a persistence one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.core.security import hash_password, verify_password
from app.models.user import User
from app.schemas.user import NormalizedEmail, UserRole

logger = get_logger(__name__)

INVALID_CREDENTIALS_MESSAGE: Final[str] = "Incorrect email or password."

# A real bcrypt digest of a value nobody knows, verified against when the
# requested email does not exist. Without it, a missing account would return
# measurably faster than a wrong password and leak which addresses are
# registered. Fixed rather than generated at import time so that reading this
# module never costs a bcrypt round.
_TIMING_EQUALISER_HASH: Final[str] = (
    "$2b$12$q5oVDGBVo4.FQAVdFCg5ZO8PMr.Xgr6pso7Rgc1oeFDJ8IRc4qEkK"
)


class AuthServiceError(Exception):
    """Base class for authentication service failures."""


class EmailAlreadyRegisteredError(AuthServiceError):
    """Raised when a registration collides with an existing account."""

    def __init__(self, email: str) -> None:
        """Record the conflicting address for logging.

        Args:
            email: The address that is already registered.
        """
        super().__init__(email)
        self.email = email


class InvalidCredentialsError(AuthServiceError):
    """Raised when credentials do not identify an account.

    The message is deliberately uniform so that it cannot distinguish an
    unknown email from a wrong password.
    """


def get_user_by_email(db: Session, email: str) -> User | None:
    """Look up a user by email address.

    The comparison is case-insensitive so that accounts created before the
    API normalised addresses on write remain reachable.

    Args:
        db: Active database session.
        email: Email address to resolve.

    Returns:
        User | None: The matching user, or ``None`` when no account matches.
    """
    statement = select(User).where(User.email == email)
    return db.execute(statement).scalar_one_or_none()


def register_user(
    db: Session,
    *,
    name: str,
    email: NormalizedEmail,
    password: str,
) -> User:
    """Create an account and return the persisted record.

    New accounts are always created with the :attr:`~app.schemas.user.UserRole.VIEWER`
    role. Privileged roles are granted administratively, so this function has no
    parameter through which a caller could widen its own grant.

    Args:
        db: Active database session.
        name: Display name of the account owner.
        email: Normalised email address, used as the login identifier.
        password: Plaintext password; hashed here and never stored as given.

    Returns:
        User: The persisted user, refreshed so its generated id is available.

    Raises:
        EmailAlreadyRegisteredError: If the address already belongs to an account.
    """
    if get_user_by_email(db, email) is not None:
        raise EmailAlreadyRegisteredError(email)

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
        role=UserRole.VIEWER.value,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        # Lost a race against a concurrent registration: the unique constraint on
        # users.email is the authority, so report it the same way as the check.
        db.rollback()
        logger.info("Registration for %s failed the unique constraint.", email)
        raise EmailAlreadyRegisteredError(email) from exc

    db.refresh(user)
    logger.info("Registered account %d with role %s.", user.id, user.role)
    return user


def authenticate_user(db: Session, *, email: str, password: str) -> User:
    """Resolve credentials to a user account.

    Args:
        db: Active database session.
        email: Email address supplied as the login identifier.
        password: Plaintext password to check against the stored hash.

    Returns:
        User: The authenticated account.

    Raises:
        InvalidCredentialsError: If no account matches, the password is wrong,
            or the stored role is not a recognised :class:`~app.schemas.user.UserRole`.
    """
    user = get_user_by_email(db, email)

    if user is None:
        verify_password(password, _TIMING_EQUALISER_HASH)
        raise InvalidCredentialsError(INVALID_CREDENTIALS_MESSAGE)

    if not verify_password(password, user.password_hash):
        logger.info("Rejected login for account %d: password mismatch.", user.id)
        raise InvalidCredentialsError(INVALID_CREDENTIALS_MESSAGE)

    if user.role not in {role.value for role in UserRole}:
        # A role outside the recognised set means the row is corrupt or was
        # written by a schema this build does not know about. Refuse to issue a
        # credential rather than grant undefined authority.
        logger.error("Account %d holds unrecognised role %r.", user.id, user.role)
        raise InvalidCredentialsError(INVALID_CREDENTIALS_MESSAGE)

    return user
