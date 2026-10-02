"""Core FastAPI dependencies and infrastructure helpers."""

import logging
import uuid
from collections.abc import Callable, Generator
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.security import ExpiredTokenError, InvalidTokenError, decode_access_token
from app.models.user import User
from app.schemas.user import VALID_ROLE_VALUES, UserRole

logger = get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{get_settings().API_V1_PREFIX}/auth/login"
)

#: Every authentication failure reports this same detail, so a caller cannot
#: distinguish an expired token from a forged one, or a deleted account from a
#: token whose subject was never a user id.
INVALID_CREDENTIALS_DETAIL: str = "Invalid authentication credentials"
BEARER_CHALLENGE: dict[str, str] = {"WWW-Authenticate": "Bearer"}


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use."""
    from app.database.session import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_request_id() -> str:
    """Return a unique request ID for tracing."""
    return str(uuid.uuid4())


def get_request_time() -> datetime:
    """Return the current UTC time for request tracing."""
    return datetime.now(timezone.utc)


def get_logger_dependency() -> logging.Logger:
    """Return the configured application logger."""
    return logger


def _unauthorized() -> HTTPException:
    """Build the single 401 response used for every authentication failure.

    Returns:
        HTTPException: A 401 carrying the bearer challenge header.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=INVALID_CREDENTIALS_DETAIL,
        headers=BEARER_CHALLENGE,
    )


def get_current_user(
    token: str = Security(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Verify JWT credentials and return the authenticated account.

    The subject claim is resolved to a row in the ``users`` table rather than
    trusted on its own, so a token stops working the moment its account is
    deleted and a role change takes effect without waiting for the token to
    expire.

    Args:
        token: Bearer token extracted by :data:`oauth2_scheme`.
        db: Database session used to load the account.

    Returns:
        User: The account that owns the token.

    Raises:
        HTTPException: 401 if the token is invalid or expired, carries no
            subject, carries a subject that is not a user id, or names an
            account that no longer exists.
    """
    try:
        payload = decode_access_token(token)
    except (InvalidTokenError, ExpiredTokenError) as exc:
        logger.info("Rejected a request carrying an unusable access token.")
        raise _unauthorized() from exc

    subject = payload.get("sub")
    if not subject:
        raise _unauthorized() from None

    try:
        user_id = int(subject)
    except (TypeError, ValueError) as exc:
        logger.warning("Access token carried a non-numeric subject %r.", subject)
        raise _unauthorized() from exc

    user = db.get(User, user_id)
    if user is None:
        logger.warning("Access token named account %d, which no longer exists.", user_id)
        raise _unauthorized()

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Return the authenticated account, rejecting unusable role assignments.

    ``users`` has no ``is_active`` column, so "active" here means the account
    carries a role this build recognises. A row holding an unexpected role is
    refused before any route can act on it.

    Args:
        current_user: Account resolved from the bearer token.

    Returns:
        User: The authenticated account, when its role is recognised.

    Raises:
        HTTPException: 403 if the stored role is not a recognised
            :class:`~app.schemas.user.UserRole`.
    """
    if current_user.role not in VALID_ROLE_VALUES:
        logger.error(
            "Account %d holds unrecognised role %r and was refused.",
            current_user.id,
            current_user.role,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account role is not permitted to access this resource",
        )
    return current_user


def require_roles(*allowed_roles: UserRole) -> Callable[..., User]:
    """Build a dependency that admits only the listed roles.

    Args:
        *allowed_roles: Roles permitted to reach the dependent route.

    Returns:
        Callable[..., User]: A FastAPI dependency raising 403 for any account
        outside ``allowed_roles``.

    Raises:
        ValueError: If no role is supplied, which would guard nothing.
    """
    if not allowed_roles:
        raise ValueError("require_roles() needs at least one role.")

    # Compared as plain strings rather than enum members: ``users.role`` is a
    # text column and SQLAlchemy hands back a bare ``str``.
    permitted = frozenset(role.value for role in allowed_roles)

    def dependency(current_user: User = Depends(get_current_active_user)) -> User:
        """Admit the request only when the account holds a permitted role.

        Args:
            current_user: Account resolved and role-checked from the token.

        Returns:
            User: The authenticated account, when its role is permitted.

        Raises:
            HTTPException: 403 if the account's role is not permitted.
        """
        if current_user.role not in permitted:
            logger.warning(
                "Account %d with role %r was refused; permitted roles are %s.",
                current_user.id,
                current_user.role,
                ", ".join(sorted(permitted)),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access this resource",
            )
        return current_user

    return dependency


#: Any account holding a recognised role.
require_viewer_or_above = require_roles(
    UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN
)

#: Accounts cleared to analyse logs and create predictions.
require_analyst_or_above = require_roles(UserRole.ANALYST, UserRole.ADMIN)

#: Accounts cleared to administer users and delete records.
require_admin = require_roles(UserRole.ADMIN)
