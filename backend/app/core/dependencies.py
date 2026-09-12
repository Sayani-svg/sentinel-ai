"""Core FastAPI dependencies and infrastructure helpers."""

import logging
import uuid
from collections.abc import Generator
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.security import ExpiredTokenError, InvalidTokenError, decode_access_token
from app.database.session import SessionLocal

logger = get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{get_settings().API_V1_PREFIX}/auth/login"
)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use."""
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


async def get_current_user(token: str = Security(oauth2_scheme)) -> str:
    """Verify JWT credentials and return the authenticated subject."""
    try:
        payload = decode_access_token(token)
        subject = payload.get("sub")
        if not subject:
            raise InvalidTokenError("Missing subject in token.")
        return str(subject)
    except (InvalidTokenError, ExpiredTokenError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_active_user(
    current_user_id: str = Depends(get_current_user),
) -> str:
    """Return the authenticated active user subject."""
    return current_user_id
