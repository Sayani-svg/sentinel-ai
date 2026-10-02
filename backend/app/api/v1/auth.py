"""Authentication endpoints: self-service registration and token issuance.

``POST /auth/login`` accepts a JSON body rather than the form encoding implied
by the OpenAPI password flow, so the interactive "Authorize" button in the docs
page does not apply to this endpoint. The bearer scheme itself is unchanged:
``Authorization: Bearer <token>`` is what :data:`app.core.dependencies.oauth2_scheme`
reads on protected routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.dependencies import BEARER_CHALLENGE, get_db
from app.core.logger import get_logger
from app.core.security import create_access_token
from app.models.user import User
from app.schemas.auth import TokenResponse, UserLoginRequest, UserRegisterRequest
from app.schemas.user import UserRead
from app.services.auth_service import (
    INVALID_CREDENTIALS_MESSAGE,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    authenticate_user,
    register_user,
)

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new account",
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "The email address already belongs to an account."
        }
    },
)
def register_account(
    payload: UserRegisterRequest,
    db: Session = Depends(get_db),
) -> User:
    """Create a new account with the Viewer role.

    Registration does not issue a token; the caller logs in afterwards so that
    credential verification always runs.

    Args:
        payload: Validated registration request.
        db: Database session.

    Returns:
        User: The created account, without its password hash.

    Raises:
        HTTPException: 409 if the email address is already registered.
    """
    try:
        return register_user(
            db, name=payload.name, email=payload.email, password=payload.password
        )
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account already exists for this email address",
        ) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for an access token",
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "The credentials do not identify an account."
        }
    },
)
def login(
    payload: UserLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Verify credentials and issue an access token.

    The token carries only the account id as its subject. Authorization always
    reads the role from the database, so revoking a role takes effect on the
    next request instead of when the token happens to expire.

    Args:
        payload: Validated login request.
        db: Database session.

    Returns:
        TokenResponse: The bearer token and its lifetime in seconds.

    Raises:
        HTTPException: 401 if the credentials do not identify an account.
    """
    try:
        user = authenticate_user(
            db, email=payload.email, password=payload.password
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INVALID_CREDENTIALS_MESSAGE,
            headers=BEARER_CHALLENGE,
        ) from exc

    token = create_access_token(subject=str(user.id))
    logger.info("Issued an access token for account %d.", user.id)
    return TokenResponse(
        access_token=token,
        expires_in=get_settings().ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
