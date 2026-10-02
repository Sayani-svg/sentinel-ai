"""Endpoints exposing the authenticated account."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependencies import require_viewer_or_above
from app.models.user import User
from app.schemas.user import UserRead

router = APIRouter()


@router.get(
    "/me",
    response_model=UserRead,
    summary="Return the authenticated account",
    responses={
        401: {"description": "The bearer token is missing, invalid or expired."},
        403: {"description": "The account holds an unusable role."},
    },
)
def read_current_account(
    current_user: User = Depends(require_viewer_or_above),
) -> User:
    """Return the profile of the account that owns the bearer token.

    Args:
        current_user: Account resolved from the token and role-checked by
            :data:`~app.core.dependencies.require_viewer_or_above`.

    Returns:
        User: The authenticated account, without its password hash.
    """
    return current_user
