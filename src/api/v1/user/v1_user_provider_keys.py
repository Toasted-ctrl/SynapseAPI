from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.providers import UserProviderRegistry, get_all_provider_configurations
from database.session import get_db_session
from database.user_keys import (
    KeyNotFoundError,
    delete_user_key_by_key_id,
    get_or_store_key,
)

from .models import UserKeyCreate, UserKeyRead

log = get_logger()


router = APIRouter()


@router.post(
    path="/user/provider-keys",
    tags=["User"],
    description="Stores and returns a User provided Provider API Key.",
    response_model=UserKeyRead,
)
async def store_user_key(
    payload: UserKeyCreate,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> UserKeyRead:
    p: UserProviderRegistry = await get_all_provider_configurations(
        session=session, user_id=user.id
    )
    if payload.provider in p.not_configured:
        sk = await get_or_store_key(
            session=session,
            api_key=payload.api_key,
            expiry_days=payload.expiry_days,
            user_id=user.id,
            provider_id=p[payload.provider].id,
        )
        return UserKeyRead(
            user_id=user.id,
            provider=p[payload.provider].name,
            api_key_short=sk.api_key_short,
            expires=sk.expiration_date,
        )
    raise HTTPException(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        detail=f"No keys can be configured for Provider '{payload.provider}'",
    )


@router.delete(
    path="/user/provider-keys",
    tags=["User"],
    description="Deletes a User's provider API key.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_user_key(
    key_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> None:
    try:
        await delete_user_key_by_key_id(session=session, key_id=key_id, user_id=user.id)
    except KeyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to delete API key: {e!s}",
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error: Please contact your administrator",
        )
