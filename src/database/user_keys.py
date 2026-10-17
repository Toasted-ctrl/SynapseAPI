import uuid
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.schemas.providers import ProvidersT
from database.schemas.user_keys import UserKeysT
from security.encryption import decrypt, encrypt

log = get_logger()


@dataclass(frozen=True)
class ProviderAPIKey:
    user_id: uuid.UUID
    api_key_id: uuid.UUID
    api_key: str
    api_key_short: str
    provider_id: uuid.UUID
    expiration_date: datetime


async def get_user_active_keys(
    session: AsyncSession,
    user_id: uuid.UUID
) -> list[ProviderAPIKey]:

    """Fetches all active Provider API keys for the user, 
    and returns them as a list containing ProviderAPIKey objects. If no keys are active,
    an empty list will be returned."""

    keys = (await session.scalars(
        select(UserKeysT)
        .where(
            UserKeysT.user_id == user_id,
            UserKeysT.expiration_date > datetime.now()
        )
    )).all()

    if not keys:
        log.debug(f"No active keys configured for User '{user_id}'. Returning empty list...")
        return []

    log.debug(f"Found {len(keys)} active key(s) for User '{user_id}'. Returning all keys...")
    return [
        ProviderAPIKey(
            user_id=k.user_id,
            provider_id=k.provider_id,
            api_key_id=k.id,
            api_key=k.encrypted_api_key,
            api_key_short=k.api_key_short,
            expiration_date=k.expiration_date
        )
        for k in keys
    ]


async def get_or_store_key(
    session: AsyncSession,
    api_key: str,
    user_id: uuid.UUID,
    provider_id: uuid.UUID
) -> ProviderAPIKey:

    """Adds a new API key to the database for the user.
    Returns the existing key if the user already has a key configured for the 
    indicated provider_id which is not yet expired. In this case, the key which was attempted to 
    be added, will not be added. Will raise ValueError if
    the key is now allowed to be added (unknown or unsupported provider_id)."""

    existing = await session.scalar(
        select(UserKeysT)
        .where(
            UserKeysT.user_id == user_id,
            UserKeysT.expiration_date > datetime.now(),
            UserKeysT.provider_id == provider_id
        )
        .limit(1)
    )

    if existing:
        log.info(f"Key with Provider '{existing.provider_id}' already exists for User '{existing.user_id}', returning existing key...")
        return ProviderAPIKey(
            user_id=existing.user_id,
            api_key_id=existing.id,
            api_key=decrypt(existing.encrypted_api_key),
            api_key_short=existing.api_key_short,
            provider_id=existing.provider_id,
            expiration_date=existing.expiration_date
        )

    if await session.scalar(
        select(func.count())
        .select_from(ProvidersT)
        .where(
            ProvidersT.id == provider_id,
            ProvidersT.requires_api_key == True
        )
    ) != 1:
        raise ValueError(f"Provider '{provider_id}' does not require an API Key, or Provider with id '{provider_id}' does not exist...")

    new_key = UserKeysT(
        user_id=user_id,
        encrypted_api_key=encrypt(api_key),
        api_key_short=api_key[:10],
        provider_id=provider_id
    )

    session.add(new_key)
    await session.flush()

    log.debug(f"New key added for User '{new_key.user_id}' for Provider '{new_key.provider_id}'...")

    return ProviderAPIKey(
        user_id=new_key.user_id,
        api_key_id=new_key.id,
        api_key=decrypt(new_key.encrypted_api_key),
        api_key_short=new_key.api_key_short,
        provider_id=new_key.provider_id,
        expiration_date=new_key.expiration_date
    )


class KeyNotFoundError(Exception):
    """Exception Class for if a User's key isn't found."""
    pass


async def delete_user_key_by_key_id(
    session: AsyncSession,
    key_id: uuid.UUID,
    user_id: uuid.UUID
) -> None:
    """Deletes a User's provider API key."""
    existing = (await session.execute(
        select(UserKeysT)
        .where(
            UserKeysT.id == key_id,
            UserKeysT.user_id == user_id
        )
    )).scalar_one_or_none()
    if not existing:
        log.debug(f"Could not locate API key with id '{key_id}' for User '{user_id}'.")
        raise KeyNotFoundError(f"Could not locate API key with id '{key_id}'")
    await session.delete(existing)
    await session.flush()
    return None