import uuid
from dataclasses import dataclass
from enum import Enum

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.schemas.clients import ClientsT
from database.session import get_db_session
from security.hash import get_hash_sha256

log = get_logger()

api_key_header = APIKeyHeader(name="X-API-Key")


class KeyType(str, Enum):
    APPLICATION = "Application"
    USER = "User"


@dataclass(frozen=True)
class VerifiedClient:
    id: uuid.UUID
    key_type: KeyType


async def depends_get_client(
    api_key: str = Security(api_key_header),
    session: AsyncSession = Depends(get_db_session),
) -> VerifiedClient:
    """Verifies that the client belonging to the API key exists in the database.
    Return VerifiedClient dataclass if so, else raises HTTPException."""

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API Key"
        )

    try:
        client = (
            await session.execute(
                select(ClientsT).where(
                    ClientsT.api_key_hash == get_hash_sha256(api_key)
                )
            )
        ).scalar_one_or_none()
    except MultipleResultsFound:
        log.critical(
            f"Multiple Clients matched with single API key hash: {get_hash_sha256(api_key)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error. Please contact your administrator.",
        )

    if not client:
        log.warning(
            f"API key authentication failed - no matching client for API key hash: {get_hash_sha256(api_key)}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )

    log.debug(f"Verified Client: {client.id}")
    return VerifiedClient(id=client.id, key_type=KeyType(client.key_type))


def depends_get_application_client(
    client: VerifiedClient = Depends(depends_get_client),
) -> VerifiedClient:
    """Verifying if the client is a frontend application client.
    Returns VerifiedClient dataclass, otherwise raises HTTPException."""

    if client.key_type != KeyType.APPLICATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client must be a frontend application, with a login requirement.",
        )

    return client


async def verify_client_from_application_id(
    application_id: str, session: AsyncSession
) -> VerifiedClient:
    """Checks and verifies the application id (api key for frontiend clients) in the db."""
    query = (
        await session.execute(
            select(ClientsT).where(
                ClientsT.api_key_hash == get_hash_sha256(application_id),
                ClientsT.key_type == KeyType.APPLICATION,
            )
        )
    ).scalar_one_or_none()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Client ID"
        )
    return VerifiedClient(id=query.id, key_type=KeyType(query.key_type))
