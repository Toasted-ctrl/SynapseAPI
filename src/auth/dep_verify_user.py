import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_client import VerifiedClient, depends_get_client
from core.logging import get_logger
from database.schemas.persons_users import UsersT
from database.schemas.user_sessions import SessionsT
from database.session import get_db_session
from security.hash import get_hash_sha256

log = get_logger()


class KeyType(str, Enum):
    APPLICATION = "Application"
    USER = "User"


@dataclass(frozen=True)
class VerifiedUser:
    id: uuid.UUID


async def depends_verify_user(
    client: VerifiedClient = Depends(depends_get_client),
    session: AsyncSession = Depends(get_db_session),
    session_id: str | None = Cookie(default=None),
) -> VerifiedUser:
    """Resolved a VerifiedClient to a VerifiedUser.
    Returns VerifiedUser if valid, otherwise raises HTTPException."""

    kt: KeyType = client.key_type

    match kt:
        case KeyType.APPLICATION:
            if not session_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing session cookie",
                )

            user = (
                await session.execute(
                    select(SessionsT).where(
                        SessionsT.session_id_hash == get_hash_sha256(session_id),
                        SessionsT.expiration_date > datetime.now(UTC),
                    )
                )
            ).scalar_one_or_none()

            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired session",
                )

            client_id_check = (
                await session.execute(
                    select(UsersT).where(
                        UsersT.api_key_id == client.id, UsersT.id == user.user_id
                    )
                )
            ).scalar_one_or_none()

            if client_id_check is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User is not authorized",
                )

            log.debug(f"Verified User: {user.user_id}")
            return VerifiedUser(id=user.user_id)

        case KeyType.USER:
            try:
                user = (
                    await session.execute(
                        select(UsersT).where(UsersT.api_key_id == client.id)
                    )
                ).scalar_one_or_none()
            except MultipleResultsFound:
                log.critical(f"Multiple Users found for api_key_id: {client.id}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unexpected error. Please contact your administrator.",
                )

            log.debug(f"Verified User: {user.id}")
            return VerifiedUser(id=user.id)

        case _:
            log.error(f"Invalid key_type: {client.key_type}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected error. Please contact your administrator.",
            )
