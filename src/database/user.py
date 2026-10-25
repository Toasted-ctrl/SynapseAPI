import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.schemas.persons_users import UsersT

log = get_logger()


@dataclass(frozen=True)
class UserDetails:
    user_id: uuid.UUID
    person_id: uuid.UUID


async def get_user_by_user_id(
    session: AsyncSession, user_id: uuid.UUID
) -> UserDetails | None:
    """Retrieves a user by ID."""

    user = (
        await session.execute(select(UsersT).where(UsersT.id == user_id))
    ).scalar_one_or_none()

    if not user:
        return None

    return UserDetails(user_id=user.id, person_id=user.person_id)


@dataclass(frozen=True)
class User:
    id: uuid.UUID


async def get_or_store_user(
    session: AsyncSession,
    person_id: uuid.UUID,
    api_key_id: uuid.UUID,
    key_type: str,
    login_provider: str | None = None,
    external_id: str | None = None,
) -> User:

    if key_type == "Application" and external_id == None:
        raise ValueError(
            "Unable to fetch or add User, if key_type = 'Application', external_id must not be None"
        )

    stored_user = await session.scalar(
        select(UsersT)
        .where(
            UsersT.person_id == person_id,
            UsersT.api_key_id == api_key_id,
            UsersT.external_id == external_id,
            UsersT.login_provider == login_provider,
        )
        .limit(1)
    )
    if stored_user:
        log.info(
            f"User already exists, returning existing record: '{stored_user.id}'..."
        )
        return User(id=stored_user.id)

    new_user = UsersT(
        person_id=person_id,
        api_key_id=api_key_id,
        external_id=external_id,
        login_provider=login_provider,
    )

    session.add(new_user)
    await session.flush()

    log.info(f"Added User with id: {new_user.id}")

    return User(id=new_user.id)
