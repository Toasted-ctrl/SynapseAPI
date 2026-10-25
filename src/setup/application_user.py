import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.person import Person, get_or_store_person
from database.user import User, get_or_store_user

log = get_logger()


@dataclass(frozen=True)
class VerifiedApplicationUser:
    client_id: uuid.UUID
    user_id: uuid.UUID
    person_id: uuid.UUID


async def get_or_create_application_user(
    first_name: str,
    last_name: str,
    client_id: uuid.UUID,
    login_provider: str,
    email: str,
    external_id: str,
    session: AsyncSession,
) -> VerifiedApplicationUser:
    """Stores or fetches a new frontend Application User."""

    log.debug("Creating or fetching Application User...")

    person: Person = await get_or_store_person(
        session=session, first_name=first_name, last_name=last_name, email=email
    )

    user: User = await get_or_store_user(
        session=session,
        person_id=person.id,
        api_key_id=client_id,
        key_type="Application",
        login_provider=login_provider,
        external_id=external_id,
    )

    log.debug("Returning Application User...")

    return VerifiedApplicationUser(
        client_id=client_id, user_id=user.id, person_id=person.id
    )
