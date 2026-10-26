import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.person import get_or_store_person
from database.store_client import store_client
from database.user import get_or_store_user

log = get_logger()


@dataclass(frozen=True)
class UserClient:
    api_key: str
    hmac_secret: str
    owner_email: str
    key_type: str
    user_id: uuid.UUID


async def create_user_client_user(
    session: AsyncSession,
    client_name: str,
    key_type: str,
    owner_email: str,
    first_name: str,
    last_name: str,
    require_jwt: bool = False,
    require_external_id: bool = False,
    api_key: str | None = None,
    hmac_secret: str | None = None,
) -> UserClient:
    """Creates a new Backend Client User. Returns None if the Backend Client User already exists, or runs into an error.
    If an error occurs, a rollback will be issued."""

    log.info("Starting Creation of new Backend Client User...")

    client = await store_client(
        session=session,
        client_name=client_name,
        key_type=key_type,
        owner_email=owner_email,
        require_jwt=require_jwt,
        require_external_id=require_external_id,
        api_key=api_key,
        hmac_secret=hmac_secret,
    )

    person = await get_or_store_person(
        session=session, first_name=first_name, last_name=last_name, email=owner_email
    )

    user = await get_or_store_user(
        session=session,
        person_id=person.id,
        api_key_id=client.id,
        key_type=client.key_type,
    )

    log.info(f"Created new Backend Client User: '{user.id}'...")

    return UserClient(
        api_key=client.api_key,
        hmac_secret=client.hmac_secret,
        owner_email=client.owner_email,
        key_type=client.key_type,
        user_id=user.id,
    )
