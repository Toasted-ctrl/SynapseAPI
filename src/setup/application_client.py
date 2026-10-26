from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.store_client import StoredClient, store_client

log = get_logger()

# TODO: Build tests.


async def create_application_client(
    session: AsyncSession,
    client_name: str,
    key_type: str,
    owner_email: str,
    require_jwt: bool,
    require_external_id: bool,
    redirect_uri: str,
    api_key: str | None = None,
    hmac_secret: str | None = None,
) -> StoredClient | None:
    """Creates a new Frontend Application client."""

    log.debug("Creating new frontend client...")

    try:
        client = await store_client(
            session=session,
            client_name=client_name,
            key_type=key_type,
            owner_email=owner_email,
            require_jwt=require_jwt,
            require_external_id=require_external_id,
            api_key=api_key,
            hmac_secret=hmac_secret,
            redirect_uri=redirect_uri,
        )

        return client

    except ValueError as e:
        log.info(e)
        return None
