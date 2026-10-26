import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import config
from core.logging import get_logger
from database.schemas.clients import ClientsT
from security.encryption import decrypt, encrypt
from security.hash import get_hash_sha256
from security.hmac import hash_hmac
from security.secret import create_secret

log = get_logger()


@dataclass(frozen=True)
class StoredClient:
    """Represents the generated secrets and metadata returned after storing a new API key."""

    api_key: str
    hmac_secret: str
    id: uuid.UUID
    key_type: str
    owner_email: str


async def store_client(
    session: AsyncSession,
    client_name: str,
    key_type: str,
    owner_email: str,
    require_jwt: bool = True,
    require_external_id: bool = True,
    is_active: bool = True,
    api_key: str | None = None,
    hmac_secret: str | None = None,
    redirect_uri: str | None = None,
) -> StoredClient:
    """Generates and stores a new API key for a client / user.
    Action still needs to be committed."""

    log.debug("Generating new client secrets...")

    if key_type not in ["User", "Application"]:
        raise ValueError("Key_type must be 'User' or 'Application'...")

    # TODO: Build a better email check
    if not owner_email or owner_email == "":
        raise ValueError("A valid owner email must be provided...")

    blind_index_client_name_value = hash_hmac(
        content=client_name, key=config.BLIND_INDEX_HMAC_KEY
    )

    blind_index_owner_email_value = hash_hmac(
        content=owner_email, key=config.BLIND_INDEX_HMAC_KEY
    )

    existing_clients = await session.scalar(
        select(func.count())
        .select_from(ClientsT)
        .where(
            ClientsT.blind_index_client_name == blind_index_client_name_value,
            ClientsT.blind_index_owner_email == blind_index_owner_email_value,
        )
    )
    if existing_clients:
        raise ValueError(
            f"Client '{client_name}' with owner '{owner_email}' already exists, skipping..."
        )

    if api_key is None:
        ext_key = create_secret()
        hsh_key = get_hash_sha256(ext_key)

    else:
        ext_key = api_key
        hsh_key = get_hash_sha256(api_key)

    if hmac_secret is None:
        hmac = create_secret()

    else:
        hmac = hmac_secret

    key = ClientsT(
        api_key_hash=hsh_key,
        require_jwt=require_jwt,
        key_type=key_type,
        require_external_id=require_external_id,
        is_active=is_active,
        encrypted_hmac_secret=encrypt(hmac),
        encrypted_owner_email=encrypt(owner_email),
        encrypted_client_name=encrypt(client_name),
        encrypted_redirect_uri=encrypt(redirect_uri) if redirect_uri else None,
        blind_index_client_name=blind_index_client_name_value,
        blind_index_owner_email=blind_index_owner_email_value,
    )

    session.add(key)
    await session.flush()

    log.debug(f"Client stored with id '{key.id}' and secrets generated...")

    return StoredClient(
        api_key=ext_key,
        hmac_secret=decrypt(key.encrypted_hmac_secret),
        id=key.id,
        key_type=key.key_type,
        owner_email=decrypt(key.encrypted_owner_email),
    )
