import uuid
from dataclasses import dataclass
from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.schemas.providers import ProvidersT
from database.user_keys import get_user_active_keys
from providers.dataclasses import LangChainCon

log = get_logger()


@dataclass(frozen=True)
class Provider:
    id: uuid.UUID
    name: str
    base_url: str
    internal: bool
    requires_api_key: bool
    langchain_con: LangChainCon
    api_key_configured: bool | None = None
    mac_address: str | None = None


@dataclass(frozen=True)
class ProviderConfiguration:
    id: uuid.UUID
    name: str
    base_url: str
    langchain_con: LangChainCon
    api_key_configured: bool
    internal: bool
    requires_api_key: bool = True
    encrypted_api_key: str | None = None
    api_key_short: str | None = None
    api_key_id: uuid.UUID | None = None
    api_key_expiry_date: datetime | None = None


class UserProviderRegistry:
    """Allows attribute-style access to providers by name."""

    def __init__(self, providers: list[ProviderConfiguration]):
        self._providers: dict[str, ProviderConfiguration] = {
            p.name: p for p in providers
        }

    def __getattr__(self, name: str) -> ProviderConfiguration:
        try:
            return self._providers[name]
        except KeyError:
            raise AttributeError(f"No provider names '{name}'...")

    def __getitem__(self, name: str) -> ProviderConfiguration:
        return self._providers[name]

    def __contains__(self, name: str) -> bool:
        return name in self._providers

    def __iter__(self):
        return iter(self._providers.values())

    @property
    def names(self) -> list[str]:
        """Returns a list of all Provider names."""
        return list(self._providers.keys())

    @property
    def not_configured(self) -> list[str]:
        """Returns a list of Provider names that require an API key, but don't have one configured."""
        return [
            name
            for name, provider in self._providers.items()
            if provider.requires_api_key and not provider.api_key_configured
        ]

    def __repr__(self):
        return f"ProviderRegistry({list(self._providers.keys())})"


async def get_or_create_provider(
    session: AsyncSession,
    name: str,
    langchain_con: str,
    base_url: str,
    internal: bool,
    requires_api_key: bool,
    host: str | None = None,
    mac_address: str | None = None,
) -> Provider:
    """Creates a new or fetches existing Provider."""

    existing = await session.scalar(
        select(ProvidersT).where(ProvidersT.base_url == base_url).limit(1)
    )

    if existing:
        log.info(
            f"Provider '{name}' with url '{base_url}' already exists, fetching existing Provider..."
        )
        return Provider(
            id=existing.id,
            name=existing.name,
            base_url=existing.base_url,
            internal=existing.internal,
            requires_api_key=existing.requires_api_key,
            langchain_con=LangChainCon(existing.langchain_con),
        )

    log.info("Creating new Provider...")

    new_provider = ProvidersT(
        name=name,
        langchain_con=langchain_con,
        base_url=base_url,
        internal=internal,
        requires_api_key=requires_api_key,
        host=host,
        mac_address=mac_address,
    )

    session.add(new_provider)
    await session.flush()

    log.info(
        f"New Provider created with url '{new_provider.base_url}', returning new Provider..."
    )

    return Provider(
        id=new_provider.id,
        name=new_provider.name,
        base_url=new_provider.base_url,
        internal=new_provider.internal,
        requires_api_key=new_provider.requires_api_key,
        langchain_con=LangChainCon(new_provider.langchain_con),
        mac_address=new_provider.mac_address,
    )


async def get_all_provider_configurations(
    session: AsyncSession, user_id: uuid.UUID
) -> UserProviderRegistry:
    """Fetches and returns a UserProviderRegistry object for a specified User.
    The registry will include for every Provider the API Key (encrypted) if available, as well as
    whether the Provider is internal or external, what connection it should use, etc."""
    providers = (
        await session.execute(
            select(
                ProvidersT.id,
                ProvidersT.base_url,
                ProvidersT.name,
                ProvidersT.langchain_con,
                ProvidersT.internal,
            )
        )
    ).all()
    keys = await get_user_active_keys(session=session, user_id=user_id)
    conf = [
        ProviderConfiguration(
            id=p.id,
            base_url=p.base_url,
            name=p.name,
            langchain_con=LangChainCon(p.langchain_con),
            encrypted_api_key=next(
                (k.api_key for k in keys if k.provider_id == p.id), None
            ),
            api_key_short=next(
                (k.api_key_short for k in keys if k.provider_id == p.id), None
            ),
            api_key_configured=not p.internal
            and any(k.provider_id == p.id for k in keys),
            api_key_id=next(
                (k.api_key_id for k in keys if k.provider_id == p.id), None
            ),
            api_key_expiry_date=next(
                (k.expiration_date for k in keys if k.provider_id == p.id), None
            ),
            requires_api_key=not p.internal,
            internal=p.internal,
        )
        for p in providers
    ]
    return UserProviderRegistry(providers=conf)


async def get_providers_by_location(
    session: AsyncSession, is_internal: bool
) -> list[Provider]:
    """Returns a list of providers based on location (internal or external)."""

    providers = (
        await session.execute(
            select(
                ProvidersT.id,
                ProvidersT.base_url,
                ProvidersT.name,
                ProvidersT.internal,
                ProvidersT.requires_api_key,
                ProvidersT.langchain_con,
            ).where(ProvidersT.internal == is_internal)
        )
    ).all()

    if not providers:
        return []

    return [
        Provider(
            id=p.id,
            base_url=p.base_url,
            name=p.name,
            langchain_con=LangChainCon(p.langchain_con),
            internal=p.internal,
            requires_api_key=p.requires_api_key,
        )
        for p in providers
    ]


# TODO: Build test
async def get_providers_by_id(
    session: AsyncSession, ids: list[uuid.UUID]
) -> list[Provider]:
    """Returns a list of Provider objects based on the provided Provider IDs."""

    providers = (
        await session.scalars(select(ProvidersT).where(ProvidersT.id.in_(ids)))
    ).all()

    if not providers:
        return []

    return [
        Provider(
            id=p.id,
            base_url=p.base_url,
            name=p.name,
            langchain_con=LangChainCon(p.langchain_con),
            internal=p.internal,
            requires_api_key=p.requires_api_key,
        )
        for p in providers
    ]


async def get_provider_config(
    session: AsyncSession, provider_name: str, user_id: uuid.UUID
) -> ProviderConfiguration:
    """ "Retrieves the Provider configuration for the indicated Provider if configured.
    If the providers is not configured, or is not supported, will raise a HTTPException."""

    p_reg = await get_all_provider_configurations(session=session, user_id=user_id)

    if provider_name not in p_reg.names or provider_name in p_reg.not_configured:
        log.debug(f"Unsupported/unconfigured provider: '{provider_name}'.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider_name}' is not supported or not configured",
        )

    return getattr(p_reg, provider_name)


@dataclass(frozen=True)
class ProviderRecord:
    id: uuid.UUID
    name: str
    base_url: str
    internal: bool
    requires_api_key: bool
    langchain_con: str
    mac_address: str | None = None
    host: str | None = None


class ProviderExistsError(ValueError):
    """Raised when a ProviderRecord the User attempts to add already exists.

    This exception is raised by `insert_provider_record` when a ProviderRecord with the same
    combination of attributes (name, base_url, internal, requires_api_key,
    langchain_con, and mac_address) already exists in the database.

    Attributes:
        None - inherits from ValueError, contains only the error message.
    """

    pass


async def insert_provider_record(
    name: str,
    base_url: str,
    internal: bool,
    requires_api_key: bool,
    langchain_con: LangChainCon,
    session: AsyncSession,
    mac_address: str | None = None,
) -> ProviderRecord:
    """Inserts a new ProviderRecord into the database after checking for duplicates.

    This function creates a new ProviderRecord in the database. Before insertion,
    it checks if a ProviderRecord with the exact same attributes already exists. If a
    duplicate is found, it raises a `ProviderExistsError`.

    Args:
        name: The name of the ProviderRecord (e.g., "Ollama", "OpenAI").
        base_url: The base URL of the ProviderRecord's API endpoint.
        internal: Boolean indicating whether the ProviderRecord is internal (True) or external (False).
        requires_api_key: Boolean indicating whether the ProviderRecord requires an API key for authentication.
        langchain_con: The LangChain connection type to use with this ProviderRecord (e.g., ChatOllama, ChatOpenAI).
        session: The async SQLAlchemy session to use for database operations.
        mac_address: Optional MAC address associated with the ProviderRecord. Defaults to None.

    Returns:
        ProviderRecord: A ProviderRecord dataclass instance containing the newly created ProviderRecord's attributes,
                       including the auto-generated ID.

    Raises:
        ProviderExistsError: If a ProviderRecord with the same combination of attributes already exists
                            in the database.

    Example:
        >>> from database.providers import insert_provider_record
        >>> from providers.dataclasses import LangChainCon
        >>> from sqlalchemy.ext.asyncio import AsyncSession
        >>>
        >>> async def create_ollama_provider_record(session: AsyncSession):
        ...     provider = await insert_provider_record(
        ...         name="MyOllama",
        ...         base_url="http://localhost:11434",
        ...         internal=True,
        ...         requires_api_key=False,
        ...         langchain_con=LangChainCon.OLLAMA,
        ...         session=session,
        ...         mac_address="00:11:22:33:44:55"
        ...     )
        ...     return provider
    """
    exists = (
        await session.execute(
            select(ProvidersT).where(
                ProvidersT.name == name,
                ProvidersT.base_url == base_url,
                ProvidersT.internal == internal,
                ProvidersT.requires_api_key == requires_api_key,
                ProvidersT.langchain_con == langchain_con,
                ProvidersT.mac_address == mac_address,
            )
        )
    ).scalar_one_or_none()
    if exists:
        raise ProviderExistsError(f"Provider {name} already exists.")
    p = ProvidersT(
        name=name,
        base_url=base_url,
        langchain_con=langchain_con,
        mac_address=mac_address,
        internal=internal,
        requires_api_key=requires_api_key,
    )
    session.add(p)
    await session.flush()
    await session.refresh(p)
    log.info(f"Added new Provider {name} at base_url {base_url}.")
    return ProviderRecord(
        id=p.id,
        name=p.name,
        base_url=p.base_url,
        internal=p.internal,
        requires_api_key=p.requires_api_key,
        langchain_con=p.langchain_con,
        mac_address=p.mac_address,
        host=p.host,
    )


class ProviderNotFoundError(ValueError):
    """Raised when a ProviderRecord cannot be found in the ProvidersT table."""


async def delete_provider_record_by_id(
    id: uuid.UUID, session: AsyncSession
) -> ProviderRecord:
    """Deletes a ProviderRecord from the database by its ID.

    Args:
        id: The UUID of the ProviderRecord to delete.
        session: The async SQLAlchemy session to use for database operations.

    Returns:
        ProviderRecord: The deleted ProviderRecord, for confirmation and logging purposes.

    Raises:
        ProviderNotFoundError: If no ProviderRecord with the given ID exists in the database.

    Example:
        >>> from database.providers import delete_provider_record_by_id
        >>> from sqlalchemy.ext.asyncio import AsyncSession
        >>>
        >>> async def remove_provider_record(session: AsyncSession, provider_id: uuid.UUID):
        ...     deleted_record = await delete_provider_record_by_id(id=provider_id, session=session)
        ...     await session.commit()
        ...     return deleted_record
    """
    exists = (
        await session.execute(select(ProvidersT).where(ProvidersT.id == id))
    ).scalar_one_or_none()
    if not exists:
        raise ProviderNotFoundError(f"ProviderRecord with ID {id} does not exist.")
    deleted_record = ProviderRecord(
        id=exists.id,
        name=exists.name,
        base_url=exists.base_url,
        internal=exists.internal,
        requires_api_key=exists.requires_api_key,
        langchain_con=exists.langchain_con,
        mac_address=exists.mac_address,
        host=exists.host,
    )
    await session.delete(exists)
    log.info(f"Deleted ProviderRecord with ID {id} from database.")
    return deleted_record


class NoProvidersFoundError(ValueError):
    """Raised when no Providers are found in the database.

    This exception is raised by `get_all_provider_records` when querying the ProvidersT table
    returns no results, indicating that the database contains no Provider records.

    Attributes:
        None - inherits from ValueError, contains only the error message.

    Example:
        >>> from database.providers import get_all_provider_records, NoProvidersFoundError
        >>> from sqlalchemy.ext.asyncio import AsyncSession
        >>>
        >>> async def get_providers(session: AsyncSession):
        ...     try:
        ...         records = await get_all_provider_records(session)
        ...         return records
        ...     except NoProvidersFoundError:
        ...         print("No providers configured in the database")
        ...         return []
    """

    pass


async def get_all_provider_records(session: AsyncSession) -> list[ProviderRecord]:
    """Retrieves all Provider records from the database.

    This function queries the ProvidersT table and returns all Provider records as a list
    of ProviderRecord dataclass instances. If no Provider records exist in the database,
    it raises a NoProvidersFoundError.

    Args:
        session: The async SQLAlchemy session to use for database operations.

    Returns:
        list[ProviderRecord]: A list of ProviderRecord dataclass instances, each containing
            the attributes (id, name, base_url, internal, requires_api_key, langchain_con,
            mac_address, host) of each Provider from the database.

    Raises:
        NoProvidersFoundError: If no Provider records are found in the database.

    Example:
        >>> from database.providers import get_all_provider_records
        >>> from sqlalchemy.ext.asyncio import AsyncSession
        >>>
        >>> async def list_all_providers(session: AsyncSession):
        ...     try:
        ...         provider_records = await get_all_provider_records(session)
        ...         for record in provider_records:
        ...             print(f"Provider: {record.name} at {record.base_url}")
        ...         return provider_records
        ...     except NoProvidersFoundError as e:
        ...         print(f"Error: {e}")
        ...         return []
    """
    records = (await session.execute(select(ProvidersT))).scalars()
    if not records:
        raise NoProvidersFoundError()
    return [
        ProviderRecord(
            id=r.id,
            name=r.name,
            base_url=r.base_url,
            internal=r.internal,
            requires_api_key=r.requires_api_key,
            langchain_con=r.langchain_con,
            mac_address=r.mac_address,
            host=r.host,
        )
        for r in records
    ]
