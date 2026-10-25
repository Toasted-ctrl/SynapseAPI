import uuid
from dataclasses import dataclass
from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.schemas.providers import ProvidersT
from database.user_keys import get_user_active_keys

log = get_logger()


@dataclass(frozen=True)
class Provider:
    id: uuid.UUID
    name: str
    base_url: str
    internal: bool
    requires_api_key: bool
    langchain_con: str
    api_key_configured: bool | None = None
    mac_address: str | None = None


@dataclass(frozen=True)
class ProviderConfiguration:
    id: uuid.UUID
    name: str
    base_url: str
    langchain_con: str
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
            langchain_con=existing.langchain_con,
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
        langchain_con=new_provider.langchain_con,
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
            langchain_con=p.langchain_con,
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
            langchain_con=p.langchain_con,
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
            langchain_con=p.langchain_con,
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
