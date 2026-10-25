from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.schemas.vector_store import VectorStoreSettingsT
from database.schemas.vector_store_collections import VectorStoreCollectionT

log = get_logger()


@dataclass(frozen=True)
class VectorStoreConfig:
    vs_collection_name: str
    vs_encrypted_api_key: str
    vs_vendor: str
    vs_base_url: str
    vs_port: int
    e_dimensions: int
    e_provider: str
    e_model: str
    e_api_key: str
    scope: str
    required_filters: list


async def get_vector_store_settings(
    scope: str, session: AsyncSession
) -> VectorStoreConfig:
    """Fetches and returns the Vector Store collection and configuration details,
    to enable building a Vector Store client."""

    log.debug(f"Retrieving Vector Store settings for scope '{scope}' ...")
    col: VectorStoreCollectionT | None = (
        await session.execute(
            select(VectorStoreCollectionT).where(VectorStoreCollectionT.scope == scope)
        )
    ).scalar_one_or_none()

    if not col:
        log.info(f"No Vector Store found with scope '{scope}'.")
        raise ValueError(f"Vector Store with name '{scope}' does not exist ...")

    vs: VectorStoreSettingsT | None = (
        await session.execute(
            select(VectorStoreSettingsT).where(
                VectorStoreSettingsT.id == col.vector_store_id
            )
        )
    ).scalar_one_or_none()

    if not vs:
        log.error(f"No Vector Store instance found with id '{col.vector_store_id}'.")
        raise ValueError(
            f"Vector Store instance with id '{col.vector_store_id}' does not exist."
        )

    log.debug("Returning Vector Store client.")
    return VectorStoreConfig(
        vs_collection_name=col.name,
        e_dimensions=col.e_dimensions,
        e_provider=col.e_provider,
        e_model=col.e_model,
        e_api_key=col.e_api_key,
        vs_encrypted_api_key=vs.encrypted_api_key,
        scope=col.scope,
        vs_vendor=vs.vendor,
        vs_base_url=vs.base_url,
        vs_port=vs.port,
        required_filters=col.required_filters,
    )
