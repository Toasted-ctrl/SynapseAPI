import uuid
from typing import Any
from dataclasses import dataclass
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.playground.models import AgentConfigurationCreate, AgentConfigurationRead
from core.logging import get_logger
from database.mcps import get_mcp_by_id

from .schemas.agent_config import AgentConfigT


log = get_logger()


# AgentConfigurationCreate minus name is stored as JSON in AgentConfigT.settings, so:
# New fields must have a default so rows stored before the field existed still validate.
# Breaking changes (renames, removals, changed meaning) bump CURRENT_SETTINGS_VERSION
# and get an upgrade step in _upgrade_settings.
# Unknown keys are dropped (Pydantic's default), so removed fields don't break old rows.
CURRENT_SETTINGS_VERSION = 1

NAME_UNIQUE_CONSTRAINT = "uq_agent_configs_owner_name"


class AgentNotFoundError(ValueError):
    """The agent does not exist or is not owned by the user."""


class AgentNameTakenError(ValueError):
    """The user already has an agent with this name."""


class AgentReferenceError(ValueError):
    """The configuration references a record (e.g. an MCP server) that does not exist."""


def _upgrade_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Brings stored settings JSON up to CURRENT_SETTINGS_VERSION."""
    version = settings.get("version", 1)
    if version > CURRENT_SETTINGS_VERSION:
        raise ValueError(f"Unsupported agent settings version '{version}'")
    return settings


def _to_settings(config: AgentConfigurationCreate) -> dict[str, Any]:
    """Serialises a configuration to the JSON stored in AgentConfigT.settings."""
    return {
        "version": CURRENT_SETTINGS_VERSION,
        **config.model_dump(mode="json", exclude={"name"})
    }


def _to_configuration(agent: AgentConfigT) -> AgentConfigurationRead:
    return AgentConfigurationRead.model_validate({
        **_upgrade_settings(agent.settings),
        "id": agent.id,
        "name": agent.name
    })


async def _validate_references(session: AsyncSession, config: AgentConfigurationCreate) -> None:
    """Raises AgentReferenceError if the configuration references records that do not exist."""
    for mcp_id in config.tools.mcp_ids:
        try:
            await get_mcp_by_id(session=session, mcp_id=mcp_id)
        except ValueError as e:
            raise AgentReferenceError(str(e)) from e


async def _flush_or_name_taken(session: AsyncSession, name: str) -> None:
    """Flushes, turning a unique name violation (e.g. from a concurrent request that
    passed _ensure_name_available at the same time) into AgentNameTakenError."""
    try:
        await session.flush()
    except IntegrityError as e:
        if NAME_UNIQUE_CONSTRAINT in str(e.orig):
            raise AgentNameTakenError(f"An agent named '{name}' already exists") from e
        raise


async def _get_agent_row(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID
) -> AgentConfigT:
    agent = (await session.execute(
        select(AgentConfigT)
        .where(
            AgentConfigT.id == agent_id,
            AgentConfigT.owner_user_id == user_id
        )
    )).scalar_one_or_none()
    if not agent:
        log.debug(f"Could not locate agent with id '{agent_id}' for user '{user_id}'")
        raise AgentNotFoundError(f"Could not locate agent with id '{agent_id}'")
    return agent


async def _ensure_name_available(
    session: AsyncSession,
    user_id: uuid.UUID,
    name: str
) -> None:
    existing = (await session.execute(
        select(AgentConfigT.id)
        .where(
            AgentConfigT.owner_user_id == user_id,
            AgentConfigT.name == name
        )
    )).scalar_one_or_none()
    if existing:
        raise AgentNameTakenError(f"An agent named '{name}' already exists")


async def create_agent_configuration(
    session: AsyncSession,
    user_id: uuid.UUID,
    config: AgentConfigurationCreate
) -> AgentConfigurationRead:
    """Stores a new agent configuration for the user."""
    await _ensure_name_available(session=session, user_id=user_id, name=config.name)
    await _validate_references(session=session, config=config)
    agent = AgentConfigT(
        owner_user_id=user_id,
        name=config.name,
        settings=_to_settings(config)
    )
    session.add(agent)
    await _flush_or_name_taken(session=session, name=config.name)
    log.debug(f"Created agent '{agent.id}' for user '{user_id}'")
    return _to_configuration(agent)


async def get_agent_configuration(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID
) -> AgentConfigurationRead:
    """Fetches an agent configuration owned by the user."""
    agent = await _get_agent_row(session=session, user_id=user_id, agent_id=agent_id)
    return _to_configuration(agent)


@dataclass(frozen=True)
class AgentConfigurationLimited:
    id: uuid.UUID
    name: str


async def list_agents_owned_by_user(
    session: AsyncSession,
    user_id: uuid.UUID
) -> list[AgentConfigurationLimited]:
    """Returns all agents owned by the user (id and name)."""
    agents = (await session.execute(
        select(AgentConfigT.id, AgentConfigT.name)
        .where(AgentConfigT.owner_user_id == user_id)
        .order_by(AgentConfigT.name)
    )).all()
    log.debug(f"Found {len(agents)} agents for user '{user_id}'")
    return [
        AgentConfigurationLimited(
            id=agent.id,
            name=agent.name
        )
        for agent in agents
    ]


async def update_agent_configuration(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID,
    config: AgentConfigurationCreate
) -> AgentConfigurationRead:
    """Replaces an agent's configuration, including its name."""
    agent = await _get_agent_row(session=session, user_id=user_id, agent_id=agent_id)
    if config.name != agent.name:
        await _ensure_name_available(session=session, user_id=user_id, name=config.name)
        agent.name = config.name
    await _validate_references(session=session, config=config)
    agent.settings = _to_settings(config)
    await _flush_or_name_taken(session=session, name=config.name)
    log.debug(f"Updated agent '{agent_id}' for user '{user_id}'")
    return _to_configuration(agent)


async def delete_agent_configuration(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID
) -> None:
    """Deletes an agent configuration owned by the user."""
    agent = await _get_agent_row(session=session, user_id=user_id, agent_id=agent_id)
    await session.delete(agent)
    await session.flush()
    log.debug(f"Deleted agent '{agent_id}' for user '{user_id}'")