import uuid
from typing import Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.mcps import get_mcp_by_id

from .schemas.agent_config import AgentConfigT


log = get_logger()


CURRENT_SETTINGS_VERSION = 1


class AgentTools(BaseModel):
    mcp_ids: list[uuid.UUID] = Field(default_factory=list)


class AgentSettings(BaseModel):
    """Agent settings stored as JSON in AgentConfigT.settings.

    New fields must have a default so rows stored before the field existed still validate.
    Breaking changes (renames, removals, changed meaning) bump CURRENT_SETTINGS_VERSION
    and get an upgrade step in _upgrade.
    Unknown keys are dropped (Pydantic's default), so removed fields don't break old rows.
    Provider-specific sampling parameters (top_p, top_k, ...) go in model_params.
    """

    version: int = CURRENT_SETTINGS_VERSION
    system_prompt: str
    provider_name: str
    provider_model: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    tools: AgentTools = Field(default_factory=AgentTools)

    @model_validator(mode="before")
    @classmethod
    def _upgrade(cls, data):
        if not isinstance(data, dict):
            return data
        version = data.get("version", 1)
        if version > CURRENT_SETTINGS_VERSION:
            raise ValueError(f"Unsupported agent settings version '{version}'")
        return data


@dataclass(frozen=True)
class AgentConfiguration:
    id: uuid.UUID
    name: str
    settings: AgentSettings


def _to_configuration(agent: AgentConfigT) -> AgentConfiguration:
    return AgentConfiguration(
        id=agent.id,
        name=agent.name,
        settings=AgentSettings.model_validate(agent.settings)
    )


async def _validate_references(session: AsyncSession, settings: AgentSettings) -> None:
    """Raises ValueError if the settings reference records that do not exist."""
    for mcp_id in settings.tools.mcp_ids:
        await get_mcp_by_id(session=session, mcp_id=mcp_id)


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
        raise ValueError(f"Could not locate agent with id '{agent_id}'")
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
        raise ValueError(f"An agent named '{name}' already exists")


async def create_agent(
    session: AsyncSession,
    user_id: uuid.UUID,
    name: str,
    settings: AgentSettings
) -> AgentConfiguration:
    """Stores a new agent configuration for the user."""
    await _ensure_name_available(session=session, user_id=user_id, name=name)
    await _validate_references(session=session, settings=settings)
    agent = AgentConfigT(
        owner_user_id=user_id,
        name=name,
        settings=settings.model_dump(mode="json")
    )
    session.add(agent)
    await session.flush()
    log.debug(f"Created agent '{agent.id}' for user '{user_id}'")
    return _to_configuration(agent)


async def get_agent(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID
) -> AgentConfiguration:
    """Fetches an agent configuration owned by the user."""
    agent = await _get_agent_row(session=session, user_id=user_id, agent_id=agent_id)
    return _to_configuration(agent)


async def list_agents(session: AsyncSession, user_id: uuid.UUID) -> list[AgentConfiguration]:
    """Returns all agent configurations owned by the user."""
    agents = (await session.scalars(
        select(AgentConfigT)
        .where(AgentConfigT.owner_user_id == user_id)
        .order_by(AgentConfigT.name)
    )).all()
    log.debug(f"Found {len(agents)} agents for user '{user_id}'")
    return [_to_configuration(agent) for agent in agents]


async def update_agent(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID,
    settings: AgentSettings,
    name: str | None = None
) -> AgentConfiguration:
    """Replaces an agent's settings, and optionally renames it."""
    agent = await _get_agent_row(session=session, user_id=user_id, agent_id=agent_id)
    if name is not None and name != agent.name:
        await _ensure_name_available(session=session, user_id=user_id, name=name)
        agent.name = name
    await _validate_references(session=session, settings=settings)
    agent.settings = settings.model_dump(mode="json")
    await session.flush()
    log.debug(f"Updated agent '{agent_id}' for user '{user_id}'")
    return _to_configuration(agent)


async def delete_agent(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID
) -> None:
    """Deletes an agent configuration owned by the user."""
    agent = await _get_agent_row(session=session, user_id=user_id, agent_id=agent_id)
    await session.delete(agent)
    await session.flush()
    log.debug(f"Deleted agent '{agent_id}' for user '{user_id}'")