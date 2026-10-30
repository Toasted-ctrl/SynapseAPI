import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.playground.models import (
    AgentConfigurationCreate,
    AgentConfigurationRead,
    AgentType,
    DecisionAgentConfigurationCreate,
    DecisionAgentConfigurationRead,
)
from core.logging import get_logger
from database.mcps import get_mcp_by_id
from security.encryption import decrypt, encrypt

from .schemas.agent_config import AgentConfigT

log = get_logger()


# Each configuration minus COLUMN_FIELDS is stored as JSON in AgentConfigT.settings, so:
# New fields must have a default so rows stored before the field existed still validate.
# Breaking changes (renames, removals, changed meaning) bump the type's settings version
# and get an upgrade step in _upgrade_settings.
# Unknown keys are dropped (Pydantic's default), so removed fields don't break old rows.
# Version 2 introduced encryption for Conversational settings data.
CURRENT_SETTINGS_VERSION = 2
DECISION_SETTINGS_VERSION = 1

NAME_UNIQUE_CONSTRAINT = "uq_agent_configs_owner_name"

# Fields stored in their own AgentConfigT columns rather than in settings.
# description is stored encrypted in AgentConfigT.description.
COLUMN_FIELDS = {"name", "description"}


class AgentNotFoundError(ValueError):
    """The agent does not exist or is not owned by the user."""


class AgentNameTakenError(ValueError):
    """The user already has an agent with this name."""


class AgentReferenceError(ValueError):
    """The configuration references a record (e.g. an MCP server) that does not exist."""


def _upgrade_settings(settings: dict[str, Any], current_version: int) -> dict[str, Any]:
    """Decrypts encrypted settings and brings stored settings JSON up to current_version."""
    # Handle encrypted data
    if settings.get("__encrypted__"):
        decrypted_json = decrypt(settings["data"])
        settings = json.loads(decrypted_json)

    version = settings.get("version", 1)
    if version > current_version:
        raise ValueError(f"Unsupported agent settings version '{version}'")
    return settings


def _to_settings(config: BaseModel, version: int) -> dict[str, Any]:
    """Serialises and encrypts a configuration to the JSON stored in AgentConfigT.settings."""
    settings_dict = {
        "version": version,
        **config.model_dump(mode="json", exclude=COLUMN_FIELDS),
    }
    # Store as encrypted string in a wrapper dict
    return {"__encrypted__": True, "data": encrypt(json.dumps(settings_dict))}


def _encrypt_description(description: str | None) -> str | None:
    return encrypt(description) if description else None


def _decrypt_description(encrypted_description: str | None) -> str | None:
    return decrypt(encrypted_description) if encrypted_description else None


def _row_fields(agent: AgentConfigT, current_version: int) -> dict[str, Any]:
    """The decrypted settings and column fields of an AgentConfigT row."""
    return {
        **_upgrade_settings(agent.settings, current_version),
        "id": agent.id,
        "name": agent.name,
        "description": _decrypt_description(agent.description),
    }


def _to_configuration(
    agent: AgentConfigT, is_encrypted: bool = False
) -> AgentConfigurationRead:
    """Converts an AgentConfigT row to an AgentConfigurationRead, with optional encryption flag."""
    return AgentConfigurationRead.model_validate(
        {
            **_row_fields(agent, CURRENT_SETTINGS_VERSION),
            "is_encrypted": is_encrypted,
        }
    )


def _to_decision_configuration(agent: AgentConfigT) -> DecisionAgentConfigurationRead:
    """Converts an AgentConfigT row to a DecisionAgentConfigurationRead."""
    return DecisionAgentConfigurationRead.model_validate(
        _row_fields(agent, DECISION_SETTINGS_VERSION)
    )


async def _validate_references(
    session: AsyncSession, config: AgentConfigurationCreate
) -> None:
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
    agent_id: uuid.UUID,
    agent_type: AgentType,
) -> AgentConfigT:
    agent = (
        await session.execute(
            select(AgentConfigT).where(
                AgentConfigT.id == agent_id,
                AgentConfigT.owner_user_id == user_id,
                AgentConfigT.agent_type == agent_type,
            )
        )
    ).scalar_one_or_none()
    if not agent:
        log.debug(
            f"Could not locate '{agent_type}' agent with id '{agent_id}' for user '{user_id}'"
        )
        raise AgentNotFoundError(f"Could not locate agent with id '{agent_id}'")
    return agent


async def _ensure_name_available(
    session: AsyncSession, user_id: uuid.UUID, name: str
) -> None:
    existing = (
        await session.execute(
            select(AgentConfigT.id).where(
                AgentConfigT.owner_user_id == user_id, AgentConfigT.name == name
            )
        )
    ).scalar_one_or_none()
    if existing:
        raise AgentNameTakenError(f"An agent named '{name}' already exists")


async def _create_agent_row(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_type: AgentType,
    name: str,
    description: str | None,
    settings: dict[str, Any],
) -> AgentConfigT:
    agent = AgentConfigT(
        owner_user_id=user_id,
        name=name,
        description=_encrypt_description(description),
        agent_type=agent_type,
        settings=settings,
    )
    session.add(agent)
    await _flush_or_name_taken(session=session, name=name)
    log.debug(f"Created '{agent_type}' agent '{agent.id}' for user '{user_id}'")
    return agent


async def _rename_agent_row(
    session: AsyncSession, user_id: uuid.UUID, agent: AgentConfigT, name: str
) -> None:
    if name != agent.name:
        await _ensure_name_available(session=session, user_id=user_id, name=name)
        agent.name = name


async def _update_agent_row(
    session: AsyncSession,
    agent: AgentConfigT,
    description: str | None,
    settings: dict[str, Any],
) -> None:
    agent.description = _encrypt_description(description)
    agent.settings = settings
    await _flush_or_name_taken(session=session, name=agent.name)
    log.debug(f"Updated '{agent.agent_type}' agent '{agent.id}'")


# Conversational Agents


async def create_agent_configuration(
    session: AsyncSession, user_id: uuid.UUID, config: AgentConfigurationCreate
) -> AgentConfigurationRead:
    """Stores a new Conversational agent configuration for the user."""
    await _ensure_name_available(session=session, user_id=user_id, name=config.name)
    await _validate_references(session=session, config=config)
    agent = await _create_agent_row(
        session=session,
        user_id=user_id,
        agent_type=AgentType.CONVERSATIONAL,
        name=config.name,
        description=config.description,
        settings=_to_settings(config, CURRENT_SETTINGS_VERSION),
    )
    # New agents are always encrypted
    return _to_configuration(agent, is_encrypted=True)


async def get_agent_configuration(
    session: AsyncSession, user_id: uuid.UUID, agent_id: uuid.UUID
) -> AgentConfigurationRead:
    """Fetches a Conversational agent configuration owned by the user."""
    agent = await _get_agent_row(
        session=session,
        user_id=user_id,
        agent_id=agent_id,
        agent_type=AgentType.CONVERSATIONAL,
    )
    # Check encryption status from raw settings BEFORE decryption
    is_encrypted = agent.settings.get("__encrypted__", False)
    return _to_configuration(agent, is_encrypted=is_encrypted)


async def update_agent_configuration(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID,
    config: AgentConfigurationCreate,
) -> AgentConfigurationRead:
    """Replaces a Conversational agent's configuration, including its name."""
    agent = await _get_agent_row(
        session=session,
        user_id=user_id,
        agent_id=agent_id,
        agent_type=AgentType.CONVERSATIONAL,
    )
    await _rename_agent_row(
        session=session, user_id=user_id, agent=agent, name=config.name
    )
    await _validate_references(session=session, config=config)
    await _update_agent_row(
        session=session,
        agent=agent,
        description=config.description,
        settings=_to_settings(config, CURRENT_SETTINGS_VERSION),
    )
    # Updated agents are now encrypted
    return _to_configuration(agent, is_encrypted=True)


# Decision Agents


async def create_decision_agent_configuration(
    session: AsyncSession, user_id: uuid.UUID, config: DecisionAgentConfigurationCreate
) -> DecisionAgentConfigurationRead:
    """Stores a new Decision agent configuration for the user."""
    await _ensure_name_available(session=session, user_id=user_id, name=config.name)
    agent = await _create_agent_row(
        session=session,
        user_id=user_id,
        agent_type=AgentType.DECISION,
        name=config.name,
        description=config.description,
        settings=_to_settings(config, DECISION_SETTINGS_VERSION),
    )
    return _to_decision_configuration(agent)


async def get_decision_agent_configuration(
    session: AsyncSession, user_id: uuid.UUID, agent_id: uuid.UUID
) -> DecisionAgentConfigurationRead:
    """Fetches a Decision agent configuration owned by the user."""
    agent = await _get_agent_row(
        session=session,
        user_id=user_id,
        agent_id=agent_id,
        agent_type=AgentType.DECISION,
    )
    return _to_decision_configuration(agent)


async def update_decision_agent_configuration(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID,
    config: DecisionAgentConfigurationCreate,
) -> DecisionAgentConfigurationRead:
    """Replaces a Decision agent's configuration, including its name."""
    agent = await _get_agent_row(
        session=session,
        user_id=user_id,
        agent_id=agent_id,
        agent_type=AgentType.DECISION,
    )
    await _rename_agent_row(
        session=session, user_id=user_id, agent=agent, name=config.name
    )
    await _update_agent_row(
        session=session,
        agent=agent,
        description=config.description,
        settings=_to_settings(config, DECISION_SETTINGS_VERSION),
    )
    return _to_decision_configuration(agent)


# All Agents


@dataclass(frozen=True)
class AgentConfigurationLimited:
    id: uuid.UUID
    name: str


async def list_agents_owned_by_user(
    session: AsyncSession, user_id: uuid.UUID, agent_type: AgentType
) -> list[AgentConfigurationLimited]:
    """Returns all agents of the given type owned by the user (id and name)."""
    agents = (
        await session.execute(
            select(AgentConfigT.id, AgentConfigT.name)
            .where(
                AgentConfigT.owner_user_id == user_id,
                AgentConfigT.agent_type == agent_type,
            )
            .order_by(AgentConfigT.name)
        )
    ).all()
    log.debug(f"Found {len(agents)} '{agent_type}' agents for user '{user_id}'")
    return [AgentConfigurationLimited(id=agent.id, name=agent.name) for agent in agents]


async def delete_agent_configuration(
    session: AsyncSession,
    user_id: uuid.UUID,
    agent_id: uuid.UUID,
    agent_type: AgentType,
) -> None:
    """Deletes an agent configuration of the given type owned by the user."""
    agent = await _get_agent_row(
        session=session, user_id=user_id, agent_id=agent_id, agent_type=agent_type
    )
    await session.delete(agent)
    await session.flush()
    log.debug(f"Deleted '{agent_type}' agent '{agent_id}' for user '{user_id}'")
