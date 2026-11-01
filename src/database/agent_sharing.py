from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

from .agent_ownership import (
    AgentNotFoundOrNotOwnedError,
    ensure_agent_owned_by_user,
)
from .schemas.agent_config import AgentConfigT
from .schemas.agent_sharing import AgentSharingT
from .schemas.persons_users import UsersT


class AgentAlreadySharedError(ValueError):
    """The Agent is already shared with the User."""

    pass


class AgentOwnedByUserError(ValueError):
    """The Agent attempted to be shared with the User is already owned by the User."""

    pass


class UserNotFoundError(ValueError):
    """The User does not exist."""

    pass


log = get_logger()


async def share_agent_with_user_id(
    owner_user_id: UUID, agent_id: UUID, share_with_user_id: UUID, session: AsyncSession
) -> AgentSharingT:
    """Shares an Agent with a User."""
    agent = (
        await session.execute(
            select(AgentConfigT).where(
                AgentConfigT.owner_user_id == owner_user_id, AgentConfigT.id == agent_id
            )
        )
    ).scalar_one_or_none()
    if not agent:
        raise AgentNotFoundOrNotOwnedError(
            f"Agent {agent_id} does not exist or is not owned by user {owner_user_id}."
        )
    if owner_user_id == share_with_user_id:
        raise AgentOwnedByUserError("Agents cannot be shared with their owner.")

    user_exists = (
        await session.execute(select(UsersT).where(UsersT.id == share_with_user_id))
    ).scalar_one_or_none()
    if not user_exists:
        raise UserNotFoundError(f"User {share_with_user_id} does not exist.")

    share = AgentSharingT(
        agent_id=agent_id, user_id=share_with_user_id, owner_user_id=owner_user_id
    )
    session.add(share)
    try:
        await session.flush()
    except IntegrityError as e:
        log.debug(
            f"Agent {agent_id} already shared with {share_with_user_id}: {str(e)}"
        )
        raise AgentAlreadySharedError(
            f"Agent {agent_id} already shared with {share_with_user_id}."
        )

    await session.refresh(share)
    return share


async def get_shared_users_by_agent_id(
    owner_user_id: UUID, agent_id: UUID, session: AsyncSession
) -> list[UUID]:
    """Retrieves a list of User ids with which the Agent is shared. Requires the Agent to be owned by the indicated User."""
    await ensure_agent_owned_by_user(
        user_id=owner_user_id, agent_id=agent_id, session=session
    )
    users = await session.scalars(
        select(AgentSharingT.user_id).where(AgentSharingT.agent_id == agent_id)
    )
    return list(users.all())


async def delete_shared_agent_from_user(
    owner_user_id: UUID,
    shared_with_user_id: UUID,
    agent_id: UUID,
    session: AsyncSession,
) -> bool:
    """Removes shared access to an Agent from a User."""
    await ensure_agent_owned_by_user(
        user_id=owner_user_id, agent_id=agent_id, session=session
    )
    sharing = (
        await session.execute(
            select(AgentSharingT).where(
                AgentSharingT.agent_id == agent_id,
                AgentSharingT.owner_user_id == owner_user_id,
                AgentSharingT.user_id == shared_with_user_id,
            )
        )
    ).scalar_one_or_none()
    if not sharing:
        raise AgentNotFoundOrNotOwnedError(
            f"Agent {agent_id} not shared with User {shared_with_user_id}"
        )
    await session.delete(sharing)
    await session.flush()
    return True
