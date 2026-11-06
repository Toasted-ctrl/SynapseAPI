from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas.agent_config import AgentConfigT


class AgentNotFoundOrNotOwnedError(ValueError):
    """Error raised if the Agent is not owned by the User or does not exist."""

    pass


async def ensure_agent_owned_by_user(
    user_id: UUID,
    agent_id: UUID,
    session: AsyncSession,
) -> None:
    agent_exists = (
        await session.execute(
            select(AgentConfigT.id).where(
                AgentConfigT.id == agent_id,
                AgentConfigT.owner_user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if agent_exists is None:
        raise AgentNotFoundOrNotOwnedError(
            f"Agent {agent_id} does not exist or is not owned by user {user_id}."
        )
