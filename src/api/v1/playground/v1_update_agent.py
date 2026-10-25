from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from database.agent import (
    AgentNameTakenError,
    AgentNotFoundError,
    update_agent_configuration,
)
from database.session import get_db_session

from .models import AgentConfigurationCreate, AgentConfigurationRead

router = APIRouter()


@router.put(
    "/playground/agent",
    tags=["Playground"],
    description="Updates the configuration of an Agent.",
    response_model=AgentConfigurationRead,
)
async def update_agent(
    payload: AgentConfigurationCreate,
    agent_id: UUID,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> AgentConfigurationRead:
    try:
        return await update_agent_configuration(
            session=session, user_id=user.id, agent_id=agent_id, config=payload
        )
    except AgentNameTakenError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Unable to update agent '{agent_id}': {e!s}",
        )
    except AgentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to update agent '{agent_id}': {e!s}",
        )
