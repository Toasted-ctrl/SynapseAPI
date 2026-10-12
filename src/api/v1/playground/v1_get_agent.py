from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from auth.dep_verify_user import depends_verify_user, VerifiedUser
from core.logging import get_logger
from database.agent import AgentNotFoundError, get_agent_configuration
from database.session import get_db_session

from .v1_models import AgentConfigurationRead


log = get_logger()


router = APIRouter()


@router.get(
    "/playground/agent",
    tags=["Playground"],
    description="Creates and stores a new agent configuration.",
    response_model=AgentConfigurationRead,
    status_code=status.HTTP_201_CREATED,
)
async def get_agent(
    agent_id: str,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> AgentConfigurationRead:
    try:
        return await get_agent_configuration(
            session=session,
            user_id=user.id,
            agent_id=UUID(agent_id)
        )
    except AgentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while creating the agent."
        )