from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.agent import AgentNotFoundError, get_decision_agent_configuration
from database.session import get_db_session

from .models import DecisionAgentConfigurationRead

log = get_logger()


router = APIRouter()


@router.get(
    "/playground/decision-agent",
    tags=["Playground"],
    description="Returns a decision agent configuration.",
    response_model=DecisionAgentConfigurationRead,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Decision agent does not exist or is not owned by the user."
        }
    },
)
async def get_decision_agent(
    agent_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> DecisionAgentConfigurationRead:
    try:
        return await get_decision_agent_configuration(
            session=session, user_id=user.id, agent_id=agent_id
        )
    except AgentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while retrieving the decision agent.",
        )
