from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.agent import AgentNameTakenError, create_decision_agent_configuration
from database.session import get_db_session

from .models import DecisionAgentConfigurationCreate, DecisionAgentConfigurationRead

log = get_logger()


router = APIRouter()


@router.post(
    "/playground/decision-agent",
    tags=["Playground"],
    description="Creates and stores a new decision agent configuration.",
    response_model=DecisionAgentConfigurationRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "An agent with this name already exists."
        },
    },
)
async def create_decision_agent(
    payload: DecisionAgentConfigurationCreate,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> DecisionAgentConfigurationRead:
    try:
        return await create_decision_agent_configuration(
            session=session, user_id=user.id, config=payload
        )
    except AgentNameTakenError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while creating the decision agent.",
        )
