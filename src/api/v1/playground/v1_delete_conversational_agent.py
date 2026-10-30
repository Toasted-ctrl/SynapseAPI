from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.playground.models import AgentType
from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.agent import AgentNotFoundError, delete_agent_configuration
from database.session import get_db_session

log = get_logger()


router = APIRouter()


@router.delete(
    "/playground/conversational-agent",
    tags=["Playground / Conversational Agent"],
    description="Deletes a Conversational Agent's configuration.",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Agent does not exist or is not owned by the user."
        }
    },
)
async def delete_conversational_agent(
    agent_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> None:
    try:
        await delete_agent_configuration(
            session=session,
            user_id=user.id,
            agent_id=agent_id,
            agent_type=AgentType.CONVERSATIONAL,
        )
    except AgentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while deleting the agent.",
        )
