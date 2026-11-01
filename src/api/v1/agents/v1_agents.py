from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.playground.models import AgentType
from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.logging import get_logger
from database.agent import list_agents_owned_by_user
from database.agent_sharing import list_agents_shared_with_user
from database.session import get_db_session

from .models import AgentListRead

log = get_logger()

router = APIRouter()


@router.get(
    "/agents",
    description="Retrieves a list of Agents available to the User.",
    response_model=AgentListRead,
    tags=["Agents"],
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid access_type parameter."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected error occurred while retrieving agents."
        },
    },
)
async def get_agents(
    access_type: str,
    agent_type: AgentType = AgentType.CONVERSATIONAL,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> AgentListRead:
    if access_type not in ["User Owned", "All"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported access_type: {access_type}. Valid values are 'User Owned' or 'All'.",
        )
    try:
        agents = await list_agents_owned_by_user(
            session=session, user_id=user.id, agent_type=agent_type
        )
        if access_type == "All":
            agents += await list_agents_shared_with_user(
                session=session, user_id=user.id, agent_type=agent_type
            )
        return AgentListRead(agents=agents)
    except Exception as e:
        log.exception(f"Error retrieving agents for user {user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while retrieving agents.",
        )
