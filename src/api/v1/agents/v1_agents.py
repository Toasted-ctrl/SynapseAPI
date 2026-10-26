from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from database.agent import list_agents_owned_by_user
from database.session import get_db_session

from .models import AgentListRead

router = APIRouter()


@router.get(
    "/agents",
    description="Retrieves a list of Agents available to the User.",
    response_model=AgentListRead,
    tags=["Agents"],
)
async def get_agents(
    access_type: str,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> AgentListRead:
    if access_type == "User Owned" or access_type == "All":
        return AgentListRead(
            agents=await list_agents_owned_by_user(session=session, user_id=user.id)
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=f"Unsupported access_type: '{access_type}'",
        )
