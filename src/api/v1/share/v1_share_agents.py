from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import VerifiedUser, depends_verify_user
from database.agent_sharing import (
    AgentAlreadySharedError,
    AgentNotFoundOrNotOwnedError,
    AgentOwnedByUserError,
    UserNotFoundError,
    get_shared_users_by_agent_id,
    share_agent_with_user_id,
)
from database.session import get_db_session

from .models import GrantAgentAccessRead, UserAgentAccessRead

router = APIRouter()


@router.get(
    "/share/agent",
    tags=["Share"],
    description=(
        "Retrieves User ids which have access to a specific Agent. "
        "Requester must be the owner of the Agent."
    ),
    response_model=UserAgentAccessRead,
)
async def get_agent_shared_users(
    agent_id: UUID,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> UserAgentAccessRead:
    try:
        users = await get_shared_users_by_agent_id(
            owner_user_id=user.id, agent_id=agent_id, session=session
        )
        return UserAgentAccessRead(users=users)
    except AgentNotFoundOrNotOwnedError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found or not owned by user {user.id}",
        )


@router.post(
    "/share/agent",
    tags=["Share"],
    description=(
        "Shares an Agent with another User. Requester must be the owner of the Agent."
    ),
    response_model=GrantAgentAccessRead,
)
async def share_agent(
    agent_id: UUID,
    share_with_user_id: UUID,
    user: VerifiedUser = Depends(depends_verify_user),
    session: AsyncSession = Depends(get_db_session),
) -> GrantAgentAccessRead:
    try:
        returned_agent_id = await share_agent_with_user_id(
            owner_user_id=user.id,
            agent_id=agent_id,
            share_with_user_id=share_with_user_id,
            session=session,
        )
        return GrantAgentAccessRead(id=returned_agent_id)
    except AgentNotFoundOrNotOwnedError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found or not owned by user {user.id}",
        )
    except AgentOwnedByUserError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot share an Agent with its owner",
        )
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {share_with_user_id} not found",
        )
    except AgentAlreadySharedError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent {agent_id} already shared with user {share_with_user_id}",
        )


# TODO: Add function to revoke sharing access.
