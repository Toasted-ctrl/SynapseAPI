from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_user import depends_verify_user, VerifiedUser
from core.logging import get_logger
from database.agent import AgentNameTakenError, AgentReferenceError, create_agent_configuration
from database.session import get_db_session

from .models import AgentConfigurationCreate, AgentConfigurationRead


log = get_logger()


router = APIRouter()


@router.post(
    "/playground/agent",
    tags=["Playground"],
    description="Creates and stores a new agent configuration.",
    response_model=AgentConfigurationRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {"description": "An agent with this name already exists."},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid configuration, e.g. an MCP server that does not exist."}
    }
)
async def create_agent(
    payload: AgentConfigurationCreate,
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user)
) -> AgentConfigurationRead:
    try:
        return await create_agent_configuration(
            session=session,
            user_id=user.id,
            config=payload
        )
    except AgentNameTakenError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except AgentReferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e)
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while creating the agent."
        )