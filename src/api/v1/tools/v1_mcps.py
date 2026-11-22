from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_cache.decorator import cache
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import VerifiedAdmin, depends_verify_admin
from auth.dep_verify_user import VerifiedUser, depends_verify_user
from core.cache import cache_key_builder
from database.mcps import MCPConfig, get_mcps, insert_new_mcp
from database.session import get_db_session

from .models import MCPConfigurationCreate, MCPConfigurationRead, McpsRead

router = APIRouter()


@router.get(
    "/tools/mcp",
    tags=["Tools"],
    description="Retrieves a list of currently configured MCP servers.",
    response_model=McpsRead,
)
@cache(expire=300, key_builder=cache_key_builder)
async def get_tools_mcp(
    session: AsyncSession = Depends(get_db_session),
    user: VerifiedUser = Depends(depends_verify_user),
) -> McpsRead:
    mcps = await get_mcps(session=session)
    if not mcps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No MCP servers found"
        )
    return McpsRead(mcps=[asdict(mcp) for mcp in mcps])


@router.post(
    "/tools/mcp",
    tags=["Tools"],
    description=("Adds a new MCP tool to the database. Requires Admin permissions."),
    response_model=MCPConfigurationRead,
)
async def add_tools_mcp(
    payload: MCPConfigurationCreate,
    session: AsyncSession = Depends(get_db_session),
    admin: VerifiedAdmin = Depends(depends_verify_admin),
) -> MCPConfigurationRead:
    mcp = await insert_new_mcp(
        session=session, mcp_config=MCPConfig(**payload.model_dump())
    )
    return MCPConfigurationRead(**asdict(mcp))
