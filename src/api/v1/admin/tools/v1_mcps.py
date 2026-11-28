from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dep_verify_admin import depends_verify_admin
from database.mcps import MCPConfig, MCPNotFoundError, delete_mcp_by_id, insert_mcp
from database.session import get_db_session

from .models import MCPConfigurationCreate, MCPConfigurationRead

router = APIRouter()


@router.post(
    "/admin/tools/mcp",
    tags=["Admin / Tools"],
    description="Adds a new MCP tool to the database. Requires Admin permissions.",
    response_model=MCPConfigurationRead,
    dependencies=[Depends(depends_verify_admin)],
)
async def add_tools_mcp(
    payload: MCPConfigurationCreate,
    session: AsyncSession = Depends(get_db_session),
) -> MCPConfigurationRead:
    mcp = await insert_mcp(
        session=session, mcp_config=MCPConfig(**payload.model_dump())
    )
    return MCPConfigurationRead(**asdict(mcp))


@router.delete(
    "/admin/tools/mcp",
    tags=["Admin / Tools"],
    description="Deletes an MCP from the database. Requires Admin permissions.",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(depends_verify_admin)],
)
async def delete_mcp(
    mcp_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    try:
        await delete_mcp_by_id(session=session, mcp_id=mcp_id)
    except MCPNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
