from dataclasses import asdict, dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

from .schemas.mcp import MCPsT

log = get_logger()


@dataclass(frozen=True)
class MCP:
    id: UUID
    name: str


async def get_mcps(session: AsyncSession) -> list[MCP]:
    """Returns a list of available MCPs."""
    mcps = (await session.scalars(select(MCPsT))).all()
    log.debug(f"Found {len(mcps)} MCP configurations.")
    return [MCP(id=mcp.id, name=mcp.name) for mcp in mcps]


@dataclass(frozen=True)
class MCPConfig:
    name: str
    url: str
    transport: str
    id: UUID | None = None


async def get_mcp_by_id(session: AsyncSession, mcp_id: UUID) -> MCPConfig:
    """Returns MCP configuration based on MCP id."""
    mcp = (
        await session.execute(select(MCPsT).where(MCPsT.id == mcp_id))
    ).scalar_one_or_none()
    if mcp is None:
        raise ValueError(f"MCP with ID {mcp_id} does not exist.")
    log.debug(f"Fetched configuration details for MCP '{mcp_id}'.")
    return MCPConfig(id=mcp.id, name=mcp.name, url=mcp.url, transport=mcp.transport)


async def insert_mcp(session: AsyncSession, mcp_config: MCPConfig) -> UUID:
    mcp = MCPsT(**asdict(mcp_config))
    session.add(mcp)
    await session.flush()
    await session.refresh(mcp)
    log.debug(f"Inserted MCP {mcp.id}")
    return mcp.id
