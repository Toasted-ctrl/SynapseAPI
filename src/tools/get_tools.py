import uuid

from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.providers import UserProviderRegistry

from .mcp_tools import get_mcp_tools
from .search_vs import get_vs_search_tool
from .skill_discovery import get_user_skill_discovery_tool
from .web_search import get_web_search_tool

# TODO: This is primarily for USER tools currently. Possible rename all get 'x' tool functions?


log = get_logger()


async def get_agent_tools(
    mcp_ids: list[uuid.UUID],
    session: AsyncSession,
    web_search: bool,
    user_vs_files: bool,
    user_vs_memories: bool,
    user_vs_skills: bool,
    user_id: uuid.UUID,
    pr: UserProviderRegistry,
) -> list[BaseTool]:
    """Collects all tools required by the User for the Agent invocation."""
    log.debug("Collecting Agent tools ...")
    tools = []
    if mcp_ids:
        tools += await get_mcp_tools(session=session, mcp_ids=mcp_ids)
    if web_search:
        tools += await get_web_search_tool()
    vs_scopes = []
    if user_vs_files:
        vs_scopes.append("user_vs_files")
    if user_vs_memories:
        vs_scopes.append("user_vs_memories")
    for scope in vs_scopes:
        tools.append(
            await get_vs_search_tool(
                session=session, user_id=user_id, scope=scope, pr=pr
            )
        )
    if user_vs_skills:
        tools.append(
            await get_user_skill_discovery_tool(
                session=session, user_id=user_id, scope="user_vs_skills", pr=pr
            )
        )
    log.debug("Agent tools collected, returning ...")
    return tools
