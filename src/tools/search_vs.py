import uuid

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.providers import UserProviderRegistry
from database.vector_store import VectorStoreConfig, get_vector_store_settings
from vs.get_vs import get_vector_store
from vs.search import search_docs_similarity

log = get_logger()


class VSSearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up in the documents.")
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of relevant documents to return.",
    )


async def get_vs_search_tool(
    session: AsyncSession, user_id: uuid.UUID, scope: str, pr: UserProviderRegistry
) -> StructuredTool:
    """Vector Store search tool factory.
    The user_id and scope are set by the API, not the LLM."""

    log.debug(
        f"Creating vector store search tool for User '{user_id}' with scope '{scope}'."
    )

    # Loaded up front so the tool never touches the request's AsyncSession, which can't be shared across concurrent tool calls.
    vscf: VectorStoreConfig = await get_vector_store_settings(
        scope=scope, session=session
    )

    def search_vector_store(query: str, limit: int = 5) -> str:
        em = pr.__getattr__(vscf.e_provider)

        vs = get_vector_store(
            vs_collection_name=vscf.vs_collection_name,
            vs_vendor=vscf.vs_vendor,
            vs_port=vscf.vs_port,
            vs_base_url=vscf.vs_base_url,
            vs_encrypted_api_key=vscf.vs_encrypted_api_key,
            e_model=vscf.e_model,
            e_base_url=em.base_url,
            e_langchain_con=em.langchain_con,
            e_encrypted_api_key=vscf.e_api_key,
            e_dimensions=vscf.e_dimensions,
        )

        filter = {"user-id": str(user_id)}

        results = search_docs_similarity(vector_store=vs, query=query, filter=filter)

        log.debug(f"Found {len(results)} in vector store.")

        if not results:
            return "No relevant documents were found."

        return "\n\n".join(
            f"[{i}] Score: {score}\n"
            f"Content:\n{doc.page_content}\n"
            f"Metadata: {doc.metadata}"
            for i, (doc, score) in enumerate(results, start=1)
        )

    return StructuredTool.from_function(
        func=search_vector_store,
        name=f"search_{scope}_documents",
        description=(
            f"Search the user's {scope} documents for relevant information. "
            "Use this when answering questions that may require information "
            "from the user's stored documents."
        ),
        args_schema=VSSearchToolInput,
    )
