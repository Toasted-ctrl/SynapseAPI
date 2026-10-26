import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from tools.skill_discovery import VSSkillSearchInput, get_user_skill_discovery_tool

pytestmark = pytest.mark.asyncio


def _mock_vscf() -> MagicMock:
    vscf = MagicMock()
    vscf.e_provider = "test_provider"
    vscf.vs_collection_name = "test_collection"
    vscf.vs_vendor = "test_vendor"
    vscf.vs_port = 8080
    vscf.vs_base_url = "http://test"
    vscf.vs_encrypted_api_key = "test_key"
    vscf.e_model = "test_model"
    vscf.e_dimensions = 1536
    vscf.e_api_key = "collection_enc_key"
    return vscf


def _mock_pr() -> MagicMock:
    em = MagicMock()
    em.base_url = "http://embeddings"
    em.langchain_con = MagicMock()
    em.encrypted_api_key = None  # User has no personal key for the embedding provider.
    pr = MagicMock()
    pr.test_provider = em
    return pr


def _mock_doc(skill_id: uuid.UUID) -> MagicMock:
    doc = MagicMock()
    doc.page_content = "Skill description"
    doc.metadata = {"user-document-id": str(skill_id)}
    return doc


@contextmanager
def _patched(search_results: list | None = None, skills: list | None = None):
    """Patches all external dependencies of the skill discovery tool."""
    with (
        patch(
            "tools.skill_discovery.get_vector_store_settings", new_callable=AsyncMock
        ) as mock_get_vscf,
        patch("tools.skill_discovery.get_vector_store") as mock_get_vs,
        patch("tools.skill_discovery.search_docs_similarity") as mock_search,
        patch(
            "tools.skill_discovery.get_user_skill_by_skill_id", new_callable=AsyncMock
        ) as mock_get_skill,
    ):
        mock_get_vscf.return_value = _mock_vscf()
        mock_get_vs.return_value = MagicMock()
        mock_search.return_value = search_results or []
        mock_get_skill.side_effect = skills or []
        yield MagicMock(
            get_vscf=mock_get_vscf,
            get_vs=mock_get_vs,
            search=mock_search,
            get_skill=mock_get_skill,
        )


async def _build_tool(user_id: uuid.UUID, session=None) -> StructuredTool:
    return await get_user_skill_discovery_tool(
        session=session or AsyncMock(),
        user_id=user_id,
        scope="user_vs_skills",
        pr=_mock_pr(),
    )


class TestGetUserSkillDiscoveryTool:
    async def test_returns_structured_tool(self):
        """Test that function returns a StructuredTool."""
        with _patched():
            result = await _build_tool(uuid.uuid4())

        assert isinstance(result, StructuredTool)
        assert result.name == "discover_skills"

    async def test_tool_is_registered_as_coroutine(self):
        """Test that the async function is registered as coroutine, not as sync func."""
        with _patched():
            result = await _build_tool(uuid.uuid4())

        assert result.coroutine is not None
        assert result.func is None

    async def test_tool_has_correct_description(self):
        """Test that the tool has the expected description."""
        with _patched():
            result = await _build_tool(uuid.uuid4())

        assert "skill database" in result.description.lower()

    async def test_tool_has_correct_args_schema(self):
        """Test that the tool has the expected args schema."""
        with _patched():
            result = await _build_tool(uuid.uuid4())

        assert result.args_schema == VSSkillSearchInput

    async def test_loads_vector_store_settings_for_scope(self):
        """Test that the vector store settings are loaded for the given scope."""
        mock_session = AsyncMock()
        with _patched() as mocks:
            await _build_tool(uuid.uuid4(), session=mock_session)

        mocks.get_vscf.assert_awaited_once_with(
            scope="user_vs_skills", session=mock_session
        )

    async def test_returns_no_results_message_when_empty(self):
        """Test that the tool returns 'No relevant results found' when search returns empty."""
        with _patched(search_results=[]) as mocks:
            tool = await _build_tool(uuid.uuid4())
            output = await tool.ainvoke({"query": "test"})

        assert output == "No relevant results found"
        mocks.get_skill.assert_not_awaited()

    async def test_returns_skill_instructions(self):
        """Test that the tool returns the instructions of each found skill."""
        user_id = uuid.uuid4()
        skill_id_1, skill_id_2 = uuid.uuid4(), uuid.uuid4()
        skills = [
            MagicMock(instructions="Instructions one"),
            MagicMock(instructions="Instructions two"),
        ]
        search_results = [(_mock_doc(skill_id_1), 0.95), (_mock_doc(skill_id_2), 0.80)]

        mock_session = AsyncMock()
        with _patched(search_results=search_results, skills=skills) as mocks:
            tool = await _build_tool(user_id, session=mock_session)
            output = await tool.ainvoke({"query": "test"})

        assert output == "Instructions one\n\nInstructions two"
        assert mocks.get_skill.await_count == 2
        first_call = mocks.get_skill.await_args_list[0].kwargs
        assert first_call == {
            "session": mock_session,
            "user_id": user_id,
            "skill_id": skill_id_1,
        }
        assert mocks.get_skill.await_args_list[1].kwargs["skill_id"] == skill_id_2

    async def test_filters_by_user_id(self):
        """Test that the search filter includes the user_id."""
        user_id = uuid.uuid4()
        with _patched() as mocks:
            tool = await _build_tool(user_id)
            await tool.ainvoke({"query": "test"})

        assert mocks.search.call_args.kwargs["filter"] == {"user-id": str(user_id)}

    async def test_passes_query_to_search(self):
        """Test that the query is passed correctly to search."""
        with _patched() as mocks:
            tool = await _build_tool(uuid.uuid4())
            await tool.ainvoke({"query": "test query", "limit": 10})

        assert mocks.search.call_args.kwargs["query"] == "test query"

    async def test_uses_correct_vector_store_config(self):
        """Test that the vector store is built from the collection and provider settings."""
        with _patched() as mocks:
            tool = await _build_tool(uuid.uuid4())
            await tool.ainvoke({"query": "test"})

        mocks.get_vs.assert_called_once()
        call_kwargs = mocks.get_vs.call_args.kwargs
        assert call_kwargs["vs_collection_name"] == "test_collection"
        assert call_kwargs["vs_vendor"] == "test_vendor"
        assert call_kwargs["vs_port"] == 8080
        assert call_kwargs["vs_base_url"] == "http://test"
        assert call_kwargs["vs_encrypted_api_key"] == "test_key"
        assert call_kwargs["e_model"] == "test_model"
        assert call_kwargs["e_dimensions"] == 1536
        assert call_kwargs["e_base_url"] == "http://embeddings"

    async def test_uses_collection_embedding_key(self):
        """Test that the collection's embedding key is used, not the User's provider key."""
        with _patched() as mocks:
            tool = await _build_tool(uuid.uuid4())
            await tool.ainvoke({"query": "test"})

        assert (
            mocks.get_vs.call_args.kwargs["e_encrypted_api_key"] == "collection_enc_key"
        )
