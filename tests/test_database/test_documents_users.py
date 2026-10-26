import uuid
from unittest.mock import AsyncMock

import pytest
from db_helpers import make_session, params, row

from database.documents_users import (
    delete_user_document_by_document_id,
    get_user_documents_by_scope,
    store_user_document,
)
from database.schemas.documents_user import DocumentsUsersT
from database.vector_store import VectorStoreConfig

pytestmark = pytest.mark.asyncio


class TestStoreUserDocument:
    async def test_stores_and_returns_document_id(self):
        user_id = uuid.uuid4()
        session = make_session(scalars=[[]])

        result = await store_user_document(
            session=session, user_id=user_id, name="cv.pdf", scope="user_vs_files"
        )

        [added] = session.added
        assert isinstance(added, DocumentsUsersT)
        assert (added.user_id, added.name, added.scope) == (
            user_id,
            "cv.pdf",
            "user_vs_files",
        )
        session.flush.assert_awaited_once()
        assert result == added.id

    async def test_duplicate_check_filters_on_name_scope_and_user(self):
        user_id = uuid.uuid4()
        session = make_session(scalars=[[]])

        await store_user_document(
            session=session, user_id=user_id, name="cv.pdf", scope="user_vs_files"
        )

        assert params(session.scalars.await_args) == [
            "cv.pdf",
            "user_vs_files",
            user_id,
        ]

    async def test_raises_when_document_exists(self):
        session = make_session(scalars=[[uuid.uuid4()]])

        with pytest.raises(ValueError, match="already exists"):
            await store_user_document(
                session=session,
                user_id=uuid.uuid4(),
                name="cv.pdf",
                scope="user_vs_files",
            )

        session.add.assert_not_called()


class TestGetUserDocumentsByScope:
    async def test_returns_name_and_id(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        user_id = uuid.uuid4()
        session = make_session(
            scalars=[[row(id=a, name="one.pdf"), row(id=b, name="two.pdf")]]
        )

        result = await get_user_documents_by_scope(
            session=session, user_id=user_id, scope="user_vs_files"
        )

        assert result == [{"name": "one.pdf", "id": a}, {"name": "two.pdf", "id": b}]
        assert params(session.scalars.await_args) == [user_id, "user_vs_files"]

    async def test_returns_empty_list(self):
        session = make_session(scalars=[[]])

        assert (
            await get_user_documents_by_scope(
                session=session, user_id=uuid.uuid4(), scope="x"
            )
            == []
        )


class TestDeleteUserDocumentByDocumentId:
    @pytest.fixture
    def qdrant(self, monkeypatch):
        vscf = VectorStoreConfig(
            vs_collection_name="files",
            vs_encrypted_api_key="enc:qdrant",
            vs_vendor="qdrant",
            vs_base_url="http://qdrant",
            vs_port=6333,
            e_dimensions=768,
            e_provider="Ollama",
            e_model="nomic-embed-text",
            e_api_key="enc:e",
            scope="user_vs_files",
            required_filters=[],
        )
        settings = AsyncMock(return_value=vscf)
        delete = AsyncMock()
        monkeypatch.setattr(
            "database.documents_users.get_vector_store_settings", settings
        )
        monkeypatch.setattr(
            "database.documents_users.delete_document_from_qdrant_by_user_document_id",
            delete,
        )
        return settings, delete

    @pytest.mark.parametrize("scope", ["user_vs_files", "user_vs_memories"])
    async def test_deletes_from_db_and_qdrant(self, qdrant, scope):
        settings, delete = qdrant
        user_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = row(id=doc_id, scope=scope)
        session = make_session(scalar=[doc])

        result = await delete_user_document_by_document_id(
            session=session, user_id=user_id, document_id=doc_id
        )

        assert result == doc_id
        assert session.delete.await_count == 1
        assert session.flush.await_count == 1
        settings.assert_awaited_once_with(scope=scope, session=session)
        delete.assert_awaited_once_with(
            user_id=user_id,
            user_document_id=doc_id,
            collection_name="files",
            url="http://qdrant",
            port=6333,
            encrypted_api_key="enc:qdrant",
        )

    async def test_only_deletes_documents_owned_by_user(self, qdrant):
        user_id, doc_id = uuid.uuid4(), uuid.uuid4()
        session = make_session(scalar=[None])

        with pytest.raises(ValueError, match="does not exist"):
            await delete_user_document_by_document_id(
                session=session, user_id=user_id, document_id=doc_id
            )

        assert params(session.scalar.await_args) == [user_id, doc_id]

    async def test_raises_value_error_when_not_found(self, qdrant):
        _, delete = qdrant
        session = make_session(scalar=[None])

        with pytest.raises(ValueError, match="does not exist"):
            await delete_user_document_by_document_id(
                session=session, user_id=uuid.uuid4(), document_id=uuid.uuid4()
            )

        session.delete.assert_not_awaited()
        delete.assert_not_awaited()

    async def test_skips_qdrant_for_non_vector_scopes(self, qdrant):
        settings, delete = qdrant
        doc_id = uuid.uuid4()
        session = make_session(scalar=[row(id=doc_id, scope="other")])

        result = await delete_user_document_by_document_id(
            session=session, user_id=uuid.uuid4(), document_id=doc_id
        )

        assert result == doc_id
        assert session.delete.await_count == 1
        assert session.flush.await_count == 1
        settings.assert_not_awaited()
        delete.assert_not_awaited()

    async def test_deletes_skill_when_scope_is_user_vs_skills(self, qdrant):
        """Test that when deleting a user_vs_skills document, the associated UserSkillsT record is also deleted."""
        settings, delete = qdrant
        user_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = row(id=doc_id, scope="user_vs_skills")
        skill = row(id=doc_id, user_id=user_id)
        session = make_session(scalar=[doc, skill])

        result = await delete_user_document_by_document_id(
            session=session, user_id=user_id, document_id=doc_id
        )

        assert result == doc_id
        # Both document and skill should be deleted
        assert session.delete.await_count == 2
        # Flush is called twice: once after document delete, once after skill delete
        assert session.flush.await_count == 2
        settings.assert_awaited_once_with(scope="user_vs_skills", session=session)
        delete.assert_awaited_once()

    async def test_does_not_delete_skill_for_non_skill_scopes(self, qdrant):
        """Test that skill deletion is NOT attempted for user_vs_files and user_vs_memories scopes."""
        settings, delete = qdrant
        user_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = row(id=doc_id, scope="user_vs_files")
        session = make_session(scalar=[doc])

        result = await delete_user_document_by_document_id(
            session=session, user_id=user_id, document_id=doc_id
        )

        assert result == doc_id
        # Only the document should be deleted, not a skill
        assert session.delete.await_count == 1
        # Only one flush since no skill deletion
        assert session.flush.await_count == 1
        settings.assert_awaited_once_with(scope="user_vs_files", session=session)
        delete.assert_awaited_once()

    async def test_deletes_skill_only_when_skill_exists(self, qdrant):
        """Test that deletion works correctly when no skill exists for the document."""
        settings, delete = qdrant
        user_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = row(id=doc_id, scope="user_vs_skills")
        session = make_session(
            scalar=[doc, None]
        )  # Second scalar returns None (no skill found)

        result = await delete_user_document_by_document_id(
            session=session, user_id=user_id, document_id=doc_id
        )

        assert result == doc_id
        # Only the document should be deleted since no skill exists
        assert session.delete.await_count == 1
        # Only one flush since no skill was deleted
        assert session.flush.await_count == 1
