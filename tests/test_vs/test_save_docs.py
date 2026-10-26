from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from vs.save_docs import (
    VectorStoreScope,
    _chunker,
    _normalize_texts,
    _prep_docs_personal_data,
    _sanitize_metadata,
    save_docs,
)


class TestChunker:
    """Tests for _chunker function."""

    def test_short_text_returns_single_chunk(self):
        """Test that short text (< 500 tokens) returns as single chunk."""
        short_text = "This is a short text."
        result = _chunker(short_text)
        assert result == [short_text]

    def test_long_text_returns_multiple_chunks(self):
        """Test that long text (> 500 tokens) returns multiple chunks."""
        long_text = "word " * 1000
        result = _chunker(long_text)
        assert len(result) > 1
        assert all(chunk != long_text for chunk in result)

    @patch("vs.save_docs.count_tokens")
    def test_token_count_border_case(self, mock_count_tokens):
        """Test behavior at exactly 500 tokens."""
        mock_count_tokens.return_value = 500
        text = "exactly 500 tokens"
        result = _chunker(text)
        assert result == [text]


class TestNormalizeTexts:
    """Tests for _normalize_texts function."""

    def test_normalizes_whitespace(self):
        """Test that extra whitespace is normalized."""
        texts = ["  hello   world  \n\n  test  "]
        result = _normalize_texts(texts)
        assert result == ["hello world\n\ntest"]

    def test_handles_multiple_texts(self):
        """Test normalization of multiple texts."""
        texts = ["  text  one  ", "  text  two  "]
        result = _normalize_texts(texts)
        assert result == ["text one", "text two"]

    def test_preserves_meaningful_whitespace(self):
        """Test that meaningful newlines are preserved."""
        texts = ["line one\n\nline two"]
        result = _normalize_texts(texts)
        assert result == ["line one\n\nline two"]


class TestSanitizeMetadata:
    """Tests for _sanitize_metadata function."""

    def test_converts_uuid_to_string(self):
        """Test that UUID values are converted to strings."""
        import uuid

        test_uuid = uuid.uuid4()
        metadatas = [{"id": test_uuid, "name": "test"}]
        result = _sanitize_metadata(metadatas)
        assert result[0]["id"] == str(test_uuid)
        assert result[0]["name"] == "test"

    def test_preserves_non_uuid_values(self):
        """Test that non-UUID values are preserved."""
        metadatas = [{"id": "string_id", "count": 42, "enabled": True}]
        result = _sanitize_metadata(metadatas)
        assert result == metadatas

    def test_handles_empty_metadata(self):
        """Test handling of empty metadata list."""
        result = _sanitize_metadata([])
        assert result == []


class TestPrepDocsPersonalData:
    """Tests for _prep_docs_personal_data function."""

    def test_creates_documents_with_metadata(self):
        """Test document creation with metadata."""
        texts = ["test text"]
        metadatas = [{"user-id": "user123", "document-name": "test_doc"}]

        result = _prep_docs_personal_data(texts, metadatas)

        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].page_content == "test text"
        assert "document-hash" in result[0].metadata
        assert "chunk-id" in result[0].metadata
        assert result[0].metadata["user-id"] == "user123"

    def test_creates_multiple_chunks_for_long_text(self):
        """Test that long text creates multiple document chunks."""
        long_text = "word " * 1000
        metadatas = [{"user-id": "user123", "document-name": "long_doc"}]

        result = _prep_docs_personal_data([long_text], metadatas)

        assert len(result) > 1
        # All chunks should have the same document-hash
        hashes = [doc.metadata["document-hash"] for doc in result]
        assert len(set(hashes)) == 1  # All chunks have same hash

    def test_handles_multiple_texts(self):
        """Test handling of multiple text/metadata pairs."""
        texts = ["text one", "text two"]
        metadatas = [{"user-id": "user1"}, {"user-id": "user2"}]

        result = _prep_docs_personal_data(texts, metadatas)

        assert len(result) == 2
        assert result[0].metadata["user-id"] == "user1"
        assert result[1].metadata["user-id"] == "user2"


class TestVectorStoreScope:
    """Tests for VectorStoreScope enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert VectorStoreScope.USER_FILES == "user_vs_files"
        assert VectorStoreScope.USER_MEMORIES == "user_vs_memories"
        assert VectorStoreScope.USER_SKILLS == "user_vs_skills"
        assert VectorStoreScope.AGENT == "agent"


class TestSaveDocs:
    """Tests for save_docs function."""

    @patch("vs.save_docs._normalize_texts")
    @patch("vs.save_docs._sanitize_metadata")
    @patch("vs.save_docs._prep_docs_personal_data")
    def test_successful_save_user_files(
        self, mock_prep_docs, mock_sanitize, mock_normalize
    ):
        """Test successful document saving for USER_FILES scope."""
        mock_vs = MagicMock(spec=QdrantVectorStore)
        mock_vs.add_documents.return_value = ["doc_id_1", "doc_id_2"]

        mock_prep_docs.return_value = [
            Document(id="doc1", page_content="content1", metadata={"user-id": "user1"})
        ]
        mock_normalize.return_value = ["normalized text"]
        mock_sanitize.return_value = [{"user-id": "user1"}]

        result = save_docs(
            vector_store=mock_vs,
            scope=VectorStoreScope.USER_FILES,
            texts=["test text"],
            metadatas=[{"user-id": "user1"}],
            required_metadata=["user-id"],
        )

        assert result == ["doc_id_1", "doc_id_2"]
        mock_vs.add_documents.assert_called_once()

    @patch("vs.save_docs._normalize_texts")
    @patch("vs.save_docs._sanitize_metadata")
    @patch("vs.save_docs._prep_docs_personal_data")
    def test_successful_save_user_skills(
        self, mock_prep_docs, mock_sanitize, mock_normalize
    ):
        """Test successful document saving for USER_SKILLS scope."""
        mock_vs = MagicMock(spec=QdrantVectorStore)
        mock_vs.add_documents.return_value = ["skill_doc_id"]

        mock_prep_docs.return_value = [
            Document(
                id="skill_doc",
                page_content="skill content",
                metadata={"user-id": "user1", "skill-id": "skill1"},
            )
        ]
        mock_normalize.return_value = ["normalized skill content"]
        mock_sanitize.return_value = [{"user-id": "user1", "skill-id": "skill1"}]

        result = save_docs(
            vector_store=mock_vs,
            scope=VectorStoreScope.USER_SKILLS,
            texts=["skill description"],
            metadatas=[{"user-id": "user1", "skill-id": "skill1"}],
            required_metadata=["user-id"],
        )

        assert result == ["skill_doc_id"]
        mock_vs.add_documents.assert_called_once()

    @patch("vs.save_docs._normalize_texts")
    @patch("vs.save_docs._sanitize_metadata")
    def test_missing_required_metadata_raises_error(
        self, mock_sanitize, mock_normalize
    ):
        """Test that missing required metadata raises ValueError."""
        mock_normalize.return_value = ["normalized text"]
        mock_sanitize.return_value = [{"wrong-key": "value"}]

        with pytest.raises(
            ValueError, match="Metadata at index 0 missing required keys"
        ):
            save_docs(
                vector_store=MagicMock(spec=QdrantVectorStore),
                scope=VectorStoreScope.USER_FILES,
                texts=["test text"],
                metadatas=[{"wrong-key": "value"}],
                required_metadata=["user-id"],
            )

    def test_mismatched_texts_and_metadatas_raises_error(self):
        """Test that mismatched texts and metadatas raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch: 2 texts vs 1 metadatas"):
            save_docs(
                vector_store=MagicMock(spec=QdrantVectorStore),
                scope=VectorStoreScope.USER_FILES,
                texts=["text1", "text2"],
                metadatas=[{"user-id": "user1"}],
                required_metadata=[],
            )

    def test_unsupported_scope_raises_error(self):
        """Test that unsupported scope raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported Vector Store scope"):
            save_docs(
                vector_store=MagicMock(spec=QdrantVectorStore),
                scope="unsupported_scope",
                texts=["test text"],
                metadatas=[{"user-id": "user1"}],
                required_metadata=[],
            )

    def test_required_metadata_without_metadatas_raises_error(self):
        """Test that required_metadata without metadatas raises ValueError."""
        with pytest.raises(
            ValueError, match="required_metadata specified but no metadatas provided"
        ):
            save_docs(
                vector_store=MagicMock(spec=QdrantVectorStore),
                scope=VectorStoreScope.USER_FILES,
                texts=["test text"],
                required_metadata=["user-id"],
            )

    def test_unsupported_vector_store_raises_error(self):
        """Test that unsupported vector store type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported Vector Store type"):
            save_docs(
                vector_store=MagicMock(),  # Not a QdrantVectorStore
                scope=VectorStoreScope.USER_FILES,
                texts=["test text"],
                metadatas=[{"user-id": "user1"}],
                required_metadata=[],
            )

    @pytest.mark.parametrize("metadatas", [None, []])
    @patch("vs.save_docs._sanitize_metadata")
    @patch("vs.save_docs._prep_docs_personal_data")
    def test_without_metadatas_uses_empty_metadata_per_text(
        self, mock_prep_docs, mock_sanitize, metadatas
    ):
        """Test that omitted or empty metadatas give each text its own empty metadata dict."""
        mock_vs = MagicMock(spec=QdrantVectorStore)
        mock_vs.add_documents.return_value = ["doc_id_1", "doc_id_2"]
        mock_prep_docs.return_value = []

        result = save_docs(
            vector_store=mock_vs,
            scope=VectorStoreScope.USER_FILES,
            texts=["text one", "text two"],
            metadatas=metadatas,
        )

        assert result == ["doc_id_1", "doc_id_2"]
        mock_sanitize.assert_not_called()
        mock_prep_docs.assert_called_once_with(
            texts=["text one", "text two"], metadatas=[{}, {}]
        )
        passed = mock_prep_docs.call_args.kwargs["metadatas"]
        assert passed[0] is not passed[1]
