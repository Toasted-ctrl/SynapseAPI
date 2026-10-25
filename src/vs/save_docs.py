import hashlib
import re
import uuid
from enum import Enum

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logging import get_logger
from vs.count import count_tokens

log = get_logger()


def _chunker(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    """Divides a provided body of text into chunks. Will return the full body as
    single chunk if the chunk is smaller than 500 tokens."""

    if count_tokens(text=text) <= 500:
        log.debug("Token count is less than or equal to 500, returning whole body ...")
        return [text]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=max_tokens, chunk_overlap=overlap
    )

    log.debug("Initialized chunker.")
    split = splitter.split_text(text=text)
    log.debug(f"Created {len(split)} from body of text, returning ...")
    return split


def _normalize_texts(texts: list[str]) -> list[str]:
    """Removes unnecessary leading/trailing and internal whitespace from text."""
    normalized = []
    for text in texts:
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        normalized.append(text)
    log.debug("Normalized texts, returning ...")
    return normalized


def _prep_docs_personal_data(texts: list[str], metadatas: list[dict]) -> list[Document]:
    """Prepares the data for ingestion. Will create 'Documents' for each piece of text to be ingested.
    Document preparation only intended for preparing PERSONAL data."""

    documents = []
    for text, metadata in zip(texts, metadatas):
        document_hash = hashlib.sha256(text.encode()).hexdigest()
        user_id = metadata.get("user-id")
        chunks = _chunker(text=text)
        for i, chunk in enumerate(chunks):
            # Qdrant only supports integers or UUIDs as point ID.
            # Use uuid.uuid(5) so we can generate a reproducable UUID, so
            # duplicate entries can be accounted for.

            document_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{user_id}:{document_hash}:{i}:{metadata.get('document-name')}",
                )
            )
            metadata["document-hash"] = str(document_hash)
            metadata["chunk-id"] = i
            documents.append(
                Document(id=document_id, page_content=chunk, metadata=metadata)
            )
    return documents


def _sanitize_metadata(metadatas: list[dict]) -> list[dict]:
    """Converts any UUID values in metadata dicts to string."""
    sanitized = []
    for metadata in metadatas:
        sanitized.append(
            {k: str(v) if isinstance(v, uuid.UUID) else v for k, v in metadata.items()}
        )
    return sanitized


class VectorStoreScope(str, Enum):
    USER_FILES = "user_vs_files"
    USER_MEMORIES = "user_vs_memories"
    USER_SKILLS = "user_vs_skills"
    AGENT = "agent"


def save_docs(
    vector_store: QdrantVectorStore,
    scope: VectorStoreScope,
    texts: list[str],
    metadatas: list[dict] | None = None,
    required_metadata: list[str] | None = None,
) -> list[str]:
    """Adds documents to indicated Vector Store.
    Currently only Qdrant supported."""

    if required_metadata and not metadatas:
        raise ValueError("required_metadata specified but no metadatas provided ...")

    if metadatas:
        if len(metadatas) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts vs {len(metadatas)} metadatas ..."
            )
        if required_metadata:
            for i, metadata in enumerate(metadatas):
                missing = [r for r in required_metadata if r not in metadata]
                if missing:
                    raise ValueError(
                        f"Metadata at index {i} missing required keys: {missing} ..."
                    )

    _texts = _normalize_texts(texts=texts)
    _metadatas = _sanitize_metadata(metadatas=metadatas)

    match scope:
        case (
            VectorStoreScope.USER_FILES
            | VectorStoreScope.USER_MEMORIES
            | VectorStoreScope.USER_SKILLS
        ):
            log.debug(f"Saving documents with scope '{scope}' ...")
            docs = _prep_docs_personal_data(texts=_texts, metadatas=_metadatas)

        # TODO: Implement case for when DocType is AGENT.

        case _:
            log.error(f"Unsupported Vector Store scope detected: {scope} ...")
            raise ValueError(f"Unsupported Vector Store scope: {scope}")

    match vector_store:
        case QdrantVectorStore():
            log.debug("Saving documents to Qdrant vector store ...")
            ids = vector_store.add_documents(documents=docs)
            return ids

        # TODO: Add support for other Vector Databases

        case _:
            log.error(
                f"Unsupported Vector Store detected: {type(vector_store).__name__} ..."
            )
            raise ValueError(
                f"Unsupported Vector Store type: {type(vector_store).__name__}"
            )
