from enum import Enum

from langchain_qdrant import QdrantVectorStore

from core.logging import get_logger
from providers.dataclasses import LangChainCon
from providers.vector_embedding import _build_embedding_model
from security.encryption import decrypt

log = get_logger()


class VectorStore(str, Enum):
    QDRANT = "qdrant"


def get_vector_store(
    vs_collection_name: str,
    vs_vendor: VectorStore,
    vs_port: int,
    vs_base_url: str,
    vs_encrypted_api_key: str | None,
    e_model: str,
    e_base_url: str,
    e_langchain_con: LangChainCon,
    e_encrypted_api_key: str,
    e_dimensions: int,
) -> QdrantVectorStore:
    """Creates and returns a Qdrant Vector Store instance."""

    emb = _build_embedding_model(
        langchain_con=e_langchain_con,
        model=e_model,
        base_url=e_base_url,
        dimensions=e_dimensions,
        encrypted_api_key=e_encrypted_api_key,
    )

    match vs_vendor:
        case VectorStore.QDRANT:
            log.debug(
                f"Building Qdrant Vector Store client for collection name '{vs_collection_name}' ... "
                f"Connecting to Qdrant at URL '{vs_base_url}' ..."
            )

            return QdrantVectorStore.from_existing_collection(
                collection_name=vs_collection_name,
                url=vs_base_url,
                embedding=emb,
                port=vs_port,
                api_key=decrypt(vs_encrypted_api_key) if vs_encrypted_api_key else None,
            )

        # TODO: Add support for additional Vector Store vendors.

        case _:
            raise ValueError(f"Vector Store {vs_vendor} currently not supported.")
