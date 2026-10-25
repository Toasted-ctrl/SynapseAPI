from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.logging import get_logger

log = get_logger()

# TODO: Write tests for both function.


def _build_qdrant_filter(filter: dict) -> Filter:
    conditions = [
        FieldCondition(
            key=f"metadata.{key}",
            match=MatchValue(
                value=str(value)
                if not isinstance(value, (str, int, bool, float))
                else value
            ),
        )
        for key, value in filter.items()
    ]
    return Filter(must=conditions)


def search_docs_similarity(
    vector_store: QdrantVectorStore, query: str, filter: dict
) -> list[tuple[Document, float]]:

    match vector_store:
        case QdrantVectorStore():
            result = vector_store.similarity_search_with_score(
                query=query, k=2, filter=_build_qdrant_filter(filter=filter)
            )
            log.debug(f"Found {len(result)} in Vector Store, returning ...")
            return result

        case _:
            log.error(
                f"Unsupported Vector Store detected: {type(vector_store).__name__} ..."
            )
            raise ValueError(
                f"Unsupported Vector Store type: {type(vector_store).__name__}"
            )
