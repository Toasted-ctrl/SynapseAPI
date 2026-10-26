import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.logging import get_logger
from security.encryption import decrypt

log = get_logger()


async def delete_document_from_qdrant_by_user_document_id(
    user_id: uuid.UUID,
    user_document_id: uuid.UUID,
    collection_name: str,
    url: str,
    port: int,
    encrypted_api_key: str,
) -> None:

    qdrant = AsyncQdrantClient(url=url, api_key=decrypt(encrypted_api_key), port=port)

    try:
        result = await qdrant.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.user-document-id",
                        match=MatchValue(value=str(user_document_id)),
                    ),
                    FieldCondition(
                        key="metadata.user-id", match=MatchValue(value=str(user_id))
                    ),
                ],
            ),
            wait=True,
        )
        log.debug(result)
    finally:
        await qdrant.close()
