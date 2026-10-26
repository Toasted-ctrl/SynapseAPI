from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from core.logging import get_logger
from core.model_types import model_config
from providers.dataclasses import LangChainCon
from security.encryption import decrypt

log = get_logger()


def _build_embedding_model(
    langchain_con: LangChainCon,
    model: str,
    base_url: str,
    dimensions: int | None = None,
    encrypted_api_key: str | None = None,
) -> OllamaEmbeddings | OpenAIEmbeddings:
    """Embedding model factory."""

    common_kwargs = {"model": model, "base_url": base_url, "dimensions": dimensions}

    match langchain_con:
        case LangChainCon.OLLAMA:
            log.debug(f"Requesting Ollama embedding model '{model}' ...")
            return OllamaEmbeddings(**common_kwargs)

        case LangChainCon.OPENAI:
            if encrypted_api_key is None:
                raise ValueError("encrypted_api_key is required for OpenAIEmbeddings")
            log.debug(f"Requesting OpenAI embedding model '{model}' ...")

            # Mistral's embeddings endpoint rejects the OpenAI 'dimensions' param.
            if "mistral" in base_url.lower():
                common_kwargs.pop("dimensions", None)

            return OpenAIEmbeddings(
                **common_kwargs,
                openai_api_key=decrypt(encrypted_api_key),
                check_embedding_ctx_length=not any(
                    x in base_url.lower() for x in ("melious", "mistral")
                ),
            )

        case _:
            raise NotImplementedError(
                f"Embedding not supported for '{langchain_con}' ..."
            )


async def get_embedding(
    langchain_con: LangChainCon,
    model: str,
    prompt: str,
    base_url: str,
    dimensions: int | None = None,
    encrypted_api_key: str | None = None,
) -> list[float]:
    """Retrieve embedding from indicated Provider with specified Embedding model."""

    # TODO: Some Providers will probably not have any embedding models.
    # Figure out some way to block if an embedding model is requested from a Provider
    # that does not support it.

    if model not in await model_config.vector_embedding_models():
        raise ValueError(f"'{model}' is not recognized as embedding model ...")

    embedding = _build_embedding_model(
        langchain_con=langchain_con,
        model=model,
        base_url=base_url,
        dimensions=dimensions,
        encrypted_api_key=encrypted_api_key,
    )

    log.debug("Embedding Model constructed, generating embedding ...")

    result = await embedding.aembed_query(prompt)

    log.debug(f"Embedding complete, returning {result[:5]}...")

    return result
