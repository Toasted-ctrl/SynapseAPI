from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.logging import get_logger
from providers.dataclasses import LangChainCon
from security.encryption import decrypt

log = get_logger()


def _build_llm(
    langchain_con: LangChainCon,
    model: str,
    base_url: str,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    encrypted_api_key: str | None,
) -> BaseChatModel:
    """Construct the correct LangChain chat model for the given provider."""
    common_kwargs = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "top_p": top_p,
    }
    match langchain_con:
        case LangChainCon.ANTHROPIC:
            if encrypted_api_key is None:
                raise ValueError("encrypted_api_key is required for ChatAnthropic")
            log.debug("Constructing Chat Model: ChatAnthropic")
            return ChatAnthropic(
                api_key=decrypt(encrypted_api_key), top_k=top_k, **common_kwargs
            )
        case LangChainCon.OPENAI:
            if encrypted_api_key is None:
                raise ValueError("encrypted_api_key is required for ChatOpenAI")
            log.debug("Constructing Chat Model: ChatOpenAI")
            return ChatOpenAI(api_key=decrypt(encrypted_api_key), **common_kwargs)
        case LangChainCon.GOOGLE:
            if encrypted_api_key is None:
                raise ValueError(
                    "encrypted_api_key is required for ChatGoogleGenerativeAI"
                )
            log.debug("Constructing Chat Model: ChatGoogleGenerativeAI")
            return ChatGoogleGenerativeAI(
                api_key=decrypt(encrypted_api_key),
                model=model,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        case LangChainCon.OLLAMA:
            # Ollama is local — no API key needed
            log.debug("Constructing Chat Model: ChatOllama")
            return ChatOllama(top_k=top_k, **common_kwargs)
        case _:
            log.warning(f"Unsupported LangChain Chat Model: '{langchain_con}' ...")
            raise ValueError(f"Unsupported provider: {langchain_con}")
