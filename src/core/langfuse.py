from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from core.config import config

langfuse = Langfuse(
    public_key=config.LANGFUSE_PUBLIC_KEY,
    secret_key=config.LANGFUSE_SECRET_KEY,
    base_url=config.LANGFUSE_BASE_URL,
)

langfuse_handler = CallbackHandler()
