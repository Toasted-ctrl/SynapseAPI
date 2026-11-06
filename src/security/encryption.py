from typing import Any, Tuple

from cryptography.fernet import Fernet
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from core.config import config


def _validate_content(content: str) -> None:
    if not isinstance(content, str):
        raise TypeError("Content must be a string")
    if content == "":
        raise ValueError("Empty string")


def encrypt(content: str) -> str:
    """Returns an encrypted string based on the configured encryption key."""
    _validate_content(content)
    key = config.ENCRYPTION_KEY
    f = Fernet(key)
    return f.encrypt(content.encode(encoding="utf-8")).decode(encoding="utf-8")


def decrypt(content: str) -> str:
    """Returns a decrypted string based on the configured encryption key."""
    _validate_content(content)
    key = config.ENCRYPTION_KEY
    f = Fernet(key)
    return f.decrypt(content.encode(encoding="utf-8")).decode(encoding="utf-8")


class EncryptedSerializer(SerializerProtocol):
    def __init__(self, key: bytes | str):
        # Fernet requires a URL-safe base64-encoded 32-byte key
        self.fernet = Fernet(key)
        self.serde = JsonPlusSerializer()

    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        # 1. Serialize the object into bytes using LangGraph's default serializer
        type_str, data_bytes = self.serde.dumps_typed(obj)
        # 2. Encrypt the bytes
        encrypted_bytes = self.fernet.encrypt(data_bytes)
        return type_str, encrypted_bytes

    def loads_typed(self, data: Tuple[str, bytes]) -> Any:
        type_str, encrypted_bytes = data
        # 1. Decrypt the bytes
        decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
        # 2. Deserialize back into the original Python/LangChain objects
        return self.serde.loads_typed((type_str, decrypted_bytes))
