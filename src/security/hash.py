import hashlib


def get_hash_sha256(content: str) -> str:
    """Hashes provided string."""
    if not isinstance(content, str):
        raise TypeError("Content must be a string")
    if content == "":
        raise ValueError("Content must not be empty")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
