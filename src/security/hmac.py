import hashlib
import hmac


def hash_hmac(content: str, key: bytes) -> str:
    """Creates and returns an hmac encoded string."""
    canonical_string = content.encode("utf-8")
    return hmac.new(key=key, msg=canonical_string, digestmod=hashlib.sha256).hexdigest()


def is_valid_hmac(provided_hmac: str, expected_hmac: str) -> bool:
    """Compares two hmac signatures.
    Returns False if the hmac digests are not the same."""
    return hmac.compare_digest(provided_hmac, expected_hmac)
