import secrets


def create_secret() -> str:
    """Creates a new random API key"""

    return secrets.token_urlsafe(nbytes=64)
