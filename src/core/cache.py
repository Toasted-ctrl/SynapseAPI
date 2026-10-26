from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

from fastapi import Request, Response
from fastapi_cache import FastAPICache

from core.config import config
from core.logging import get_logger
from security.hmac import hash_hmac

log = get_logger()


def cache_key_builder(
    func: Callable[..., Any],
    namespace: str = "",
    *,
    request: Request | None = None,
    response: Response | None = None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Build cache key to store results including hashed api_key, jwt and path.
    Requires request: Request to be included in the main function call.
    Should ONLY be used on non-sensitive GET requests."""

    if request is None:
        raise ValueError(
            f"cache_key_builder requires 'request: Request' on '{func.__name__}'"
        )

    key = config.BLIND_INDEX_HMAC_KEY

    user = kwargs.get("user")
    user_id = hash_hmac(content=str(user.id) if user else "Anonymous", key=key)

    api_key = hash_hmac(content=request.headers.get("X-API-Key", "Not Set"), key=key)
    path = request.url.path
    # Sort so that ?a=1&b=2 and ?b=2&a=1 share a cache entry.
    query = urlencode(sorted(request.query_params.multi_items()))
    prefix = FastAPICache.get_prefix()

    cache_key = f"{prefix}:{func.__name__}:{api_key}:{user_id}:{path}?{query}"

    log.debug(f"Full cache key: '{cache_key}'")

    # TODO: Now the entries are hashed. Do we also need to hash encrypt the output?

    return cache_key
