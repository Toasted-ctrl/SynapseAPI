from urllib.parse import urlencode

from fastapi import Request, Response
from fastapi_cache import FastAPICache

from core.config import config
from core.logging import get_logger
from security.hmac import hash_hmac

log = get_logger()


def cache_key_builder(
    func,
    namespace: str = "",
    request: Request = None,
    response: Response = None,
    *args,
    **kwargs,
):

    key = config.BLIND_INDEX_HMAC_KEY

    """Build cache key to store results including hashed api_key, jwt and path.
    Requires request: Request to be included in the main function call.
    Should ONLY be used on non-sensitive GET requests."""

    user = kwargs.get("kwargs", {}).get("user")
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
