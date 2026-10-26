from types import SimpleNamespace

import pytest
from starlette.requests import Request

from core.cache import cache_key_builder
from security.hmac import hash_hmac

HMAC_KEY = b"test-hmac-key"
PREFIX = "test-prefix"


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    """Patches the HMAC key and the FastAPICache prefix."""
    monkeypatch.setattr(
        "core.cache.config", SimpleNamespace(BLIND_INDEX_HMAC_KEY=HMAC_KEY)
    )
    monkeypatch.setattr("core.cache.FastAPICache.get_prefix", lambda: PREFIX)


def make_request(path="/v1/providers/models/sampling", query=b"", api_key=None):
    headers = [(b"x-api-key", api_key.encode())] if api_key else []
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": query,
            "headers": headers,
        }
    )


def endpoint():
    pass


def build(request, user=None, func=endpoint):
    """Calls the builder the same way fastapi-cache does."""
    return cache_key_builder(
        func,
        "",
        request=request,
        response=None,
        args=(),
        kwargs={"user": user} if user else {},
    )


class TestCacheKeyBuilder:
    def test_key_format(self):
        """Test that the key contains prefix, function, hashed api key, hashed user, path and query."""
        user = SimpleNamespace(id=42)
        key = build(
            make_request(query=b"provider_name=ollama", api_key="secret"), user=user
        )

        expected = (
            f"{PREFIX}:endpoint:"
            f"{hash_hmac(content='secret', key=HMAC_KEY)}:"
            f"{hash_hmac(content='42', key=HMAC_KEY)}:"
            "/v1/providers/models/sampling?provider_name=ollama"
        )
        assert key == expected

    def test_different_query_params_give_different_keys(self):
        """Test that query params are part of the key."""
        a = build(make_request(query=b"provider_name=ollama&model_name=llama3"))
        b = build(make_request(query=b"provider_name=ollama&model_name=gemma"))
        assert a != b

    def test_query_param_order_does_not_matter(self):
        """Test that query params are normalized by sorting."""
        a = build(make_request(query=b"provider_name=ollama&model_name=llama3"))
        b = build(make_request(query=b"model_name=llama3&provider_name=ollama"))
        assert a == b

    def test_repeated_query_params_are_kept(self):
        """Test that repeated params are all included in the key."""
        a = build(make_request(query=b"tag=a"))
        b = build(make_request(query=b"tag=a&tag=b"))
        assert a != b

    def test_no_query_params(self):
        """Test that a request without query params ends with an empty query."""
        key = build(make_request(path="/v1/providers"))
        assert key.endswith(":/v1/providers?")

    def test_query_values_are_encoded(self):
        """Test that special characters in values can't inject extra params."""
        a = build(make_request(query=b"q=a%26b%3Dc"))
        b = build(make_request(query=b"q=a&b=c"))
        assert a != b

    def test_different_paths_give_different_keys(self):
        """Test that the path is part of the key."""
        assert build(make_request(path="/a")) != build(make_request(path="/b"))

    def test_different_users_give_different_keys(self):
        """Test that the user id is part of the key."""
        request = make_request()
        assert build(request, user=SimpleNamespace(id=1)) != build(
            request, user=SimpleNamespace(id=2)
        )

    def test_anonymous_user(self):
        """Test that a missing user is hashed as 'Anonymous'."""
        key = build(make_request())
        assert hash_hmac(content="Anonymous", key=HMAC_KEY) in key

    def test_different_api_keys_give_different_keys(self):
        """Test that the api key is part of the key."""
        assert build(make_request(api_key="one")) != build(make_request(api_key="two"))

    def test_missing_api_key(self):
        """Test that a missing api key is hashed as 'Not Set'."""
        key = build(make_request())
        assert hash_hmac(content="Not Set", key=HMAC_KEY) in key

    def test_raw_secrets_not_in_key(self):
        """Test that the api key and user id are not stored in plain text."""
        key = build(
            make_request(api_key="super-secret-api-key"),
            user=SimpleNamespace(id="user-uuid-1234"),
        )
        assert "super-secret-api-key" not in key
        assert "user-uuid-1234" not in key

    def test_different_functions_give_different_keys(self):
        """Test that the function name is part of the key."""

        def other_endpoint():
            pass

        request = make_request()
        assert build(request) != build(request, func=other_endpoint)
