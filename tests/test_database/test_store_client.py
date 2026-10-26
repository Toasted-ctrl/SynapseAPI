import pytest
from db_helpers import make_session, params

from core.config import config
from database.schemas.clients import ClientsT
from database.store_client import StoredClient, store_client
from security.hash import get_hash_sha256
from security.hmac import hash_hmac

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("fake_crypto")]


def _kwargs(**overrides):
    kwargs = dict(
        client_name="my-app", key_type="User", owner_email="owner@example.com"
    )
    kwargs.update(overrides)
    return kwargs


class TestStoreClient:
    async def test_stores_client_with_generated_secrets(self):
        session = make_session(scalar=[0])

        result = await store_client(session=session, **_kwargs())

        [added] = session.added
        assert isinstance(added, ClientsT)
        assert isinstance(result, StoredClient)
        assert result.id == added.id
        assert result.key_type == "User"
        assert result.owner_email == "owner@example.com"
        assert added.api_key_hash == get_hash_sha256(result.api_key)
        assert added.encrypted_hmac_secret == f"enc:{result.hmac_secret}"
        assert added.encrypted_client_name == "enc:my-app"
        assert added.encrypted_owner_email == "enc:owner@example.com"
        assert added.encrypted_redirect_uri is None
        session.flush.assert_awaited_once()

    async def test_uses_provided_api_key_and_hmac_secret(self):
        session = make_session(scalar=[0])

        result = await store_client(
            session=session, **_kwargs(api_key="given-key", hmac_secret="given-hmac")
        )

        [added] = session.added
        assert result.api_key == "given-key"
        assert result.hmac_secret == "given-hmac"
        assert added.api_key_hash == get_hash_sha256("given-key")

    async def test_encrypts_redirect_uri_when_provided(self):
        session = make_session(scalar=[0])

        await store_client(
            session=session,
            **_kwargs(key_type="Application", redirect_uri="https://app/cb"),
        )

        assert session.added[0].encrypted_redirect_uri == "enc:https://app/cb"

    async def test_sets_blind_indexes_and_flags(self):
        session = make_session(scalar=[0])

        await store_client(
            session=session,
            **_kwargs(require_jwt=False, require_external_id=False, is_active=False),
        )

        [added] = session.added
        assert added.blind_index_client_name == hash_hmac(
            content="my-app", key=config.BLIND_INDEX_HMAC_KEY
        )
        assert added.blind_index_owner_email == hash_hmac(
            content="owner@example.com", key=config.BLIND_INDEX_HMAC_KEY
        )
        assert added.require_jwt is False
        assert added.require_external_id is False
        assert added.is_active is False

    async def test_duplicate_check_uses_blind_indexes(self):
        session = make_session(scalar=[0])

        await store_client(session=session, **_kwargs())

        assert set(params(session.scalar.await_args)) == {
            hash_hmac(content="my-app", key=config.BLIND_INDEX_HMAC_KEY),
            hash_hmac(content="owner@example.com", key=config.BLIND_INDEX_HMAC_KEY),
        }

    async def test_raises_when_client_already_exists(self):
        session = make_session(scalar=[1])

        with pytest.raises(ValueError, match="already exists"):
            await store_client(session=session, **_kwargs())

        session.add.assert_not_called()

    async def test_rejects_invalid_key_type(self):
        session = make_session()

        with pytest.raises(ValueError, match="Key_type must be"):
            await store_client(session=session, **_kwargs(key_type="Admin"))

        session.scalar.assert_not_awaited()

    @pytest.mark.parametrize("email", ["", None])
    async def test_rejects_missing_owner_email(self, email):
        session = make_session()

        with pytest.raises(ValueError, match="valid owner email"):
            await store_client(session=session, **_kwargs(owner_email=email))
