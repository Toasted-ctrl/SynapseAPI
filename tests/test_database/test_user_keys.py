import uuid
from datetime import datetime, timedelta

import pytest

from database.schemas.user_keys import UserKeysT
from database.user_keys import ProviderAPIKey, get_or_store_key, get_user_active_keys
from db_helpers import make_session, params, row


pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("fake_crypto")]


def _key_row(user_id, provider_id, api_key="sk-1234567890abc"):
    return row(
        id=uuid.uuid4(),
        user_id=user_id,
        provider_id=provider_id,
        encrypted_api_key=f"enc:{api_key}",
        api_key_short=api_key[:10],
        expiration_date=datetime.now() + timedelta(days=10)
    )


class TestGetUserActiveKeys:

    async def test_returns_keys_with_encrypted_value(self):
        user_id = uuid.uuid4()
        rows = [_key_row(user_id, uuid.uuid4()), _key_row(user_id, uuid.uuid4())]
        session = make_session(scalars=[rows])

        result = await get_user_active_keys(session=session, user_id=user_id)

        assert result == [
            ProviderAPIKey(
                user_id=r.user_id,
                provider_id=r.provider_id,
                api_key_id=r.id,
                api_key=r.encrypted_api_key,
                api_key_short=r.api_key_short,
                expiration_date=r.expiration_date
            )
            for r in rows
        ]

    async def test_returns_empty_list_when_no_keys(self):
        session = make_session(scalars=[[]])

        assert await get_user_active_keys(session=session, user_id=uuid.uuid4()) == []

    async def test_filters_on_user_and_unexpired(self):
        user_id = uuid.uuid4()
        session = make_session(scalars=[[]])

        await get_user_active_keys(session=session, user_id=user_id)

        bound = params(session.scalars.await_args)
        assert bound[0] == user_id
        assert isinstance(bound[1], datetime)


class TestGetOrStoreKey:

    async def test_returns_existing_key_decrypted(self):
        user_id, provider_id = uuid.uuid4(), uuid.uuid4()
        existing = _key_row(user_id, provider_id, api_key="sk-existing-key")
        session = make_session(scalar=[existing])

        result = await get_or_store_key(session=session, api_key="sk-new", user_id=user_id, provider_id=provider_id)

        assert result.api_key == "sk-existing-key"
        assert result.api_key_id == existing.id
        assert result.provider_id == provider_id
        session.add.assert_not_called()

    async def test_stores_new_key(self):
        user_id, provider_id = uuid.uuid4(), uuid.uuid4()
        session = make_session(scalar=[None, 1])

        result = await get_or_store_key(
            session=session,
            api_key="sk-1234567890abcdef",
            user_id=user_id,
            provider_id=provider_id
        )

        [added] = session.added
        assert isinstance(added, UserKeysT)
        assert added.user_id == user_id
        assert added.provider_id == provider_id
        assert added.encrypted_api_key == "enc:sk-1234567890abcdef"
        assert added.api_key_short == "sk-1234567"
        session.flush.assert_awaited_once()
        assert result.api_key == "sk-1234567890abcdef"
        assert result.api_key_short == "sk-1234567"

    async def test_provider_check_filters_on_provider_requiring_key(self):
        provider_id = uuid.uuid4()
        session = make_session(scalar=[None, 1])

        await get_or_store_key(session=session, api_key="sk-abc", user_id=uuid.uuid4(), provider_id=provider_id)

        provider_check = session.scalar.await_args_list[1]
        assert params(provider_check) == [provider_id]
        assert "providers.requires_api_key = true" in str(provider_check.args[0])

    async def test_rejects_provider_without_key_requirement(self):
        provider_id = uuid.uuid4()
        session = make_session(scalar=[None, 0])

        with pytest.raises(ValueError, match=str(provider_id)):
            await get_or_store_key(session=session, api_key="sk-abc", user_id=uuid.uuid4(), provider_id=provider_id)

        session.add.assert_not_called()
