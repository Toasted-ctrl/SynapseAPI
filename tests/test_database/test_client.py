import uuid

import pytest
from db_helpers import make_session, params, row

from database.client import ApplicationClient, get_client_from_client_id

pytestmark = pytest.mark.asyncio


class TestGetClientFromClientId:
    async def test_returns_application_client(self):
        client_id = uuid.uuid4()
        session = make_session(
            execute=[row(id=client_id, encrypted_redirect_uri="enc:https://app")]
        )

        result = await get_client_from_client_id(session=session, client_id=client_id)

        assert result == ApplicationClient(
            id=client_id, encrypted_redirect_uri="enc:https://app"
        )
        assert params(session.execute.await_args) == [client_id]

    async def test_raises_lookup_error_when_not_found(self):
        session = make_session(execute=[None])

        with pytest.raises(LookupError, match="Invalid Client ID"):
            await get_client_from_client_id(session=session, client_id=uuid.uuid4())
