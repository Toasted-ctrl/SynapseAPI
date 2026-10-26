import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from db_helpers import make_session, params, row
from sqlalchemy.exc import IntegrityError

from core.config import config
from database.person import (
    Person,
    PersonDetails,
    get_or_store_person,
    get_person_by_person_id,
)
from database.schemas.persons_users import PersonsT
from security.hmac import hash_hmac

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("fake_crypto")]

EMAIL = "jane@example.com"


def _blind_index(email: str) -> str:
    return hash_hmac(content=email, key=config.BLIND_INDEX_HMAC_KEY)


class TestGetPersonByPersonId:
    async def test_returns_decrypted_details(self):
        person_id = uuid.uuid4()
        session = make_session(
            execute=[
                row(
                    id=person_id,
                    encrypted_first_name="enc:Jane",
                    encrypted_last_name="enc:Doe",
                    encrypted_email=f"enc:{EMAIL}",
                )
            ]
        )

        result = await get_person_by_person_id(session=session, person_id=person_id)

        assert result == PersonDetails(
            person_id=person_id, first_name="Jane", last_name="Doe", email=EMAIL
        )
        assert params(session.execute.await_args) == [person_id]

    async def test_returns_none_when_not_found(self):
        session = make_session(execute=[None])

        assert (
            await get_person_by_person_id(session=session, person_id=uuid.uuid4())
            is None
        )


class TestGetOrStorePerson:
    async def test_returns_existing_person_by_blind_index(self):
        existing_id = uuid.uuid4()
        session = make_session(scalar=[row(id=existing_id)])

        result = await get_or_store_person(
            session=session, first_name="Jane", last_name="Doe", email=EMAIL
        )

        assert result == Person(id=existing_id)
        assert params(session.scalar.await_args)[0] == _blind_index(EMAIL)
        session.add.assert_not_called()

    async def test_creates_encrypted_person_in_savepoint(self):
        session = make_session(scalar=[None])

        result = await get_or_store_person(
            session=session, first_name="Jane", last_name="Doe", email=EMAIL
        )

        [added] = session.added
        assert isinstance(added, PersonsT)
        assert added.encrypted_first_name == "enc:Jane"
        assert added.encrypted_last_name == "enc:Doe"
        assert added.encrypted_email == f"enc:{EMAIL}"
        assert added.blind_index_email == _blind_index(EMAIL)
        session.begin_nested.assert_awaited_once()
        assert result == Person(id=added.id)

    async def test_concurrent_insert_rolls_back_savepoint_and_returns_winner(self):
        winner_id = uuid.uuid4()
        session = make_session(scalar=[None, row(id=winner_id)])
        nested = MagicMock(rollback=AsyncMock())
        session.begin_nested = AsyncMock(return_value=nested)
        session.flush = AsyncMock(
            side_effect=IntegrityError("INSERT", {}, Exception("duplicate"))
        )

        result = await get_or_store_person(
            session=session, first_name="Jane", last_name="Doe", email=EMAIL
        )

        nested.rollback.assert_awaited_once()
        assert result == Person(id=winner_id)

    async def test_concurrent_insert_without_winner_raises(self):
        session = make_session(scalar=[None, None])
        session.begin_nested = AsyncMock(return_value=MagicMock(rollback=AsyncMock()))
        session.flush = AsyncMock(
            side_effect=IntegrityError("INSERT", {}, Exception("duplicate"))
        )

        with pytest.raises(ValueError, match="Failed to create or retrieve person"):
            await get_or_store_person(
                session=session, first_name="Jane", last_name="Doe", email=EMAIL
            )
