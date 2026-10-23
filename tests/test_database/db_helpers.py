import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


def fake_encrypt(content: str) -> str:
    return f"enc:{content}"


def fake_decrypt(content: str) -> str:
    return content.removeprefix("enc:")


def make_session(execute=(), scalar=(), scalars=()):
    """Mock AsyncSession. Each argument is the ordered list of values returned by
    successive execute().scalar_one_or_none(), scalar(), and scalars().all() calls."""
    session = MagicMock()
    session.added = []

    def _execute_result(row):
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        result.all.return_value = row
        return result

    def _scalars_result(rows):
        result = MagicMock()
        result.all.return_value = rows
        return result

    async def _flush():
        # Mimic the database assigning primary keys on flush.
        for obj in session.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    session.execute = AsyncMock(side_effect=[_execute_result(r) for r in execute])
    session.scalar = AsyncMock(side_effect=list(scalar))
    session.scalars = AsyncMock(side_effect=[_scalars_result(r) for r in scalars])
    session.add = MagicMock(side_effect=session.added.append)
    session.flush = AsyncMock(side_effect=_flush)
    session.delete = AsyncMock()
    session.begin_nested = AsyncMock()
    session.commit = AsyncMock()
    return session


def params(mock_call) -> list:
    """Bound parameter values of the SQL statement passed in a mock call."""
    return list(mock_call.args[0].compile().params.values())


def row(**kwargs) -> SimpleNamespace:
    # For user_skills compatibility, add is_encrypted=False if not present
    if 'is_encrypted' not in kwargs and 'description' in kwargs:
        kwargs['is_encrypted'] = False
    return SimpleNamespace(**kwargs)
