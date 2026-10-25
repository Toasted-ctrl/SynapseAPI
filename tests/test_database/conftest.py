import pytest
from db_helpers import fake_decrypt, fake_encrypt


@pytest.fixture
def fake_crypto(monkeypatch):
    """Replaces encrypt/decrypt in every database module with a reversible fake."""
    import database.person
    import database.store_client
    import database.user_keys
    import database.user_skills

    for module in (
        database.person,
        database.store_client,
        database.user_keys,
        database.user_skills,
    ):
        if hasattr(module, "encrypt"):
            monkeypatch.setattr(module, "encrypt", fake_encrypt)
        if hasattr(module, "decrypt"):
            monkeypatch.setattr(module, "decrypt", fake_decrypt)
