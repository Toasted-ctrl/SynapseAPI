from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from api.v1.admin.models import v1_add_model_with_expertise as add_module
from api.v1.admin.models import v1_delete_model as delete_module
from auth.dep_verify_admin import depends_verify_admin
from core.model_types import ModelType
from database.models import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelWithExpertise,
)
from database.session import get_db_session

CREATED = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)


def _client(admin_ok: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(add_module.router)
    app.include_router(delete_module.router)

    async def fake_admin():
        if not admin_ok:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

    async def fake_session():
        yield object()

    app.dependency_overrides[depends_verify_admin] = fake_admin
    app.dependency_overrides[get_db_session] = fake_session
    return TestClient(app)


@pytest.fixture
def add_model():
    with patch.object(add_module, "add_model_with_expertise", new=AsyncMock()) as m:
        yield m


@pytest.fixture
def delete_model():
    with patch.object(delete_module, "delete_model_by_name", new=AsyncMock()) as m:
        yield m


@pytest.fixture
def invalidate():
    # Both modules share the same model_config instance.
    with patch.object(add_module.model_config, "invalidate") as m:
        yield m


def test_creates_model(add_model, invalidate):
    add_model.return_value = ModelWithExpertise(
        model_name="llama3", expertise="chat_completion", created_date=CREATED
    )

    response = _client().post(
        "/admin/models", json={"model_name": "llama3", "expertise": "chat_completion"}
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json() == {
        "model_name": "llama3",
        "expertise": "chat_completion",
        "created_date": "2026-11-27T12:00:00Z",
    }
    assert add_model.await_args.kwargs["model_name"] == "llama3"
    assert add_model.await_args.kwargs["expertise"] == ModelType.CHAT_COMPLETION
    invalidate.assert_called_once_with(ModelType.CHAT_COMPLETION)


def test_duplicate_returns_conflict(add_model, invalidate):
    add_model.side_effect = ModelAlreadyExistsError("Model llama3 already exists.")

    response = _client().post(
        "/admin/models", json={"model_name": "llama3", "expertise": "translation"}
    )

    assert response.status_code == status.HTTP_409_CONFLICT
    assert "already exists" in response.json()["detail"]
    invalidate.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {"model_name": "llama3", "expertise": "not_an_expertise"},
        {"model_name": "", "expertise": "chat_completion"},
        {"model_name": "x" * 256, "expertise": "chat_completion"},
        {"expertise": "chat_completion"},
    ],
)
def test_invalid_body_is_rejected(add_model, body):
    response = _client().post("/admin/models", json=body)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    add_model.assert_not_awaited()


def test_requires_admin(add_model):
    response = _client(admin_ok=False).post(
        "/admin/models", json={"model_name": "llama3", "expertise": "chat_completion"}
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    add_model.assert_not_awaited()


def test_deletes_model(delete_model, invalidate):
    delete_model.return_value = ModelWithExpertise(
        model_name="llama3:8b", expertise="translation", created_date=CREATED
    )

    response = _client().delete("/admin/models", params={"model_name": "llama3:8b"})

    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert response.content == b""
    assert delete_model.await_args.kwargs["model_name"] == "llama3:8b"
    invalidate.assert_called_once_with("translation")


def test_delete_unknown_model_returns_not_found(delete_model, invalidate):
    delete_model.side_effect = ModelNotFoundError("Model llama3 does not exist.")

    response = _client().delete("/admin/models", params={"model_name": "llama3"})

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Model llama3 does not exist."
    invalidate.assert_not_called()


@pytest.mark.parametrize("params", [{}, {"model_name": ""}, {"model_name": "x" * 256}])
def test_delete_invalid_name_is_rejected(delete_model, params):
    response = _client().delete("/admin/models", params=params)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    delete_model.assert_not_awaited()


def test_delete_requires_admin(delete_model):
    response = _client(admin_ok=False).delete(
        "/admin/models", params={"model_name": "llama3"}
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    delete_model.assert_not_awaited()
