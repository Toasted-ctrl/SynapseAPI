import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from api.v1.admin.tools import v1_mcps as mcps_module
from auth.dep_verify_admin import depends_verify_admin
from database.mcps import MCPConfig, MCPNotFoundError
from database.session import get_db_session

MCP_ID = uuid.uuid4()
BODY = {"name": "search", "url": "http://mcp.local/mcp"}


def _client(admin_ok: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(mcps_module.router)

    async def fake_admin():
        if not admin_ok:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

    async def fake_session():
        yield object()

    app.dependency_overrides[depends_verify_admin] = fake_admin
    app.dependency_overrides[get_db_session] = fake_session
    return TestClient(app)


@pytest.fixture
def insert_mcp():
    with patch.object(mcps_module, "insert_mcp", new=AsyncMock()) as m:
        yield m


@pytest.fixture
def delete_mcp():
    with patch.object(mcps_module, "delete_mcp_by_id", new=AsyncMock()) as m:
        yield m


def test_adds_mcp(insert_mcp):
    insert_mcp.return_value = MCPConfig(
        id=MCP_ID,
        name="search",
        url="http://mcp.local/mcp",
        transport="streamable_http",
    )

    response = _client().post("/admin/tools/mcp", json=BODY)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        **BODY,
        "transport": "streamable_http",
        "id": str(MCP_ID),
    }
    assert insert_mcp.await_args.kwargs["mcp_config"] == MCPConfig(
        name="search", url="http://mcp.local/mcp", transport="streamable_http"
    )


def test_add_rejects_unsupported_transport(insert_mcp):
    response = _client().post("/admin/tools/mcp", json={**BODY, "transport": "stdio"})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    insert_mcp.assert_not_awaited()


def test_deletes_mcp(delete_mcp):
    response = _client().delete("/admin/tools/mcp", params={"mcp_id": str(MCP_ID)})

    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert delete_mcp.await_args.kwargs["mcp_id"] == MCP_ID


def test_delete_unknown_mcp_returns_not_found(delete_mcp):
    delete_mcp.side_effect = MCPNotFoundError(f"MCP with ID {MCP_ID} does not exist.")

    response = _client().delete("/admin/tools/mcp", params={"mcp_id": str(MCP_ID)})

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "does not exist" in response.json()["detail"]


@pytest.mark.parametrize(
    "method, kwargs",
    [("post", {"json": BODY}), ("delete", {"params": {"mcp_id": str(MCP_ID)}})],
)
def test_requires_admin(insert_mcp, delete_mcp, method, kwargs):
    response = getattr(_client(admin_ok=False), method)("/admin/tools/mcp", **kwargs)

    assert response.status_code == status.HTTP_403_FORBIDDEN
    insert_mcp.assert_not_awaited()
    delete_mcp.assert_not_awaited()
