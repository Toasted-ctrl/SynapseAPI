from unittest.mock import patch

import httpx
import pytest

from api.v1.playground.models import ChoiceQuestion, NoulQuestion, ScoreQuestion
from providers.melious.system_one import _to_question, get_decision

pytestmark = pytest.mark.asyncio

QUESTIONS = [
    ChoiceQuestion(
        name="resolution",
        instructions="Pick the support resolution that matches policy.",
        options=["refund", "credit", "deny"],
    ),
    ScoreQuestion(
        name="urgency", instructions="How urgent is it?", legend=["low", "high"]
    ),
    NoulQuestion(name="fraud", instructions="How likely is fraud?"),
]


async def test_to_question_maps_each_type():
    assert _to_question(QUESTIONS[0]) == {
        "type": "choice",
        "instructions": "Pick the support resolution that matches policy.",
        "criteria": {"refund": "refund", "credit": "credit", "deny": "deny"},
    }
    assert _to_question(QUESTIONS[1])["legend"] == ["low", "high"]
    assert _to_question(QUESTIONS[2]) == {
        "type": "noul",
        "instructions": "How likely is fraud?",
    }


async def test_get_decision_posts_to_systemone_and_returns_answers():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"answers": {"resolution": {"choice": "credit"}}}
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    with (
        patch("providers.melious.system_one.decrypt", return_value="sk-mel-test"),
        patch(
            "providers.melious.system_one.httpx.AsyncClient",
            lambda **kw: real_client(transport=transport, **kw),
        ),
    ):
        answers = await get_decision(
            base_url="https://api.melious.ai/v1",
            encrypted_api_key="encrypted",
            model="decision-model",
            state="Ticket #4128",
            questions=QUESTIONS[:1],
        )

    assert answers == {"resolution": {"choice": "credit"}}
    request = seen[0]
    assert str(request.url) == "https://api.melious.ai/v1/systemone"
    assert request.headers["Authorization"] == "Bearer sk-mel-test"


async def test_get_decision_raises_on_error_status():
    transport = httpx.MockTransport(lambda r: httpx.Response(422, json={}))
    real_client = httpx.AsyncClient

    with (
        patch("providers.melious.system_one.decrypt", return_value="k"),
        patch(
            "providers.melious.system_one.httpx.AsyncClient",
            lambda **kw: real_client(transport=transport, **kw),
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await get_decision(
            base_url="https://api.melious.ai/v1",
            encrypted_api_key="encrypted",
            model="m",
            state="s",
            questions=QUESTIONS,
        )
