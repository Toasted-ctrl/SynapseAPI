from typing import Any

import httpx

from api.v1.playground.models import (
    ChoiceQuestion,
    DecisionQuestion,
    NoulQuestion,
    ScoreQuestion,
)
from core.logging import get_logger
from security.encryption import decrypt

log = get_logger()

# A decision can take a lot longer than httpx's 5 second default.
DECISION_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def _to_question(question: DecisionQuestion) -> dict[str, Any]:
    """Translates a stored question into the shape the System One endpoint expects."""
    body: dict[str, Any] = {
        "type": question.type,
        "instructions": question.instructions,
    }
    match question:
        case ChoiceQuestion():
            # Options are stored without descriptions, so the name doubles as one.
            body["criteria"] = {option: option for option in question.options}
        case ScoreQuestion():
            body["legend"] = question.legend
        case NoulQuestion():
            pass
    return body


async def get_decision(
    base_url: str,
    encrypted_api_key: str,
    model: str,
    state: str,
    questions: list[DecisionQuestion],
) -> dict[str, dict[str, Any]]:
    """Requests a non-streaming decision and returns its answers keyed by question name.
    Raises httpx.HTTPStatusError if the provider rejects the request."""
    log.debug(f"Requesting decision from System One model '{model}' ...")

    async with httpx.AsyncClient(base_url=base_url, timeout=DECISION_TIMEOUT) as client:
        response = await client.post(
            "/systemone",
            headers={"Authorization": f"Bearer {decrypt(encrypted_api_key)}"},
            json={
                "model": model,
                "state": state,
                "questions": {q.name: _to_question(q) for q in questions},
            },
        )
        response.raise_for_status()

    return response.json()["answers"]
