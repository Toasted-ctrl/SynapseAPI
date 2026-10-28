import pytest
from fastapi import HTTPException

from api.v1.inference.v1_decision import _render_state
from api.v1.playground.models import (
    DecisionState,
    DecisionStateField,
    DecisionStateType,
)

TEXT = DecisionState()
FIELDS = DecisionState(
    type=DecisionStateType.FIELDS,
    fields=[DecisionStateField(name="ticket"), DecisionStateField(name="days")],
)


def test_text_state_is_passed_through():
    assert _render_state("Ticket #4128", TEXT) == "Ticket #4128"


@pytest.mark.parametrize("state", ["", "   ", {"ticket": "x"}])
def test_text_state_rejects_empty_or_object(state):
    with pytest.raises(HTTPException) as exc:
        _render_state(state, TEXT)
    assert exc.value.status_code == 400


def test_fields_state_is_rendered_in_agent_order():
    assert (
        _render_state({"days": "11", "ticket": "#4128"}, FIELDS)
        == "ticket: #4128\ndays: 11"
    )


def test_fields_state_rejects_string():
    with pytest.raises(HTTPException) as exc:
        _render_state("Ticket", FIELDS)
    assert exc.value.status_code == 400


def test_fields_state_reports_missing_and_unknown():
    with pytest.raises(HTTPException) as exc:
        _render_state({"ticket": "#4128", "extra": "x"}, FIELDS)
    assert "missing fields: days" in exc.value.detail
    assert "unknown fields: extra" in exc.value.detail
