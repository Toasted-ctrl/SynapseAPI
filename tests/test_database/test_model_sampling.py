import pytest
from db_helpers import make_session, row

from database.model_sampling import ModelSampling, get_model_sampling_settings

pytestmark = pytest.mark.asyncio


class TestGetModelSamplingSettings:
    async def test_returns_all_supported(self):
        session = make_session(
            scalars=[
                [
                    row(parameter="temperature", supported=True),
                    row(parameter="top_k", supported=True),
                    row(parameter="top_p", supported=True),
                ]
            ]
        )

        result = await get_model_sampling_settings(
            session=session, provider_name="openai", model_name="gpt-4"
        )

        assert result == ModelSampling(temperature=True, top_k=True, top_p=True)

    async def test_returns_none_supported(self):
        session = make_session(
            scalars=[
                [
                    row(parameter="temperature", supported=False),
                    row(parameter="top_k", supported=False),
                    row(parameter="top_p", supported=False),
                ]
            ]
        )

        result = await get_model_sampling_settings(
            session=session, provider_name="openai", model_name="gpt-4"
        )

        assert result == ModelSampling(temperature=False, top_k=False, top_p=False)

    async def test_returns_mixed_support(self):
        session = make_session(
            scalars=[
                [
                    row(parameter="temperature", supported=True),
                    row(parameter="top_k", supported=False),
                    row(parameter="top_p", supported=True),
                ]
            ]
        )

        result = await get_model_sampling_settings(
            session=session, provider_name="anthropic", model_name="claude-3"
        )

        assert result == ModelSampling(temperature=True, top_k=False, top_p=True)

    async def test_returns_defaults_when_no_records(self):
        session = make_session(scalars=[[]])

        result = await get_model_sampling_settings(
            session=session, provider_name="mistral", model_name="mixtral-8x7b"
        )

        assert result == ModelSampling()

    async def test_returns_defaults_when_only_some_parameters_exist(self):
        session = make_session(
            scalars=[
                [
                    row(parameter="temperature", supported=True),
                    row(parameter="top_p", supported=False),
                ]
            ]
        )

        result = await get_model_sampling_settings(
            session=session, provider_name="google", model_name="gemini-1.5"
        )

        assert result == ModelSampling(temperature=True, top_k=False, top_p=False)
