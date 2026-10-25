import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from db_helpers import fake_decrypt, fake_encrypt, make_session, row

from database.schemas.user_skills import UserSkillsT
from database.user_skills import (
    SkillDescription,
    get_user_skill_by_skill_id,
    post_user_skill,
    update_user_skill_by_skill_id,
)

pytestmark = pytest.mark.asyncio


class TestPostUserSkill:
    """Tests for post_user_skill function."""

    @patch("database.user_skills.encrypt", side_effect=fake_encrypt)
    @patch("database.user_skills.decrypt", side_effect=fake_decrypt)
    async def test_successful_skill_creation(self, mock_decrypt, mock_encrypt):
        """Test successful creation of a user skill with all components."""
        user_id = uuid.uuid4()
        skill_name = "test_skill"
        skill_description = "This is a test skill"
        skill_text = "def test_skill(): pass"
        parameters_schema = {
            "type": "object",
            "properties": {"param1": {"type": "string"}},
        }

        # Mock store_user_document to return a document_id
        document_id = uuid.uuid4()

        with (
            patch(
                "database.user_skills.store_user_document", new_callable=AsyncMock
            ) as mock_store_doc,
            patch(
                "database.user_skills.get_vector_store_settings", new_callable=AsyncMock
            ) as mock_get_vs_settings,
            patch(
                "database.user_skills.get_all_provider_configurations",
                new_callable=AsyncMock,
            ) as mock_get_providers,
            patch(
                "database.user_skills.get_vector_store", new_callable=MagicMock
            ) as mock_get_vs,
            patch(
                "database.user_skills.save_docs", new_callable=MagicMock
            ) as mock_save_docs,
            patch(
                "fastapi.concurrency.run_in_threadpool",
                side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
            ),
        ):
            # Setup mocks
            mock_store_doc.return_value = document_id

            # Mock vector store config
            mock_vs_config = MagicMock()
            mock_vs_config.vs_collection_name = "test_collection"
            mock_vs_config.vs_vendor = "qdrant"
            mock_vs_config.vs_port = 6333
            mock_vs_config.vs_base_url = "http://localhost"
            mock_vs_config.vs_encrypted_api_key = "encrypted_key"
            mock_vs_config.e_model = "test-model"
            mock_vs_config.e_provider = "openai"
            mock_vs_config.e_api_key = "api_key"
            mock_vs_config.e_dimensions = 1536
            mock_vs_config.scope = "user_vs_skills"
            mock_vs_config.required_filters = ["user-id", "scope"]
            mock_get_vs_settings.return_value = mock_vs_config

            # Mock provider configuration
            mock_provider_config = MagicMock()
            mock_provider_config.base_url = "http://provider.com"
            mock_provider_config.langchain_con = "openai"
            mock_provider_config.encrypted_api_key = "encrypted_api_key"
            mock_provider_config.api_key_short = "short_key"
            mock_provider_config.internal = False
            mock_provider_config.api_key_configured = True

            mock_registry = MagicMock()
            mock_registry.openai = mock_provider_config
            mock_get_providers.return_value = mock_registry

            # Mock vector store
            mock_vs = MagicMock()
            mock_get_vs.return_value = mock_vs

            # Mock save_docs
            mock_save_docs.return_value = ["doc_id_1"]

            # Create session
            session = make_session()

            # Call function
            result = await post_user_skill(
                session=session,
                user_id=user_id,
                name=skill_name,
                description=skill_description,
                skill_text=skill_text,
                scope="user_vs_skills",
                parameters_schema=parameters_schema,
            )

            # Assertions
            assert isinstance(result, uuid.UUID)

            # Check that store_user_document was called correctly
            mock_store_doc.assert_awaited_once_with(
                session=session,
                user_id=user_id,
                name=skill_name,
                scope="user_vs_skills",
            )

            # Check that UserSkillsT record was created
            assert len(session.added) == 1
            added_skill = session.added[0]
            assert isinstance(added_skill, UserSkillsT)
            assert added_skill.user_id == user_id
            assert added_skill.name == skill_name
            # Fields are encrypted - decrypt them for comparison
            assert fake_decrypt(added_skill.description) == skill_description
            assert fake_decrypt(added_skill.skill_text) == skill_text
            # parameters_schema is encrypted as a dict with __encrypted__ marker
            if isinstance(
                added_skill.parameters_schema, dict
            ) and added_skill.parameters_schema.get("__encrypted__"):
                decrypted_params = json.loads(
                    fake_decrypt(added_skill.parameters_schema["data"])
                )
                assert decrypted_params == parameters_schema
            else:
                assert added_skill.parameters_schema == parameters_schema

            # Check that vector store operations were called
            mock_get_vs_settings.assert_awaited_once_with(
                session=session, scope="user_vs_skills"
            )
            mock_get_providers.assert_awaited_once_with(
                session=session, user_id=user_id
            )
            mock_get_vs.assert_called_once()
            mock_save_docs.assert_called_once()

            # No explicit commit since session is context managed
            session.commit.assert_not_awaited()


@pytest.mark.usefixtures("fake_crypto")
class TestGetUserSkillBySkillId:
    """Tests for get_user_skill_by_skill_id function."""

    async def test_get_existing_skill(self):
        """Test retrieving an existing user skill."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        # Create mock skill data
        mock_skill = row(
            id=skill_id,
            user_id=user_id,
            name="test_skill",
            description="Test description",
            skill_text="def test(): pass",
            parameters_schema={"type": "object"},
            is_encrypted=False,
            created_date=None,
            created_by="test_user",
        )

        # Create session with the skill in the database
        session = make_session(execute=[mock_skill])

        # Call the function
        result = await get_user_skill_by_skill_id(
            session=session, user_id=user_id, skill_id=skill_id
        )

        # Assertions
        assert isinstance(result, SkillDescription)
        assert result.name == "test_skill"
        assert result.description == "Test description"
        assert result.instructions == "def test(): pass"
        assert result.parameter_schema == {"type": "object"}

        # Verify the query was executed
        session.execute.assert_awaited_once()

    async def test_get_nonexistent_skill(self):
        """Test that retrieving a non-existent skill raises ValueError."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        # Create session with no matching skill
        session = make_session(execute=[None])

        # Call should raise ValueError
        with pytest.raises(
            ValueError, match=f"Could not locate skill with id '{skill_id}'"
        ):
            await get_user_skill_by_skill_id(
                session=session, user_id=user_id, skill_id=skill_id
            )

    async def test_get_skill_from_different_user(self):
        """Test that accessing another user's skill raises ValueError."""
        user_id = uuid.uuid4()
        other_user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        # Create mock skill belonging to a different user
        mock_skill = row(
            id=skill_id,
            user_id=other_user_id,  # Different user
            name="other_user_skill",
            description="Other user's skill",
            skill_text="def other(): pass",
            parameters_schema={},
            is_encrypted=False,
            created_date=None,
            created_by="other_user",
        )

        # Create session - will return None because user_id doesn't match
        session = make_session(execute=[None])

        # Call should raise ValueError because the user_id doesn't match
        with pytest.raises(
            ValueError, match=f"Could not locate skill with id '{skill_id}'"
        ):
            await get_user_skill_by_skill_id(
                session=session, user_id=user_id, skill_id=skill_id
            )

    async def test_get_skill_with_empty_description(self):
        """Test retrieving a skill with empty/null description."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        mock_skill = row(
            id=skill_id,
            user_id=user_id,
            name="no_desc_skill",
            description=None,
            skill_text="def test(): pass",
            parameters_schema=None,
            is_encrypted=False,
            created_date=None,
            created_by="test_user",
        )

        session = make_session(execute=[mock_skill])

        result = await get_user_skill_by_skill_id(
            session=session, user_id=user_id, skill_id=skill_id
        )

        assert isinstance(result, SkillDescription)
        assert result.name == "no_desc_skill"
        assert result.description is None
        assert result.instructions == "def test(): pass"
        assert result.parameter_schema is None

    async def test_get_skill_with_empty_parameters_schema(self):
        """Test retrieving a skill with empty parameters_schema defaults correctly."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        mock_skill = row(
            id=skill_id,
            user_id=user_id,
            name="simple_skill",
            description="A simple skill",
            skill_text="def simple(): pass",
            parameters_schema={},
            created_date=None,
            created_by="test_user",
        )

        session = make_session(execute=[mock_skill])

        result = await get_user_skill_by_skill_id(
            session=session, user_id=user_id, skill_id=skill_id
        )

        assert isinstance(result, SkillDescription)
        assert result.parameter_schema == {}

    @patch("database.user_skills.encrypt", side_effect=fake_encrypt)
    @patch("database.user_skills.decrypt", side_effect=fake_decrypt)
    async def test_skill_creation_without_parameters_schema(
        self, mock_decrypt, mock_encrypt
    ):
        """Test skill creation without parameters_schema defaults to empty dict."""
        user_id = uuid.uuid4()

        with (
            patch(
                "database.user_skills.store_user_document", new_callable=AsyncMock
            ) as mock_store_doc,
            patch(
                "database.user_skills.get_vector_store_settings", new_callable=AsyncMock
            ) as mock_get_vs_settings,
            patch(
                "database.user_skills.get_all_provider_configurations",
                new_callable=AsyncMock,
            ) as mock_get_providers,
            patch(
                "database.user_skills.get_vector_store", new_callable=MagicMock
            ) as mock_get_vs,
            patch(
                "database.user_skills.save_docs", new_callable=MagicMock
            ) as mock_save_docs,
            patch(
                "fastapi.concurrency.run_in_threadpool",
                side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
            ),
        ):
            mock_store_doc.return_value = uuid.uuid4()

            mock_vs_config = MagicMock()
            mock_vs_config.required_filters = []
            mock_vs_config.e_provider = "openai"
            mock_get_vs_settings.return_value = mock_vs_config

            mock_registry = MagicMock()
            mock_provider_config = MagicMock()
            mock_provider_config.base_url = "http://provider.com"
            mock_provider_config.langchain_con = "openai"
            mock_registry.openai = mock_provider_config
            mock_get_providers.return_value = mock_registry

            mock_get_vs.return_value = MagicMock()
            mock_save_docs.return_value = []

            session = make_session()

            result = await post_user_skill(
                session=session,
                user_id=user_id,
                name="test_skill",
                description="test description",
                skill_text="test code",
                scope="user_vs_skills",
                # No parameters_schema provided
            )

            # Check that parameters_schema defaults to empty dict (encrypted if encrypt=True)
            added_skill = session.added[0]
            if isinstance(
                added_skill.parameters_schema, dict
            ) and added_skill.parameters_schema.get("__encrypted__"):
                # It's encrypted, decrypt it
                decrypted_params = json.loads(
                    fake_decrypt(added_skill.parameters_schema["data"])
                )
                assert decrypted_params == {}
            else:
                assert added_skill.parameters_schema == {}

    async def test_vector_store_failure_does_not_roll_back_database(self):
        """Test that database records are still created even if vector store operations fail."""
        user_id = uuid.uuid4()

        with (
            patch(
                "database.user_skills.store_user_document", new_callable=AsyncMock
            ) as mock_store_doc,
            patch(
                "database.user_skills.get_vector_store_settings", new_callable=AsyncMock
            ) as mock_get_vs_settings,
        ):
            # Setup successful document storage
            mock_store_doc.return_value = uuid.uuid4()

            # Make vector store settings fail
            mock_get_vs_settings.side_effect = Exception("Vector Store error")

            session = make_session()

            # The function now re-raises vector store exceptions
            # but database records are still created before the exception
            with pytest.raises(Exception, match="Vector Store error"):
                await post_user_skill(
                    session=session,
                    user_id=user_id,
                    name="test_skill",
                    description="test description",
                    skill_text="test code",
                    scope="user_vs_skills",
                )

            # Database records should still be created (no commit call since session is context managed)
            assert len(session.added) == 1
            assert isinstance(session.added[0], UserSkillsT)
            session.commit.assert_not_awaited()

    @patch("database.user_skills.store_user_document", new_callable=AsyncMock)
    async def test_duplicate_document_raises_error(self, mock_store_doc):
        """Test that duplicate document names raise an error."""
        user_id = uuid.uuid4()

        mock_store_doc.side_effect = ValueError(
            "File 'test_skill' with scope 'user_vs_skills' scope already exists."
        )

        session = make_session()

        with pytest.raises(
            ValueError,
            match="File 'test_skill' with scope 'user_vs_skills' scope already exists",
        ):
            await post_user_skill(
                session=session,
                user_id=user_id,
                name="test_skill",
                description="test description",
                skill_text="test code",
                scope="user_vs_skills",
            )


@pytest.mark.usefixtures("fake_crypto")
class TestUpdateUserSkillBySkillId:
    """Tests for update_user_skill_by_skill_id function."""

    async def test_successful_skill_update(self):
        """Test successfully updating a skill's instructions."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        # Create existing skill data
        existing_skill = row(
            id=skill_id,
            user_id=user_id,
            name="test_skill",
            description="Test description",
            skill_text="old instructions",
            parameters_schema={
                "type": "object",
                "properties": {"param1": {"type": "string"}},
            },
            created_date=None,
            created_by="test_user",
        )

        # Create session with the skill
        session = make_session(execute=[existing_skill])

        # New instructions to update with
        new_instructions = "new updated instructions"

        # Call the function
        result = await update_user_skill_by_skill_id(
            session=session,
            user_id=user_id,
            skill_id=skill_id,
            instructions=new_instructions,
        )

        # Assertions
        assert isinstance(result, SkillDescription)
        assert result.skill_id == skill_id
        assert result.name == "test_skill"
        assert result.description == "Test description"
        assert result.instructions == new_instructions
        assert result.parameter_schema == {
            "type": "object",
            "properties": {"param1": {"type": "string"}},
        }

        # Verify the skill text was updated in the session (now encrypted)
        assert fake_decrypt(existing_skill.skill_text) == new_instructions

        # Verify flush was called
        session.flush.assert_awaited_once()

    async def test_update_nonexistent_skill(self):
        """Test that updating a non-existent skill raises ValueError."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        # Create session with no matching skill
        session = make_session(execute=[None])

        # Call should raise ValueError
        with pytest.raises(
            ValueError, match=f"Could not locate skill with id '{skill_id}'"
        ):
            await update_user_skill_by_skill_id(
                session=session,
                user_id=user_id,
                skill_id=skill_id,
                instructions="new instructions",
            )

    async def test_update_skill_from_different_user(self):
        """Test that updating a skill from a different user raises ValueError."""
        user_id = uuid.uuid4()
        other_user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        # Create mock skill belonging to a different user
        mock_skill = row(
            id=skill_id,
            user_id=other_user_id,  # Different user
            name="other_user_skill",
            description="Other user's skill",
            skill_text="def other(): pass",
            parameters_schema={},
            created_date=None,
            created_by="other_user",
        )

        # Session will return None because the user_id in the where clause doesn't match
        session = make_session(execute=[None])

        # Call should raise ValueError because no matching skill found for the given user_id
        with pytest.raises(
            ValueError, match=f"Could not locate skill with id '{skill_id}'"
        ):
            await update_user_skill_by_skill_id(
                session=session,
                user_id=user_id,
                skill_id=skill_id,
                instructions="new instructions",
            )

    async def test_update_with_empty_instructions(self):
        """Test updating a skill with empty instructions."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        existing_skill = row(
            id=skill_id,
            user_id=user_id,
            name="empty_instructions_skill",
            description="A skill",
            skill_text="old instructions",
            parameters_schema={},
            created_date=None,
            created_by="test_user",
        )

        session = make_session(execute=[existing_skill])

        result = await update_user_skill_by_skill_id(
            session=session,
            user_id=user_id,
            skill_id=skill_id,
            instructions="",  # Empty instructions
        )

        assert isinstance(result, SkillDescription)
        assert result.instructions == ""
        assert fake_decrypt(existing_skill.skill_text) == ""

    async def test_update_with_special_characters(self):
        """Test updating a skill with special characters in instructions."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        existing_skill = row(
            id=skill_id,
            user_id=user_id,
            name="special_chars_skill",
            description="A skill",
            skill_text="old",
            parameters_schema={},
            created_date=None,
            created_by="test_user",
        )

        session = make_session(execute=[existing_skill])

        special_instructions = (
            "def skill():\n    # Special chars: !@#$%^&*()\n    return 'test'"
        )

        result = await update_user_skill_by_skill_id(
            session=session,
            user_id=user_id,
            skill_id=skill_id,
            instructions=special_instructions,
        )

        assert result.instructions == special_instructions
        assert fake_decrypt(existing_skill.skill_text) == special_instructions

    async def test_update_preserves_other_fields(self):
        """Test that updating instructions doesn't affect other fields."""
        user_id = uuid.uuid4()
        skill_id = uuid.uuid4()

        original_name = "original_name"
        original_description = "original description"
        original_params = {"type": "object", "properties": {"x": {"type": "number"}}}

        existing_skill = row(
            id=skill_id,
            user_id=user_id,
            name=original_name,
            description=original_description,
            skill_text="old instructions",
            parameters_schema=original_params,
            created_date=None,
            created_by="test_user",
        )

        session = make_session(execute=[existing_skill])

        result = await update_user_skill_by_skill_id(
            session=session,
            user_id=user_id,
            skill_id=skill_id,
            instructions="new instructions",
        )

        # Other fields should be preserved
        assert result.name == original_name
        assert result.description == original_description
        assert result.parameter_schema == original_params
        assert result.skill_id == skill_id

        # Only skill_text should change
        assert result.instructions == "new instructions"
        assert fake_decrypt(existing_skill.skill_text) == "new instructions"
        assert existing_skill.name == original_name
        # Other fields are now also encrypted
        assert fake_decrypt(existing_skill.description) == original_description
        if isinstance(
            existing_skill.parameters_schema, dict
        ) and existing_skill.parameters_schema.get("__encrypted__"):
            import json

            decrypted_params = json.loads(
                fake_decrypt(existing_skill.parameters_schema["data"])
            )
            assert decrypted_params == original_params
        else:
            assert existing_skill.parameters_schema == original_params
