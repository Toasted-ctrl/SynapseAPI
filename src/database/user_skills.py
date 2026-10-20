import json
import uuid
from dataclasses import dataclass
from typing import Any
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from database.documents_users import store_user_document
from database.providers import get_all_provider_configurations, UserProviderRegistry, ProviderConfiguration
from database.vector_store import get_vector_store_settings, VectorStoreConfig
from security.encryption import decrypt, encrypt
from vs.delete_document import delete_document_from_qdrant_by_user_document_id
from vs.get_vs import get_vector_store
from vs.save_docs import save_docs, VectorStoreScope

from .schemas.user_skills import UserSkillsT


log = get_logger()


# Fields to encrypt for user skills
ENCRYPTED_FIELDS = {"description", "skill_text", "parameters_schema"}


def _should_encrypt(field_name: str) -> bool:
    """Check if a field should be encrypted."""
    return field_name in ENCRYPTED_FIELDS


def _encrypt_field(data: Any) -> Any:
    """Encrypts a field value.
    
    For strings: encrypts directly
    For dicts: encrypts as JSON string with __encrypted__ marker
    For None: returns None
    """
    if data is None:
        return None
    
    if isinstance(data, dict):
        # Encrypt dict as JSON string with marker
        return {
            "__encrypted__": True,
            "data": encrypt(json.dumps(data))
        }
    
    if isinstance(data, str):
        return encrypt(data)
    
    # For other types, convert to string first
    return encrypt(str(data))


def _decrypt_field(data: Any, is_encrypted: bool = False) -> Any:
    """Decrypts a field value.
    
    Args:
        data: The field value to decrypt
        is_encrypted: Whether the data is known to be encrypted
        
    Returns:
        Decrypted value
    """
    if data is None:
        return None
    
    # Handle dict with __encrypted__ marker
    if isinstance(data, dict) and data.get("__encrypted__"):
        encrypted_str = data.get("data", "")
        if encrypted_str:
            decrypted_json = decrypt(encrypted_str)
            return json.loads(decrypted_json)
        return {}
    
    # If is_encrypted flag is True but no marker, try to decrypt as string
    if is_encrypted and isinstance(data, str):
        try:
            return decrypt(data)
        except Exception:
            # If decryption fails, return as-is (might be unencrypted old data)
            return data
    
    # Not encrypted, return as-is
    return data


def _encrypt_skill_data(
    name: str,
    description: str | None,
    skill_text: str,
    parameters_schema: dict | None,
    is_encrypted: bool = True
) -> dict[str, Any]:
    """Encrypts skill data for storage.
    
    Returns a dict with encrypted values for the specified fields.
    """
    result = {}
    
    if _should_encrypt("description") and is_encrypted:
        result["description"] = _encrypt_field(description)
    else:
        result["description"] = description
    
    if _should_encrypt("skill_text") and is_encrypted:
        result["skill_text"] = _encrypt_field(skill_text)
    else:
        result["skill_text"] = skill_text
    
    if _should_encrypt("parameters_schema") and is_encrypted:
        result["parameters_schema"] = _encrypt_field(parameters_schema)
    else:
        result["parameters_schema"] = parameters_schema
    
    result["name"] = name
    result["is_encrypted"] = is_encrypted
    
    return result


def _decrypt_skill_data_from_row(skill: UserSkillsT) -> dict[str, Any]:
    """Decrypts skill data from a database row.
    
    Checks the is_encrypted flag and decrypts fields accordingly.
    """
    result = {}
    
    # Always include non-encrypted fields
    result["id"] = skill.id
    result["user_id"] = skill.user_id
    result["name"] = skill.name
    result["is_encrypted"] = skill.is_encrypted
    result["created_date"] = skill.created_date
    result["created_by"] = skill.created_by
    
    # Decrypt encrypted fields
    result["description"] = _decrypt_field(skill.description, skill.is_encrypted)
    result["skill_text"] = _decrypt_field(skill.skill_text, skill.is_encrypted)
    result["parameters_schema"] = _decrypt_field(skill.parameters_schema, skill.is_encrypted)
    
    return result


async def post_user_skill(
    session: AsyncSession,
    user_id: uuid.UUID,
    name: str,
    description: str,
    skill_text: str,
    scope: str,
    parameters_schema: dict | None = None,
    encrypt: bool = True
) -> uuid.UUID:
    """Creates a user skill record, stores it in the UserSkillsT table, and saves the description to the Vector Store.
    
    Args:
        session: Async database session
        user_id: The user's UUID
        name: Name of the skill
        description: Description of the skill
        skill_text: The actual skill implementation/content
        scope: The scope (e.g., "user_vs_skills")
        parameters_schema: Optional JSON schema for skill parameters
        encrypt: Whether to encrypt the sensitive fields (default: True)
        
    Returns:
        The ID of the created UserSkillsT record
    """
    
    # Step 1: Create document record with scope user_vs_skills
    document_id = await store_user_document(
        session=session,
        user_id=user_id,
        name=name,
        scope="user_vs_skills"
    )
    
    log.debug(f"Created user document with ID '{document_id}' for skill '{name}'")
    
    # Step 2: Encrypt data if requested
    encrypted_data = _encrypt_skill_data(
        name=name,
        description=description,
        skill_text=skill_text,
        parameters_schema=parameters_schema or {},
        is_encrypted=encrypt
    )
    
    # Step 3: Create UserSkillsT entry with encrypted data
    user_skill = UserSkillsT(
        id=document_id,
        user_id=user_id,
        name=name,
        description=encrypted_data["description"],
        skill_text=encrypted_data["skill_text"],
        parameters_schema=encrypted_data["parameters_schema"],
        is_encrypted=encrypt
    )
    
    session.add(user_skill)
    await session.flush()
    
    skill_id = user_skill.id
    log.debug(f"Created UserSkillsT record with ID '{skill_id}' (encrypted={encrypt})")
    
    # Step 4: Store description in Vector Store
    # Note: We store the ORIGINAL (unencrypted) description for search purposes
    try:
        vscf: VectorStoreConfig = await get_vector_store_settings(
            session=session,
            scope=scope
        )

        p_reg: UserProviderRegistry = await get_all_provider_configurations(
            session=session,
            user_id=user_id
        )

        prov: ProviderConfiguration = getattr(
            p_reg,
            vscf.e_provider
        )

        vs = await run_in_threadpool(
            get_vector_store,
            vs_collection_name=vscf.vs_collection_name,
            vs_vendor=vscf.vs_vendor,
            vs_port=vscf.vs_port,
            vs_base_url=vscf.vs_base_url,
            vs_encrypted_api_key=vscf.vs_encrypted_api_key,
            e_model=vscf.e_model,
            e_base_url=prov.base_url,
            e_encrypted_api_key=vscf.e_api_key,
            e_langchain_con=prov.langchain_con,
            e_dimensions=vscf.e_dimensions
        )

        metadata = {
            "document-name": name,
            "user-id": str(user_id),
            "user-document-id": str(document_id),
            "scope": scope
        }

        doc_ids = await run_in_threadpool(
            save_docs,
            vector_store=vs,
            scope=VectorStoreScope.USER_SKILLS,
            texts=[description],
            metadatas=[metadata],
            required_metadata=vscf.required_filters
        )

        log.debug(f"Stored skill description in Vector Store with document IDs: {doc_ids}")

    except Exception as e:
        log.error(f"Failed to store skill in Vector Store: {e}")
        raise

    return skill_id


@dataclass(frozen=True)
class SkillDescription:
    skill_id: uuid.UUID
    name: str
    description: str | None
    instructions: str
    parameter_schema: dict | None
    is_encrypted: bool


async def get_user_skill_by_skill_id(
    session: AsyncSession,
    user_id: uuid.UUID,
    skill_id: uuid.UUID
) -> SkillDescription:
    """Fetches a user skill record, and returns it as a SkillDescription object.
    
    Automatically decrypts fields if the skill is marked as encrypted.
    """
    skill = (await session.execute(
        select(UserSkillsT)
        .where(
            UserSkillsT.id == skill_id,
            UserSkillsT.user_id == user_id
        )
    )).scalar_one_or_none()
    if not skill:
        log.debug(f"Could not locate skill with id '{skill_id}' for user '{user_id}'")
        raise ValueError(f"Could not locate skill with id '{skill_id}'")
    
    # Decrypt the data
    decrypted_data = _decrypt_skill_data_from_row(skill)
    
    return SkillDescription(
        skill_id=decrypted_data["id"],
        name=decrypted_data["name"],
        description=decrypted_data["description"],
        instructions=decrypted_data["skill_text"],
        parameter_schema=decrypted_data["parameters_schema"],
        is_encrypted=decrypted_data["is_encrypted"]
    )


async def update_user_skill_by_skill_id(
    session: AsyncSession,
    user_id: uuid.UUID,
    skill_id: uuid.UUID,
    name: str | None = None,
    instructions: str | None = None,
    description: str | None = None,
    parameters_schema: dict | None = None,
    scope: str = "user_vs_skills"
) -> SkillDescription:
    """Updates a skill's fields. Data is always encrypted.
    
    If description is updated, also updates the Vector Store entry for semantic search.
    
    Args:
        session: Async database session
        user_id: The user's UUID
        skill_id: The skill's UUID
        name: New name (optional)
        instructions: New skill text/instructions (optional)
        description: New description (optional)
        parameters_schema: New parameters schema (optional)
        scope: The scope for Vector Store operations (default: "user_vs_skills")
    """
    skill = (await session.execute(
        select(UserSkillsT)
        .where(
            UserSkillsT.id == skill_id,
            UserSkillsT.user_id == user_id
        )
    )).scalar_one_or_none()
    if not skill:
        log.debug(f"Could not locate skill with id '{skill_id}' for user '{user_id}'")
        raise ValueError(f"Could not locate skill with id '{skill_id}'")
    
    # Track if description is being updated
    description_changed = description is not None
    
    # Always encrypt all fields - get current values if not provided
    if instructions is None:
        # Decrypt current skill_text first
        current_instructions = _decrypt_field(skill.skill_text, skill.is_encrypted)
        instructions = current_instructions if current_instructions else ""
    
    if description is None:
        current_description = _decrypt_field(skill.description, skill.is_encrypted)
        description = current_description if current_description else ""
        description_changed = False  # Using existing, not changing
    
    if name is None:
        # Keep existing name
        name = skill.name
    
    if parameters_schema is None:
        current_params = _decrypt_field(skill.parameters_schema, skill.is_encrypted)
        parameters_schema = current_params if current_params else {}
    
    # Encrypt all fields
    skill.name = name
    skill.description = _encrypt_field(description)
    skill.skill_text = _encrypt_field(instructions)
    skill.parameters_schema = _encrypt_field(parameters_schema)
    skill.is_encrypted = True
    
    await session.flush()
    
    # If description was changed, update Vector Store
    if description_changed and description is not None:
        try:
            vscf: VectorStoreConfig = await get_vector_store_settings(
                session=session,
                scope=scope
            )
            
            # Delete old entry from Vector Store
            await delete_document_from_qdrant_by_user_document_id(
                user_id=user_id,
                user_document_id=skill_id,
                collection_name=vscf.vs_collection_name,
                url=vscf.vs_base_url,
                port=vscf.vs_port,
                encrypted_api_key=vscf.vs_encrypted_api_key
            )
            
            # Get provider config
            p_reg: UserProviderRegistry = await get_all_provider_configurations(
                session=session,
                user_id=user_id
            )
            prov: ProviderConfiguration = getattr(p_reg, vscf.e_provider)
            
            # Save new description to Vector Store
            vs = await run_in_threadpool(
                get_vector_store,
                vs_collection_name=vscf.vs_collection_name,
                vs_vendor=vscf.vs_vendor,
                vs_port=vscf.vs_port,
                vs_base_url=vscf.vs_base_url,
                vs_encrypted_api_key=vscf.vs_encrypted_api_key,
                e_model=vscf.e_model,
                e_base_url=prov.base_url,
                e_encrypted_api_key=vscf.e_api_key,
                e_langchain_con=prov.langchain_con,
                e_dimensions=vscf.e_dimensions
            )
            
            metadata = {
                "document-name": skill.name,
                "user-id": str(user_id),
                "user-document-id": str(skill_id),
                "scope": scope
            }
            
            await run_in_threadpool(
                save_docs,
                vector_store=vs,
                scope=VectorStoreScope.USER_SKILLS,
                texts=[description],
                metadatas=[metadata],
                required_metadata=vscf.required_filters
            )
            
            log.debug(f"Updated skill description in Vector Store for skill '{skill_id}'")
            
        except Exception as e:
            log.error(f"Failed to update Vector Store for skill '{skill_id}': {e}")
            # Don't fail the update just because Vector Store failed
    
    # Return decrypted data
    decrypted_data = _decrypt_skill_data_from_row(skill)
    
    return SkillDescription(
        skill_id=decrypted_data["id"],
        name=decrypted_data["name"],
        description=decrypted_data["description"],
        instructions=decrypted_data["skill_text"],
        parameter_schema=decrypted_data["parameters_schema"],
        is_encrypted=decrypted_data["is_encrypted"]
    )
