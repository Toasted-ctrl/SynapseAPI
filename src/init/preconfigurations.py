import json
import os
import uuid

from sqlalchemy import select

from core.logging import get_logger
from database.providers import get_or_create_provider
from database.schemas.clients import ClientsT
from database.schemas.mcp import MCPsT
from database.schemas.model_sampling import ModelSamplingT
from database.schemas.models import ModelsT
from database.schemas.persons_users import PersonsT, UsersT
from database.schemas.providers import ProvidersT
from database.schemas.user_keys import UserKeysT
from database.schemas.vector_store import VectorStoreSettingsT
from database.schemas.vector_store_collections import VectorStoreCollectionT
from database.session import get_db_session_ctx
from security.encryption import encrypt
from security.hash import get_hash_sha256

log = get_logger()


async def create_preconfigured_providers() -> None:
    """Loads the configure_init_providers.json file and adds all preconfigured providers to the database.
    Will force a shutdown if the file appears misconfigured."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_init_providers.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_PROVIDERS is enabled, but 'configure_init_providers.json' is missing. Shutting down..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not data.get("configured", False):
        log.error(
            "ENV: CREATE_PRECONFIGURED_PROVIDERS is enabled, but 'configure_init_providers.json' appears misconfigured. Shutting down..."
        )
        raise SystemExit(1)

    providers = data.get("providers")
    if len(providers) == 0:
        log.info("No Providers to configure, skipping...")
        return

    for provider in providers:
        async with get_db_session_ctx() as session:
            await get_or_create_provider(
                session=session,
                name=provider.get("name"),
                langchain_con=provider.get("langchain_con"),
                base_url=provider.get("base_url"),
                internal=provider.get("internal"),
                requires_api_key=provider.get("requires_api_key"),
                host=provider.get("host"),
                mac_address=provider.get("mac_address"),
            )

    return


async def create_preconfigured_vector_store() -> None:
    """Loads the configure_init_vs.json file and adds all preconfigured vector stores to the database.
    Will force a shutdown if the file is missing."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_init_vs.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_VS is enabled, but 'configure_init_vs.json' is missing. Shutting down ..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    num_vs = len(data)
    if num_vs == 0:
        log.error(
            "ENV: CREATE_PRECONFIGURED_VS is enabled, but 'configure_init_vs.json' is empty. Shutting down ..."
        )
        raise SystemExit(1)

    for vs in data:
        async with get_db_session_ctx() as session:
            exists = (
                await session.scalars(
                    select(VectorStoreSettingsT).where(
                        VectorStoreSettingsT.id == uuid.UUID(vs.get("id"))
                    )
                )
            ).all()

            if len(exists) > 0:
                log.info(
                    f"Vector Store '{vs.get('vendor')}' with URL '{vs.get('base_url')}' already exists, skipping ..."
                )
                continue

            log.info(
                f"Adding new Vector Store, vendor '{vs.get('vendor')}', URL '{vs.get('base_url')}' ..."
            )
            async with get_db_session_ctx() as session:
                nvs = VectorStoreSettingsT(
                    id=uuid.UUID(vs.get("id")),
                    encrypted_api_key=encrypt(vs.get("api_key")),
                    vendor=vs.get("vendor"),
                    base_url=vs.get("base_url"),
                    port=vs.get("port"),
                )

                session.add(nvs)
                log.info("Vectore Store added.")

    log.info("DONE: All Vector Stores configured.")


async def create_preconfigured_vector_store_collections() -> None:
    """Loads the configure_init_vs_colections.json file and adds all preconfigured collections to the database.
    Will force a shutdown if the file is missing."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_init_vs_collections.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_VS_COLLECTIONS is enabled, but 'configure_init_vs_collections.json' is missing. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    num_col = len(data)
    if num_col == 0:
        log.error(
            "ENV: CREATE_PRECONFIGURED_VS is enabled, but 'configure_init_vs_collections.json' is empty. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    async with get_db_session_ctx() as session:
        for col in data:
            exists = (
                await session.scalars(
                    select(VectorStoreCollectionT).where(
                        VectorStoreCollectionT.id == uuid.UUID(col.get("id"))
                    )
                )
            ).all()

            if len(exists) > 0:
                log.info(
                    f"Vector Store collection '{col.get('name')}' already exists, skipping ..."
                )
                continue

            log.info(f"Adding new Vector Store with name '{col.get('name')}' ...")
            ncol = VectorStoreCollectionT(
                id=uuid.UUID(col.get("id")),
                name=col.get("name"),
                vector_store_id=uuid.UUID(col.get("vector_store_id")),
                e_provider=col.get("embedding_provider"),
                e_model=col.get("embedding_model"),
                e_dimensions=col.get("embedding_dimensions"),
                e_api_key=encrypt(col.get("api_key")),
                scope=col.get("scope"),
                required_filters=col.get("required_filters"),
            )

            session.add(ncol)
            log.info(f"Vector Store collection '{col.get('name')}' added.")

    log.info("DONE: All Vector collections configured.")


async def create_preconfigured_application_clients() -> None:
    """Loads the configure_init_clients.json file and adds all preconfigured application clients to the database.
    Will force a shutdown if the file is missing."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_init_clients.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_APPLICATION_CLIENTS is enabled, but 'configure_init_clients.json' is missing. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    clients = data.get("applications")

    num_clients = len(clients)
    if num_clients == 0:
        log.error(
            "ENV: CREATE_PRECONFIGURED_CLIENTS is enabled, but 'configure_init_clients.json' > 'applications' is empty. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    for client in clients:
        async with get_db_session_ctx() as session:
            exists = (
                await session.scalars(
                    select(ClientsT).where(ClientsT.id == client.get("id"))
                )
            ).all()

            if len(exists) > 0:
                log.info(
                    f"Client '{client.get('client_name')}' already exists, skipping ..."
                )
                continue

            nc = ClientsT(
                id=uuid.UUID(client.get("id")),
                api_key_hash=get_hash_sha256(client.get("api_key")),
                key_type="Application",
                require_jwt=client.get("require_jwt"),
                is_active=True,
                require_external_id=True,
                encrypted_owner_email=encrypt(client.get("owner_email")),
                encrypted_hmac_secret=encrypt(client.get("hmac")),
                encrypted_client_name=encrypt(client.get("client_name")),
                encrypted_redirect_uri=encrypt(client.get("redirect_uri")),
                blind_index_client_name=get_hash_sha256(client.get("client_name")),
                blind_index_owner_email=get_hash_sha256(client.get("owner_email")),
            )

            session.add(nc)
            log.info(f"Added Client '{client.get('client_name')}' to the database ...")

    log.info("DONE: All application Clients configured.")


async def create_preconfigured_user_clients() -> None:
    """Loads the configure_init_clients.json file and adds all preconfigured user clients to the database.
    Will force a shutdown if the file is missing."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_init_clients.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_VS_COLLECTIONS is enabled, but 'configure_init_clients.json' is missing. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    clients = data.get("users")

    num_clients = len(clients)
    if num_clients == 0:
        log.error(
            "ENV: CREATE_PRECONFIGURED_USER_CLIENTS is enabled, but 'configure_init_clients.json' > 'users' is empty. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    for client in clients:
        async with get_db_session_ctx() as session:
            c_exists = (
                await session.scalars(
                    select(ClientsT).where(
                        ClientsT.id == uuid.UUID(client.get("client_id"))
                    )
                )
            ).all()

            if len(c_exists) > 0:
                log.info(
                    f"Client '{client.get('client_name')}' already exists, skipping client creation."
                )

            else:
                nc = ClientsT(
                    id=uuid.UUID(client.get("client_id")),
                    api_key_hash=get_hash_sha256(client.get("api_key")),
                    key_type="User",
                    require_jwt=client.get("require_jwt"),
                    is_active=True,
                    require_external_id=True,
                    encrypted_owner_email=encrypt(client.get("owner_email")),
                    encrypted_hmac_secret=encrypt(client.get("hmac")),
                    encrypted_client_name=encrypt(client.get("client_name")),
                    encrypted_redirect_uri=None,
                    blind_index_client_name=get_hash_sha256(client.get("client_name")),
                    blind_index_owner_email=get_hash_sha256(client.get("owner_email")),
                )

                session.add(nc)
                log.info(
                    f"Added Client '{client.get('client_name')}' to the database ..."
                )

            p_exists = (
                await session.scalars(
                    select(PersonsT).where(
                        PersonsT.id == uuid.UUID(client.get("person_id"))
                    )
                )
            ).all()

            if len(p_exists) > 0:
                log.info(
                    f"User with Client ID '{client.get('client_id')}' already exists, skipping person and user generation ..."
                )

            else:
                np = PersonsT(
                    id=uuid.UUID(client.get("person_id")),
                    encrypted_email=encrypt(client.get("owner_email")),
                    encrypted_first_name=encrypt(client.get("first_name")),
                    encrypted_last_name=encrypt(client.get("last_name")),
                    blind_index_email=get_hash_sha256(client.get("owner_email")),
                )

                session.add(np)

                nu = UsersT(
                    person_id=uuid.UUID(client.get("person_id")),
                    api_key_id=uuid.UUID(client.get("client_id")),
                    external_id=None,
                    login_provider=None,
                )

                session.add(nu)
                log.info(f"Created new user for Client '{client.get('client_id')}' ...")

            await session.flush()

            user_id = await session.scalar(
                select(UsersT.id).where(
                    UsersT.api_key_id == uuid.UUID(client.get("client_id"))
                )
            )

            log.info(
                f"User ID is '{user_id}' for Client '{client.get('client_id')}' ..."
            )

            api_keys = client.get("keys")
            if api_keys == []:
                log.info(
                    f"No user keys configured for user with Client ID '{client.get('client_id')}', skipping key creation ..."
                )

            else:
                for key, value in api_keys.items():
                    provider_id = await session.scalar(
                        select(ProvidersT.id).where(ProvidersT.name == key)
                    )

                    k_exists = (
                        await session.scalars(
                            select(UserKeysT).where(
                                UserKeysT.provider_id == provider_id,
                                UserKeysT.user_id == user_id,
                            )
                        )
                    ).all()

                    if len(k_exists) > 0:
                        log.info(
                            f"Key '{key}' already exists for user '{user_id}', skipping key creation ..."
                        )
                        continue

                    else:
                        nk = UserKeysT(
                            user_id=user_id,
                            provider_id=provider_id,
                            encrypted_api_key=encrypt(value),
                            api_key_short=value[:10],
                        )

                    session.add(nk)
                    log.info(f"'{key}' added for user '{user_id}' ...")

    log.info("DONE: All user Clients configured.")


async def create_preconfigured_models() -> None:
    """Adds all preconfigured models to the ModelsT table based on their respective expertise."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_model_types.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_MODELS is enabled, but 'configure_model_types.json' is missing. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    async with get_db_session_ctx() as session:
        chat_completion_models = data.get("chat_completion")
        for model in chat_completion_models:
            exists = (
                await session.scalars(select(ModelsT).where(ModelsT.name == model))
            ).all()
            if not exists:
                log.info(
                    f"Model '{model}' does not exist in ModelsT with expertise 'chat_completion', adding ..."
                )
                nm = ModelsT(name=model, expertise="chat_completion")
                session.add(nm)

        translation_models = data.get("translation")
        for model in translation_models:
            exists = (
                await session.scalars(select(ModelsT).where(ModelsT.name == model))
            ).all()
            if not exists:
                log.info(
                    f"Model '{model}' does not exist in ModelsT with expertise 'translation', adding ..."
                )
                nm = ModelsT(name=model, expertise="translation")
                session.add(nm)

        vector_embedding_models = data.get("vector_embedding")
        for model in vector_embedding_models:
            exists = (
                await session.scalars(select(ModelsT).where(ModelsT.name == model))
            ).all()
            if not exists:
                log.info(
                    f"Model '{model}' does not exist in ModelsT with expertise 'vector_embedding', adding ..."
                )
                nm = ModelsT(name=model, expertise="vector_embedding")
                session.add(nm)

        system_one_models = data.get("system_one")
        for model in system_one_models:
            exists = (
                await session.scalars(select(ModelsT).where(ModelsT.name == model))
            ).all()
            if not exists:
                log.info(
                    f"Model '{model}' does not exist in ModelsT with expertise 'system_one', adding ..."
                )
                nm = ModelsT(name=model, expertise="system_one")
                session.add(nm)


async def create_preconfigured_mcps() -> None:
    """Adds all preconfigured mcps to the MCPsT table."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_init_mcps.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_MCPS is enabled, but 'configure_init_mcps.json' is missing. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    async with get_db_session_ctx() as session:
        for mcp in data:
            exists = (
                await session.scalars(
                    select(MCPsT).where(
                        MCPsT.name == mcp.get("name"),
                        MCPsT.transport == mcp.get("transport"),
                        MCPsT.url == mcp.get("url"),
                    )
                )
            ).all()

            if exists:
                continue

            log.info(f"Adding MCP {mcp.get('name')} to MCPsT.")
            nmcp = MCPsT(
                name=mcp.get("name"), transport=mcp.get("transport"), url=mcp.get("url")
            )

            session.add(nmcp)


async def create_preconfigured_model_sampling() -> None:
    """Adds all preconfigured model sampling parameters to the ModelSamplingT table.
    Only new (model, provider, parameter) combinations are added, existing entries are left untouched."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "configure_model_sampling.json")
    if not os.path.exists(file_path):
        log.error(
            "ENV: CREATE_PRECONFIGURED_MODEL_SAMPLING is enabled, but 'configure_model_sampling.json' is missing. "
            "Shutting down ..."
        )
        raise SystemExit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    async with get_db_session_ctx() as session:
        for provider, models in data.items():
            for model in models:
                for parameter, supported in model.get("capabilities", {}).items():
                    exists = (
                        await session.scalars(
                            select(ModelSamplingT).where(
                                ModelSamplingT.name == model.get("name"),
                                ModelSamplingT.provider == provider,
                                ModelSamplingT.parameter == parameter,
                            )
                        )
                    ).all()

                    if exists:
                        continue

                    log.info(
                        f"Adding sampling parameter '{parameter}' for model '{model.get('name')}' ({provider}) to ModelSamplingT."
                    )
                    nms = ModelSamplingT(
                        name=model.get("name"),
                        provider=provider,
                        parameter=parameter,
                        supported=supported,
                    )

                    session.add(nms)

    log.info("DONE: All model sampling parameters configured.")
