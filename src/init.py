import asyncio

from langgraph.checkpoint.postgres import PostgresSaver

from core.config import config
from core.logging import get_logger
from database.session import async_engine
from init.create_tables import create_tables
from init.preconfigurations import (
    create_preconfigured_application_clients,
    create_preconfigured_mcps,
    create_preconfigured_model_sampling,
    create_preconfigured_models,
    create_preconfigured_providers,
    create_preconfigured_user_clients,
    create_preconfigured_vector_store,
    create_preconfigured_vector_store_collections,
)

# -------------------------------------------------------------------
# Init file to create all database tables and initial users
# -------------------------------------------------------------------

# This file should only be executed if you need to set up the required
# database tables, as well as when needing to create the preconfigured clients.

CREATE_TABLES = True
CREATE_CHECKPOINT_TABLES = True
CREATE_PRECONFIGURED_USER_CLIENTS = True
CREATE_PRECONFIGURED_APPLICATION_CLIENTS = True
CREATE_PRECONFIGURED_PROVIDERS = True
CREATE_PRECONFIGURED_VECTOR_STORES = True
CREATE_PRECONFIGURED_VECTOR_STORE_COLLECTIONS = True
CREATE_PRECONFIGURED_MODELS = True
CREATE_PRECONFIGURED_MCPS = True
CREATE_PRECONFIGURED_MODEL_SAMPLING = True

log = get_logger()


print("\n>>> STARTING INITIALiZATION PROCESS\n")


async def main() -> None:
    if CREATE_TABLES:
        print("\n---- STARTING TABLE CREATION ----\n")
        create_tables()

    if CREATE_PRECONFIGURED_PROVIDERS:
        print("\n---- STARTING CREATION OF / CHECKING PRECONFIGURED PROVIDERS ----\n")
        await create_preconfigured_providers()

    if CREATE_PRECONFIGURED_APPLICATION_CLIENTS:
        print(
            "\n---- STARTING CREATION OF / CHECKING PRECONFIGURED APPLICATION CLIENTS ----\n"
        )
        await create_preconfigured_application_clients()

    if CREATE_PRECONFIGURED_USER_CLIENTS:
        print(
            "\n---- STARTING CREATION OF / CHECKING PRECONFIGURED USER CLIENTS ----\n"
        )
        await create_preconfigured_user_clients()

    if CREATE_PRECONFIGURED_VECTOR_STORES:
        print(
            "\n---- STARTING CREATION OF / CHECKING PRECONFIGURED VECTOR STORES ----\n"
        )
        await create_preconfigured_vector_store()

    if CREATE_PRECONFIGURED_VECTOR_STORE_COLLECTIONS:
        print(
            "\n---- STARTING CREATION OF / CHECKING PRECONFIGURED VECTOR STORE COLLECTIONS ----\n"
        )
        await create_preconfigured_vector_store_collections()

    if CREATE_CHECKPOINT_TABLES:
        print("\n---- STARTING CREATION OF CHECKPOINT TABLES ----\n")
        with PostgresSaver.from_conn_string(
            conn_string=config.PG_CHECKPOINTER_URL
        ) as checkpointer:
            checkpointer.setup()
            log.info("DONE: Checkpoint tables created.")

    if CREATE_PRECONFIGURED_MODELS:
        print("\n---- STARTING ADDITION OF MODELS TO MODELST ----\n")
        await create_preconfigured_models()

    if CREATE_PRECONFIGURED_MCPS:
        print("\n---- ADDING MCPS TO MCPST ----\n")
        await create_preconfigured_mcps()

    if CREATE_PRECONFIGURED_MODEL_SAMPLING:
        print("\n---- ADDING MODEL SAMPLING PARAMETERS TO MODELSAMPLINGT ----\n")
        await create_preconfigured_model_sampling()

    await async_engine.dispose()


asyncio.run(main())

print("\n>>> DONE: FINISHED INITIALIZATION!\n")
