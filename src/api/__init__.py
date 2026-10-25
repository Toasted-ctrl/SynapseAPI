import importlib
import pkgutil

from fastapi import APIRouter, FastAPI

from core.logging import get_logger

log = get_logger()


def include_routers(app: FastAPI) -> None:
    """Includes every module-level `router` (APIRouter) found under this package.

    The URL prefix is taken from the version package, e.g. `api.v1.auth.me` -> `/api/v1`.
    A module can opt out by setting `ENABLED = False`."""

    for module_info in sorted(
        pkgutil.walk_packages(__path__, prefix=f"{__name__}."), key=lambda m: m.name
    ):
        if module_info.ispkg:
            continue

        module = importlib.import_module(module_info.name)
        router = getattr(module, "router", None)
        if not isinstance(router, APIRouter):
            continue

        if not getattr(module, "ENABLED", True):
            log.info(f"Router '{module_info.name}' is disabled, skipping.")
            continue

        prefix = "/" + "/".join(module_info.name.split(".")[:2])
        app.include_router(router=router, prefix=prefix)
        log.debug(f"Included router '{module_info.name}' at '{prefix}'.")
