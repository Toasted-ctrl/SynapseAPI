# Contributing to SynapseAPI

> [!NOTE]
> This document was last reviewed on 2026-11-03.

PRs are welcome! This guide covers what to check before opening one and the conventions the codebase follows.

For security issues, please see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Before opening a PR
Make sure the tests pass, the code is linted and formatted with [ruff](https://docs.astral.sh/ruff/), and it type-checks with [mypy](https://mypy.readthedocs.io/):

```bash
uv sync --group dev
uv run pytest
uv run ruff check --fix .   # lint and auto-fix what it can
uv run ruff format .        # format code
uv run mypy src             # type-check
```

Ruff and mypy are both configured in `pyproject.toml`. Please fix any remaining `ruff check` errors by hand before opening the PR.

`mypy src` currently passes with no errors, so please keep it that way. mypy runs with the Pydantic plugin, so it checks the fields and types of the API models too. Add type hints to new functions, including return types. Use `# type: ignore[<code>]` only as a last resort, and always with the specific error code.

## API model naming
Request and response bodies are Pydantic models in each router's `models.py` (e.g. `src/api/v1/playground/models.py`). Name them after the resource, with a suffix that says which way the data flows:

| Suffix   | Used for                         | Example                    |
|----------|----------------------------------|----------------------------|
| `Base`   | Fields shared by Create and Read | `AgentConfigurationBase`   |
| `Create` | Request body sent by the client  | `AgentConfigurationCreate` |
| `Read`   | Response body returned by the API | `AgentConfigurationRead`  |

Rules of thumb:

- **Server-assigned fields (ids, timestamps, flags) only go on `Read`.** A `Create` model must not accept an `id`, and on a `Read` model the `id` is required, not optional.
- **When a resource has both a request and a response model, put the shared fields in a `Base`** and have `Create` and `Read` inherit from it. Endpoints that only return data (e.g. `StatusRead`, `ProvidersRead`) need just the `Read` model.
- **`Create` is also used for update bodies** (`PUT`), unless an update needs different fields, such as all-optional ones. In that case, add an `Update` model.
- **Use suffixes, not prefixes.** Write `AgentConfigurationRead`, not `ResponseAgentConfiguration` or `AgentConfigurationResponse`. Avoid the older `Payload`/`Response` naming.

```python
class AgentConfigurationBase(BaseModel):
    name: str
    ...


class AgentConfigurationCreate(AgentConfigurationBase):
    """Request body for creating or updating an Agent. The server assigns the id."""


class AgentConfigurationRead(AgentConfigurationBase):
    """An Agent as returned by the API."""

    id: Annotated[UUID, Field(description="ID of the Agent.")]
```

Set the `Read` model as the route's `response_model` so the OpenAPI docs at `/docs` show the right schema.
