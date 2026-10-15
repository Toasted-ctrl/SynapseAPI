# SynapseAPI

![Python](https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-5B2EFF?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-D6405B?logo=qdrant&logoColor=white)
![Firecrawl](https://img.shields.io/badge/Firecrawl-FF6B35?style=flat&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-v1.36-blue?logo=kubernetes&logoColor=white)
![Keel](https://img.shields.io/badge/Keel-1b1f23?logo=keel.sh&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)


> One self-hostable API for many LLM providers, with agents, memory, RAG and tools built in.

> [!NOTE]
> This document was last reviewed on 2026-10-15.

SynapseAPI is a lightweight gateway that sits between your apps and the LLM providers you use. Your apps talk to a single, unified API, and SynapseAPI handles the rest: authentication, provider keys, agents, tool use, and retrieval over your own documents. It is designed to run comfortably on low-end, self-hosted hardware.

*Why "Synapse"? A synapse is where signals pass from one neuron to the next. SynapseAPI is the point where your apps connect to many models.*

## Features

**Authentication**
- Google OAuth2 login with server-side sessions, for frontend apps serving multiple users.
- API key verification for clients that call the API directly, with no login required.

**Inference**
- Chat completion (regular and streaming) through a single API, regardless of provider.
- Users can bring their own API keys for third-party providers.
- Local Ollama instances can be added as a provider.

**Agents**
- Create, update and delete reusable agents with their own model, sampling parameters, system prompt and tools.
- Talk to a stored agent by passing its `agent_id`, or configure a model and tools inline.
- Conversations are kept in threads, so they can be continued later.

**Knowledge & Memory**
- Upload your own files, skills and memories.
- Vector search over your documents, backed by Qdrant.

**Tools**
- MCP server support.
- Web search through Firecrawl.

## Supported LLM Providers
Every provider listed here has been tested and verified with a personally obtained API key.

- Anthropic
- Melious
- Mistral
- Ollama (self-hosted)
- OpenAI
- Z.ai

## Tech Stack

| Component  | Role                                      |
|------------|-------------------------------------------|
| FastAPI    | HTTP API                                  |
| LangChain  | Provider integrations and agent runtime   |
| Langfuse   | Tracing and observability                 |
| PostgreSQL | Application data and conversation threads |
| Redis      | Session storage and caching               |
| Qdrant     | Vector store for files, skills, memories  |
| Firecrawl  | Web search                                |
| GCP OAuth2 | Google login                              |
| Docker     | Container image                           |
| Kubernetes | Deployment (with Keel for auto-updates)   |

## Quickstart
This gets a local development server running. You will need Python 3.14, [uv](https://docs.astral.sh/uv/), PostgreSQL, Redis 6+ and Qdrant. See [SETUP.md](SETUP.md) for full configuration and Kubernetes deployment.

```bash
git clone https://github.com/Toasted-ctrl/SynapseAPI.git
cd SynapseAPI
cp .env.example .env    # fill in your database, Redis, and secret values
uv sync
uv run src/init.py      # create tables and preconfigured data
uv run src/main.py      # serves on http://localhost:8000
```

Check that it is running:

```bash
curl http://localhost:8000/api/v1/status
```

## Usage Example
Send a prompt to a model. Authentication uses either a session cookie (after Google login) or an `X-API-Key` header:

```bash
curl -X POST http://localhost:8000/api/v1/conversation \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
        "prompt": "Explain what a synapse is in one sentence.",
        "model": { "provider": "anthropic", "name": "<model-name>" }
      }'
```

The response includes a `thread_id`. Send it back with your next request to continue the conversation. To talk to a stored agent, replace `model` with `"agent_id": "<agent-id>"`.

## API Overview
All endpoints are prefixed with `/api/v1`. Full interactive documentation is available at `/docs` (Swagger UI) or `/redoc` (ReDoc) while the server is running.

| Area         | Endpoints                                                                 | Purpose                                         |
|--------------|---------------------------------------------------------------------------|-------------------------------------------------|
| Conversation | `POST /conversation`, `POST /conversation/stream`                         | Chat with a model or a stored agent             |
| Agents       | `GET /agents`                                                             | List your agents                                |
| Playground   | `POST, GET, PUT, DELETE /playground/agent`                                | Create and manage a single agent                |
| Auth         | `GET /auth/google/login`, `GET /auth/google/callback`, `GET /auth/me`     | Google login and current user                   |
| Documents    | `GET, DELETE /documents`                                                  | List and remove uploaded documents              |
| Skills       | `POST, GET, PATCH /skill`                                                 | Manage skills                                   |
| Vectors      | `POST /vector_embedding`, `POST /vector_store/add`, `POST /vector_store/search` | Embed, store and search documents         |
| Providers    | `GET /providers/models`, `GET /providers/models/sampling`, `GET /providers/configuration` | Available models and provider setup |
| Tools        | `GET /tools/mcp`                                                          | Available MCP servers                           |
| Translation  | `POST, GET /translation/translategemma`                                   | Translation via TranslateGemma                  |
| Settings     | `POST /settings/user/keys`                                                | Store your own provider API keys                |
| Health       | `GET /root`, `GET /status`                                                | Service info and health check                   |

## Roadmap

### Short Term
- [ ] HMAC hashing and verification for each path.
- [ ] Locate which server currently supports TranslateGemma: check internal servers first, then external ones, and prefer internal since it is free of charge.

### Long Term
- [x] Agent building (create, read, update, delete).
- [ ] Support for more login providers.
- [ ] Log conversation IDs and history in a conversation table.
- [ ] Share agents with other users.

## Contributing
PRs are welcome! Before opening one, please make sure the tests pass:

```bash
uv sync --group dev
uv run pytest
```

For security issues, please see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## License
MIT. See [LICENSE](LICENSE).
