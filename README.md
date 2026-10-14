# Artificial Intelligence API (AIA)

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

> A unified API gateway for multiple LLM providers.

> [!NOTE]
> This document was last reviewed on 2026-10-14.

This project intends to build an "Artificial Intelligence API" (AIA), which will serve as an API gateway to multiple LLM providers. The goal is to make an easy to integrate unified API, which could easily be self-hosted on low-end hardware.

## Features
- Login support (Google OAuth2) for Applications intended to serve multiple users.
- API Key verification for users that should be allowed to poll the API directly, no Login required.
- Chat completion through LLM providers.
- Local Ollama configurations may be added.
- Users may add their own API keys for interaction with third party LLM Providers.
- Integration with Qdrant.
- Server side session management for frontend apps.
- MCP support.
- User document upload (Files, Skills, Memories).
- Web Search through FireCrawl.

## Tech Stack
- FastAPI
- LangChain
- LangFuse
- Docker
- Redis
- PostgreSQL
- GCP (OAuth2)
- Kubernetes
- Qdrant

## Supported LLM Providers
- Current supported Providers are tested and verified with personally obtained API keys.

### Current
- Ollama (if self-hosted)
- Anthropic
- Melious
- OpenAI
- Z.ai
- Mistral

## Setup
- Please check the 'SETUP.md' file to get started.

## API Endpoints
> All Endpoints are prefixed with '/api/v1'.

### Agent
- GET /agents

### Conversation
- POST /conversation
- POST /conversation/stream

### Auth
- GET /auth/google/login
- GET /auth/google/callback
- GET /auth/me

### Documents
- GET, DELETE /documents

### Skills
- POST, GET, PATCH /skill

### Vector Store / Embedding
- POST /vector_embedding
- POST /vector_store/add
- POST /vector_store/search

### Providers
- GET /providers/models
- GET /providers/models/sampling
- GET /providers/configuration

### Playground
- POST, GET, DELETE /agent

### Root & Status
- GET /root
- GET /status

### Tools
- GET /tools/mcp

### Translations
- POST, GET /translation/translategemma

### Settings
- POST /settings/user/keys

## Roadmap

### Short Term
- Add HMAC hashing and verification for each path.
- Build a function to locate which server is currently supporting translategemma: Check internal
servers first, follow by external ones. Return internal as this one is free of charge.

### Long Term
- Add support for more Login providers.
- Add agent building functionalities.
- Log conversation ids and history in a conversation table.
- Enable possibility to share agents with other users.

## Contributing
PRs welcome!

## Documentation
Full interactive API documentation is available at `/docs` (Swagger UI)
or `/redoc` (ReDoc) when the server is running.