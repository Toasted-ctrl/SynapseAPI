# Security

> [!NOTE]
> This document was last reviewed on 2026-11-06.

## Authentication
This API supports authentication via:
- **API keys**
- **Google OAuth**

## Hashing
Client API keys for connection to the API are stored as a hash, as well as Session IDs.

## Encryption
The following fields are encrypted at rest (the encryption key is only known to the API):

### Clients (for Applications or Users)
- Client Owner email
- Client Name
- Client redirect uri (for login), if enabled
- Client HMAC secret

### Users / Persons
- First Name
- Last name
- Email
- User provided API keys
- Agent configurations
- Skills

### Messages
- Checkpointer (blob storage)

### Planned
- Stored messages

## Data Sharing
All information is processed locally and is **not** shared with third parties, with one exception: message completion requests and model queries are sent to the relevant external AI provider (e.g., Anthropic) using a User's own provided API keys. Requests to external AI Providers using a personal API key may be logged by the Provider.
The User's message history is encrypted at rest.

## MCP
MCPs are currently available to ALL users, regardless of access type or platform. In a later version I'll introduce MCP management so admins can decide whom should have access to which MCP.

## Session Management
Sessions are handled server side. Authentication through frontend applications will store a cookie with session_id after logging in.

## Network
It is recommended to run this behind a VPN or on a local network (e.g., Tailscale).