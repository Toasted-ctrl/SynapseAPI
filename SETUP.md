# Setup

> [!NOTE]
> This document was last reviewed on 2026-08-23.

This guide covers setting up, building, and deploying SynapseAPI.

The project supports two ways of running the application:

- Local development using uv
- Kubernetes deployment using the manifests provided in k8s/

Docker Compose is no longer used or supported as a deployment method.
## 1. Prerequisites

Depending on how you intend to run the application, you will need:
### Local development

- Python
- uv
- PostgreSQL
- Redis 6+
- Qdrant 

### Kubernetes deployment

- A running Kubernetes cluster
- kubectl configured to access the cluster
- Docker or another container image builder
- A container registry accessible by the Kubernetes cluster
- An NGINX Ingress Controller if you want to use the included Ingress manifest

PostgreSQL and Redis must be available to the application. They may be hosted inside or outside the Kubernetes cluster.

## 2. Clone the Repository
```bash
git clone https://github.com/Toasted-ctrl/SynapseAPI.git
cd SynapseAPI
```
## 3. Run the Required Services

For local development, PostgreSQL, Redis 6+ and Qdrant must be running before you
start the API. The quickest way is three Docker containers:

```bash
# PostgreSQL
docker run -d --name synapse-postgres \
  -e POSTGRES_USER=synapse \
  -e POSTGRES_PASSWORD=synapse \
  -e POSTGRES_DB=synapse \
  -p 5432:5432 \
  postgres:17

# Redis 6+
docker run -d --name synapse-redis \
  -p 6379:6379 \
  redis:7

# Qdrant
docker run -d --name synapse-qdrant \
  -p 6333:6333 \
  qdrant/qdrant
```

With these defaults, the matching `.env` values are:

```dotenv
REDIS_USER=
REDIS_HOSTNAME=localhost
REDIS_PASSWORD=
REDIS_PREFIX=synapse_1
REDIS_PORT=6379

PG_HOSTNAME=localhost
PG_USERNAME=synapse
PG_PASSWORD=synapse
PG_DATABASE=synapse
PG_DIALECT=postgresql
PG_DRIVER=psycopg2
PG_PORT=5432
```

Qdrant is not configured through `.env`. Instead, it is registered as a vector
store in `src/init/configure_init_vs.json` (see
`configure_init_vs_example.json`). For the container above, use
`"base_url": "localhost"`, `"port": 6333` and any string as `api_key` (the
local container does not require one).

If you already have these services running elsewhere, skip the containers and
point `.env` (and `configure_init_vs.json`) at your existing instances.

## 4. Initialize the Database
The project includes an initialization script for creating the required database tables and optionally creating preconfigured clients, providers, vector stores and vector store collections.

Configuration is controlled in src/init.py:

```python
CREATE_PRECONFIGURED_CLIENTS = True
CREATE_PRECONFIGURED_PROVIDERS = True
CREATE_TABLES = True
```

Once your configuration is complete, initialize the database:

```bash
uv run src/init.py
```


All commands in this guide assume that you are running them from the repository root.
## 5. Local Development
```bash
cp .env.example .env
nano .env
```
### 5.1 Redis
Redis 6+ is required by the application.
```dotenv
REDIS_USER=
REDIS_HOSTNAME=
REDIS_PASSWORD=
REDIS_PREFIX=
REDIS_PORT=
```
### 5.2 PostgreSQL
PostgreSQL is required for application data.
```dotenv
PG_HOSTNAME=
PG_USERNAME=
PG_PASSWORD=
PG_DATABASE=
PG_DIALECT=
PG_DRIVER=
PG_PORT=
```
### 5.3 Encryption

Sensitive data is encrypted at rest. Configure a strong encryption key:

```dotenv
ENCRYPTION_KEY=
```
### 5.4 Blind Index

The blind index key is used for searchable encrypted data.

Use a separate key from ENCRYPTION_KEY:

```dotenv
BLIND_INDEX_KEY=
```
### 5.5 JWT

JWT tokens are signed using:

```dotenv
JWT_SECRET=
```
### 5.6 Logging

The available log levels are:

```dotenv
debug
info
warn
error
```

### 5.7 Google Login

Google OAuth is optional.

To enable it:

```dotenv
ENABLE_GOOGLE_LOGIN=true
```

Then configure the remaining Google OAuth variables:

```dotenv
GOOGLE_CLIENT_ID=
GOOGLE_REDIRECT_URI=
GOOGLE_AUTH_URL=
GOOGLE_HMAC=
GOOGLE_TOKEN_URL=
GOOGLE_CLIENT_SECRET=
```
### 5.8 Cookies

Cookie configuration is also available:

```dotenv
COOKIE_SECURE=false
COOKIE_MAX_AGE=86400
```

For production deployments using HTTPS, COOKIE_SECURE should normally be set to true.

## 6. Building the image

The repository includes build.sh for building the application container image.

To see the available options:

```bash
./build.sh --help
```
The script creates both a versioned image and a latest tag.

For example:

```
storage01:5000/artificial-intelligence-api:0.1.1
storage01:5000/artificial-intelligence-api:latest
```

If you are using a different container registry, update the registry configuration in build.sh before building and pushing the image.
## 7. Kubernetes Deployment

The repository includes the Kubernetes manifests required to deploy the application under the k8s/ directory. The current manifests include a namespace, ConfigMap, Deployment example, Service, PersistentVolume, PersistentVolumeClaim, Secret example, and Ingress.

The Kubernetes deployment is intended to run the application as a container and expose it through a Kubernetes Service.
### 7.1 Prepare the Kubernetes Manifests

Before deploying, review the files in:

`k8s/`

The directory contains:

```
configmap.yaml
deployment_example.yaml
ingress.yaml
namespace.yaml
pv.yaml
pvc.yaml
secrets_example.yaml
service.yaml
```
Update all manifests with your own details.
### 7.2 Deploy the Application

From the repository root, apply the manifests:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/pv.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
Verify the deployment:

```bash
kubectl get all -n artificial-intelligence-api
```
Check the Pods:

```bash
kubectl get pods -n artificial-intelligence-api
```