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
## 3. Initialize the Database
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
## 4. Local Development
```bash
cp .env.example .env
nano .env
```
### 4.1 Redis
Redis 6+ is required by the application.
```dotenv
REDIS_USER=
REDIS_HOSTNAME=
REDIS_PASSWORD=
REDIS_PREFIX=
REDIS_PORT=
```
### 4.2 PostgreSQL
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
### 4.3 Encryption

Sensitive data is encrypted at rest. Configure a strong encryption key:

```dotenv
ENCRYPTION_KEY=
```
### 4.4 Blind Index

The blind index key is used for searchable encrypted data.

Use a separate key from ENCRYPTION_KEY:

```dotenv
BLIND_INDEX_KEY=
```
### 4.5 JWT

JWT tokens are signed using:

```dotenv
JWT_SECRET=
```
### 4.6 Logging

The available log levels are:

```dotenv
debug
info
warn
error
```

### 4.7 Google Login

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
### 4.8 Cookies

Cookie configuration is also available:

```dotenv
COOKIE_SECURE=false
COOKIE_MAX_AGE=86400
```

For production deployments using HTTPS, COOKIE_SECURE should normally be set to true.

## 5. Building the image

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
## 6. Kubernetes Deployment

The repository includes the Kubernetes manifests required to deploy the application under the k8s/ directory. The current manifests include a namespace, ConfigMap, Deployment example, Service, PersistentVolume, PersistentVolumeClaim, Secret example, and Ingress.

The Kubernetes deployment is intended to run the application as a container and expose it through a Kubernetes Service.
### 6.1 Prepare the Kubernetes Manifests

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
### 6.2 Deploy the Application

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