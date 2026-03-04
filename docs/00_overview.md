# Overview

Flower handles **remote orchestration** of federated learning, but it assumes
that every remote machine already has a compatible Python and Flower environment.

Docker is used to:
- Standardize OS and dependencies
- Isolate runtime environments
- Control ports, volumes, permissions, certificates
- Enable long-term scalability (e.g. Kubernetes)

---

## High-level architecture

- **Server / Superlink**
  - Coordinates rounds and client participation
  - Does not access raw client data

- **Client node**
  - Runs local training and evaluation
  - Keeps data local
  - Can disconnect and later reconnect (FL should continue)

- **Docker Compose**
  - Standardizes deployment and configuration per machine
  - Encodes ports, mounts, env variables, and certificates



## Deployment Starte Pack

A minimal deployment usually contains:

container/
├─ client/
│ ├─ compose.yml #template
│ └─ .env
├─ superlink-certificates/
│ └─ ca.crt
└─ pyproject.toml



Notes:
- `.env` is **machine-specific** (server IP, local data path, project path).
- `ca.crt` is **provided by the server operator** and mounted by clients for secure communication.
