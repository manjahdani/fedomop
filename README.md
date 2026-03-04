# Client Guide — Federated Learning with Flower & Docker

**Author:** Dani Manjah  
**Last update:** 16/12/2025

This repository documents how to deploy and run a stateful (i.e., has an idea of its past) **Federated Learning (FL) experiments**
using **Flower** and **Docker Compose** in a distributed, multi-machine setup.

The goal is to:
- Ensure reproducible environments across clients
- Avoid “works on my machine” issues
- Enable scalable deployment (Docker / Kubernetes ready)
- Support client drop-in / drop-out without stopping FL

---

## Documentation

### Simulation 
- [Overview](docs/06_simu.md)


### Deploy
- [Overview](docs/00_overview.md)
- [Prerequisites](docs/01_prerequisites.md)
- [Server Setup](docs/02_server_setup.md)
- [Client Setup](docs/03_client_setup.md)
- [Running Experiments](docs/04_start_flower.md)
- [Troubleshooting](docs/05_troubleshooting.md)

---

## Scope

This guide focuses on **deployment and operations**.
It does **not** document FL algorithms or model internals.
