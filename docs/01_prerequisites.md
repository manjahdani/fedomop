# Prerequisites

## System requirements
- Linux (Ubuntu recommended) or Windows + WSL2 
- Stable network connection
- Docker + Docker Compose (or Docker Desktop on Windows) — see [Docker Installation](#docker-installation)
- Flower CLI
- TCP access to the Superlink port (see [Connectivity](#connectivity))

⚠️ If Docker or WSL were already installed, **update them** to avoid incompatibilities.

---


## Docker Installation

### Windows
Install Docker Desktop:  
https://docs.docker.com/desktop/setup/install/windows-install/

Verify:
```bash
docker run -d -p 8080:80 docker/welcome-to-docker
```

### Linux (Ubuntu)
Install Docker Engine

https://docs.docker.com/engine/install/ubuntu/

Install Docker Compose

https://docs.docker.com/desktop/setup/install/linux/ubuntu/


## Connectivity 
nc -vz -w 3 <SERVER_IP> 9092

Expected output:

Connection to <SERVER_IP> 9092 port [tcp/*] succeeded!

If TCP fails, Flower will not connect regardless of Docker correctness.