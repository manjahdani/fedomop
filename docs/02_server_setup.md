# Server Setup

This section is for the machine hosting the **Superlink / server**.

---

## GENERATE THE CERTIFICATES

## Certificates

Federated communication uses TLS certificates.

- A CA certificate (`ca.crt`) is generated on the server side
- Clients must mount this certificate into their containers

IDENTIFY THE SERVER IP ADDRESS and put it in an .env file or do the follwing in the terminal: 

example: 

export SUPERLINK_IP=192.168.2.33

$ docker compose -f certs.yml -f ../containers/certs.yml run --rm --build gen-certs

NOTE THIS WILL GENERATE A folder "superlink-certificates" that will have to be put in every client deployment bundle see [Docker Installation](#docker-installation)

To be stateful : 

When running Docker Compose on Linux, the server may need a writable state directory.

mkdir -p server/state
sudo chown -R 49999:49999 server/state

