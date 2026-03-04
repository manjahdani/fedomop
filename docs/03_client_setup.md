# Client Setup

This section is for each **client machine**.
---


## Machine-specific .env

Each client has its own `.env` file:

IT HAS TO BE ADAPTED

```env
SUPERLINK_IP=130.104.204.58
PROJECT_DIR=../ 
HOST_DATA_DIR=/export/home/manjah/data 
```


SUPERLINK_IP: IP of the server hosting Superlink

PROJECT_DIR: path to the Flower app build context (where pyproject.toml is copied from)

HOST_DATA_DIR: local path on this machine containing datasets (mounted read-only)



## Data Structure

Your Data Folder should look like this

data/
├─ ds1/
├─ ds2/
└─ cho/ #if CHO is what you are using.

and you just have to provide the path to your data folder 

for example: 

```
/home/user/data
```

## Create Container
```bash
cd containers
docker compose -f client/compose.yml up -d
```
UNDERSTAND the commmand: 

docker compose
up

## Basic check 
docker ps

You should see your supernode-* and superexec-clientapp-* containers running as following:

CONTAINER ID   IMAGE                          COMMAND                  CREATED      STATUS       PORTS     NAMES
76c7e469757d   client-superexec-clientapp-1   "flower-superexec --…"   7 days ago   Up 8 hours             client-superexec-clientapp-1-1
e253ff25acb4   flwr/supernode:1.24.0          "flower-supernode --…"   7 days ago   Up 8 hours             client-supernode-1-1


# Adding a new client

This should be provided for each machine: 

container/
├─ client/
│ ├─ compose.yml #template
│ └─ .env
├─ superlink-certificates/
│ └─ ca.crt
└─ pyproject.toml

an .env file should have the most relevant supervariables

To add a client:
1. Duplicate an existing service
2. Change:
   - Service name
   - `clientappio-api-address` port
   - `partition-id`
   - `num-partitions`

Example:
```yaml
--node-config "partition-id=4 num-partitions=5"



@COMING COMPOSE.YAML GENERATOR
## Advanced 
HOW TO UNderstand and CREATE A COMPOSE OF CLIENT : 

```yaml
services:
  supernode-5:
    image: flwr/supernode:${FLWR_VERSION:-1.24.0}
    command:
      - --superlink
      - ${SUPERLINK_IP:-127.0.0.1}:9092
      - --clientappio-api-address
      - 0.0.0.0:9098 #CHANGE
      - --isolation
      - process
      - --node-config
      - "partition-id=4 num-partitions=5" #Change
      - --root-certificates
      - certificates/superlink-ca.crt
    secrets:
      - source: superlink-ca-certfile
        target: /app/certificates/superlink-ca.crt


  superexec-clientapp-5:
    build:
      context: ${PROJECT_DIR:-.}
      dockerfile_inline: |
        FROM flwr/superexec:${FLWR_VERSION:-1.24.0}


        WORKDIR /app
        COPY --chown=app:app pyproject.toml .
        RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
          && python -m pip install -U --no-cache-dir .


        ENTRYPOINT ["flower-superexec"]
    command:
      - --insecure
      - --plugin-type
      - clientapp
      - --appio-api-address
      - supernode-5:9098 #Change
    volumes:
      - ${HOST_DATA_DIR}:/home/dani/data:ro
    environment:
        - INSILICO_DATA_DIR=/home/dani/data/cho
    deploy:
      resources:
        limits:
          cpus: "2"
    stop_signal: SIGINT
    depends_on:
      - supernode-5 #Change


secrets:
  superlink-ca-certfile:
    file: ../superlink-certificates/ca.crt

    Ports must not collide with other client instances on the same machine.

```