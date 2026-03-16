# Personalized Federated Framework with Flower & Docker for OMOP-CDM Multi-Hospital Readmission

**Author:** Dani Manjah and Pierre Remacle

**Last update:** 07/03/2026
---

## About

This repository documents how to run simulations and deploy **Federated Learning (FL) experiments**
using **Flower** in a distributed, multi-machine setup for OMOP-CDM Multi-Hospital Data. The use case of readmission within 30 days is taken as an illustrative example.
 
---
> **Note**  
> This repository uses a simplified demonstration dataset.  
> The full experimental archive described in the paper is not publicly distributed and is planned for a future release.

---

## Installation

You can install the project either with `venv` or Conda.

### Option 1 — Python virtual environment

```bash
python -m venv fedomop
source fedomop/bin/activate
pip install --upgrade pip
pip install -e .
```

### Option 2 — Conda

```bash
conda create -n fedomop python=3.10
conda activate fedomop
pip install -e .
```

---

## Dataset Generation

This pipeline preprocesses **MIMIC-IV v2.2** Electronic Health Record (EHR) data into structured **static** and **time-series** features.

The code snippet provided here is dedicated to the **readmission** use case.  
The same overall pipeline can be adapted to other tasks such as:
- mortality prediction
- length of stay
- phenotyping

---

### Dataset Access

Before downloading the data, access must be approved through the official **PhysioNet** (PhysioNet portal:  
https://physionet.org) data use agreement.

Once access is granted:
1. Download the v2.2 **MIMIC-IV** version.
2. Place the raw files in the directory specified by `RawDataPath` in the configuration file of "pipeline_with_checkpoints.py".
3. For the readmission pipeline, use the `base_config` defined in the code:

```bash
python fedomop/preprocess_MIMIC/pipeline_with_checkpoints.py
```

This generates a csv containing the feature matrix `X` and the readmission target `y` in "fedomop/preprocess_MIMIC/data/output".


For more details about the data pipeline and outputs, see:
- [MIMIC-IV Overview](docs/mimiciv.md)
---

## Running Experiments

### 1. Simulation Mode

Simulation is the default mode in this repository.

Run a fully local federated simulation with:

```bash
flwr run .
```

This will:
- spawn virtual clients
- partition the dataset
- train the federated model
- log metrics

#### Simulation configuration

The `local-simulation` runtime is defined in the Flower configuration file:

```bash
~/.flwr/config.toml
```

Example:

```toml
[superlink.local-simulation]
options.num-supernodes = 3
```

This runs the simulation locally with **3 virtual SuperNodes (clients)**.

#### Custom simulation parameters

You can override parameters defined in `pyproject.toml` with `--run-config`:

```bash
flwr run . --run-config='partitioner="dirichlet" num-server-rounds=50 local-epochs=100'
```

---

### 2. Deployment Mode

Deployment mode simulates a real multi-hospital distributed setup.

For each link and node, start a dedicated terminal.

#### Step 1 — Start the SuperLink

```bash
flower-superlink --insecure
```

#### Step 2 — Start the SuperNodes

Example with 3 hospitals:

```bash
flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9104 \
    --node-config "partition-id=0 num-partitions=3"
```

```bash
flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9105 \
    --node-config "partition-id=1 num-partitions=3"
```

```bash
flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9106 \
    --node-config "partition-id=2 num-partitions=3"
```

#### Step 3 — Launch the federated run

```bash
flwr run . local-deployment --stream
```

The `local-deployment` runtime is defined in `config.toml`:

```toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

---

## Metrics and Outputs

The framework reports both **centralized** and **distributed** metrics per round, including:
- loss
- accuracy
- AUROC
- AUPR

It also tracks summary statistics across clients, including:
- variance
- minimum

Simulation results are automatically saved in the `results/` directory. And the final model is uploaded as .pt

---

## License

This app is open-source under the **Apache 2.0 License**.

---

## Funding

This project was developed as part of the **MAIDAM** BioWin project funded by the **Walloon Region** under grant agreement:

**PIT ATMP - Convention 8881**
