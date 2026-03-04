

Simulation is the default mode of this redo. You can simply do 


Create a running environment : 

```bash
conda create -n fedomop python=3.10
```

```bash
conda activate fedomop
```

```bash
pip install -e .
```

```bash
flwr run . --run-config='strategy="FedAvg"'
```

