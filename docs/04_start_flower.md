
# Running Experiments

From the FL project directory:

When you are inside the directory MAIDAM
```bash
flwr run . remote-deployment --stream
```

Notes:

. is to indicate the current directory, where the pyproject toml should be there.

remote-deployment is the deployment config name (replace if needed).

--stream streams logs and is useful for debugging