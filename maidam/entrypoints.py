import os

# Logic to pick the right one based on an ENV var
mode = os.getenv("MODE", "DFL").upper()
if mode == "DFL":
    from maidam.dfl.client_app import app as dfl_client
    from maidam.dfl.server_app import app as dfl_server
    server_app = dfl_server
    client_app = dfl_client
elif mode =="CFL":
    from maidam.cfl.client_app import app as cfl_client
    from maidam.cfl.server_app import app as cfl_server
    server_app = cfl_server
    client_app = cfl_client
else:
    raise(NotImplementedError)