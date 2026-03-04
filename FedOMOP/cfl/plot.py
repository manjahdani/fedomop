# plot.py
# Live per-client plotting (one subplot per client), with:
#  - larger/clearer global labels only (no per-subplot axis labels)
#  - ability to select which clients to display
#  - legend names: FedAvg (server) vs FedCoal (self/ours)

import math
from collections import defaultdict
from typing import Iterable, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_plot_state = {
    "init": False,
    "round": 0,
    "rounds": [],
    "by_client_srv": defaultdict(list),   # cid -> list of rel_mse (FedAvg/server)
    "by_client_self": defaultdict(list),  # cid -> list of rel_mse (FedCoal/self)
    "fig": None,
    "axs": None,
    "layout_n": 0,
    "filter": None,                       # optional: set of cids to display (or None for all)
}


def set_plot_clients(clients: Optional[Iterable[int]] = None) -> None:
    """
    Select which clients to show.
      - clients=None  -> show all clients (default)
      - clients=[...] -> show only these client IDs

    Call this anytime; the next _update_plot call will adapt the layout.
    """
    st = _plot_state
    if clients is None:
        st["filter"] = None
    else:
        st["filter"] = set(clients)


def _ensure_layout(num_clients: int) -> None:
    """Create/refresh the subplot grid to fit num_clients panels."""
    st = _plot_state
    # Keep at least one panel to avoid matplotlib errors
    num_clients = max(1, num_clients)

    if st["layout_n"] == num_clients and st["fig"] is not None:
        return

    # Close old fig if layout changed
    if st["fig"] is not None:
        plt.close(st["fig"])

    # square-ish grid
    cols = max(1, math.ceil(math.sqrt(num_clients)))
    rows = math.ceil(num_clients / cols)

    st["fig"], st["axs"] = plt.subplots(
        rows,
        cols,
        sharex=True,
        sharey=True,
        figsize=(5.2 * cols, 3.8 * rows),
        constrained_layout=False,
    )
    if not isinstance(st["axs"], (list, tuple, np.ndarray)):
        st["axs"] = [st["axs"]]
    else:
        st["axs"] = st["axs"].ravel()

    st["layout_n"] = num_clients
    st["fig"].suptitle(
        "Flower – per-client rel_mse (FedAvg vs SPWF)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    # Global labels (only)
    st["fig"].supxlabel("Round", fontsize=14)
    st["fig"].supylabel("Relative MSE", fontsize=14)

    # Reserve a bit of top space for the suptitle and legend
    plt.tight_layout(rect=(0, 0, 1, 0.93))


def _update_plot(perf_by_server: Dict[int, float], perf_by_client: Dict[int, float]) -> None:
    """
    Update the live plot.

    Parameters
    ----------
    perf_by_server : dict[int, float]
        Per-client rel_mse measured on the global (FedAvg) model for this round.
    perf_by_client : dict[int, float]
        Per-client rel_mse measured on the personalized (SPWF) model for this round.
    """
    st = _plot_state
    if not st["init"]:
        plt.ion()
        st["init"] = True

    st["round"] += 1
    st["rounds"].append(st["round"])

    # union of cids for full history tracking
    cids_all = sorted(set(perf_by_server) | set(perf_by_client))

    # append to full histories
    for cid in cids_all:
        if cid in perf_by_server:
            st["by_client_srv"][cid].append(float(perf_by_server[cid]))
        if cid in perf_by_client:
            st["by_client_self"][cid].append(float(perf_by_client[cid]))

    # visible subset (filter) for this redraw
    if st["filter"] is None:
        cids = cids_all
    else:
        cids = [cid for cid in cids_all if cid in st["filter"]]

    _ensure_layout(len(cids))

    # clear/prepare axes
    for ax in st["axs"]:
        ax.clear()
        ax.set_visible(False)

    # plot each visible client on its own subplot
    for i, cid in enumerate(cids):
        ax = st["axs"][i]
        ax.set_visible(True)

        s_series = st["by_client_srv"].get(cid, [])
        p_series = st["by_client_self"].get(cid, [])

        # Align x with each series length
        if s_series:
            xs = st["rounds"][-len(s_series):]
            ax.plot(xs, s_series, label="FedAvg", color="C0", linewidth=2.2)
        if p_series:
            xs = st["rounds"][-len(p_series):]
            ax.plot(xs, p_series, label="FedCoal", color="C1", linestyle="--", linewidth=2.2)

        # Keep only per-subplot titles; remove per-subplot axis labels
        ax.set_title(f"Client {cid}", fontsize=13, fontweight="semibold")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.35, linestyle=":")
        ax.margins(x=0.03, y=0.08)

    # single shared legend (FedAvg vs FedCoal)
    legend_elems = [
        Line2D([0], [0], color="C0", linewidth=2.2, label="FedAvg"),
        Line2D([0], [0], color="C1", linestyle="--", linewidth=2.2, label="PWF(ours)"),
    ]
    st["fig"].legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        fontsize=20,
        frameon=True,
    )

    st["fig"].canvas.draw()
    st["fig"].canvas.flush_events()
