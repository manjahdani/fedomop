#!/usr/bin/env python3
"""
Full analysis for Flower-style JSON result files.

What it does
- Loads result-*.json files
- Extracts per-round metrics: <metric> and <metric>_var (if present)
- Treats SPWF as a separate strategy per cooling_factor:
    "SPWF" -> "SPWF-(clf={cooling_factor})"
  So multiple cooling factors are not merged.
- Adds grouping keys: partition_split + parameter (derived from JSON run_config)
    * partition_split in {"iid", "natural"}  -> parameter = "none"
    * partition_split == "Dirichlet"        -> parameter = dataset_split_alpha (as string)
    * else                                  -> parameter = run_config["parameter"] if present, else "none"
- Collapses duplicate files safely by averaging within (strategy, seed, config, round)
- Produces results PER (partition_split, parameter):
  1) Per-seed summary table (final + best) and per-seed plots (<metric>, <metric>_var)
  2) Average-across-seeds plots (mean curves only; NO shading) for <metric> and <metric>_var
  3) Seed coverage + final-round mean/std tables (printed/saved)

NEW:
- Figure titles now include:
    dataset:<dataset> - num_clients:<number_of_nodes> - partition_split:<partition_split>
  and if partition_split == Dirichlet, also:
    - alpha=<dataset_split_alpha>   (here alpha is shown from "parameter")

Usage examples
  python full_analysis.py --glob 'results/**/result-*.json' --metric eval_acc --print-tables
  python full_analysis.py --glob 'results/**/result-*.json' --metric rmse --metric-mode min --save-dir ./out
  python full_analysis.py --glob 'results/**/result-*.json' --clients 6 --alpha 0.5 --cooling-factor 25 --metric eval_acc
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Helpers
# --------------------------

def ceil2(x: Any) -> Any:
    """Round UP to 2 decimals (ceil), consistent with user preference."""
    if x is None:
        return x
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return x
        return math.ceil(xf * 100.0) / 100.0
    except Exception:
        return x


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def pick_first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def format_number_like(x: Any) -> str:
    """Format numeric-like values so 25.0 -> '25', 0.5 -> '0.5'."""
    if x is None:
        return "none"
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return str(x)
        if xf.is_integer():
            return str(int(xf))
        return str(xf)
    except Exception:
        return str(x)


def normalize_strategy(strategy_raw: Any, cooling_factor: Any) -> Any:
    """
    Treat SPWF as a separate strategy per cooling_factor:
      SPWF -> SPWF-(clf={cooling_factor})
    """
    if strategy_raw is None:
        return None
    s = str(strategy_raw)
    if s == "SPWF":
        return f"SPWF-(clf={format_number_like(cooling_factor)})"
    return s


def metric_mode_auto(metric_name: str) -> str:
    """
    Heuristic:
      - minimize for loss-like metrics
      - maximize otherwise
    """
    m = metric_name.lower()
    minimize_tokens = ["loss", "rmse", "mse", "mae", "error"]
    return "min" if any(t in m for t in minimize_tokens) else "max"


def normalize_partition_split(x: Any) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip()
    return s if s else "unknown"


def derive_parameter(partition_split: str, dataset_split_alpha: Optional[float], explicit_param: Any) -> str:
    """
    Required behavior:
      - iid / natural: no parameter -> "none"
      - Dirichlet: parameter = dataset_split_alpha (each alpha is a set of experiments)
      - otherwise: if JSON has explicit parameter -> use it, else "none"
    """
    ps = partition_split.strip().lower()
    if ps in {"iid", "natural"}:
        return "none"
    if ps == "dirichlet":
        return format_number_like(dataset_split_alpha)
    if explicit_param is not None:
        return format_number_like(explicit_param)
    return "none"


def _safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def _unique_str(values: pd.Series, default: str = "unknown") -> str:
    vals = [str(v) for v in values.dropna().unique().tolist()]
    if not vals:
        return default
    vals_sorted = sorted(vals)
    return ",".join(vals_sorted)


def build_fig_title(sub: pd.DataFrame, partition_split: str, parameter: str, metric_label: str) -> str:
    dataset_str = _unique_str(sub["dataset"], default="unknown")
    num_clients_str = _unique_str(sub["num_clients"], default="unknown")

    title = f"dataset:{dataset_str} - num_clients:{num_clients_str} - partition_split:{partition_split}"
    if str(partition_split).strip().lower() == "dirichlet":
        # user requested: add alpha=$dataset_split_alpha; in this script alpha == "parameter"
        title += f" - alpha={parameter}"
    # keep metric label as last token (helps distinguish plots)
    title += f" - metric:{metric_label}"
    return title


# --------------------------
# Config signature
# --------------------------

@dataclass(frozen=True)
class ConfigSignature:
    num_clients: Optional[int]
    dataset_split_alpha: Optional[float]
    cooling_factor: Optional[float]
    partition_split: str
    parameter: str
    dataset: Optional[str]
    model: Optional[str]

    @staticmethod
    def from_run_config(cfg: Dict[str, Any]) -> "ConfigSignature":
        num_clients = pick_first(cfg, ["num_clients", "clients", "n_clients", "num-clients"])
        alpha = pick_first(cfg, ["dataset_split_alpha", "alpha", "dirichlet_alpha", "split_alpha"])
        cooling = pick_first(cfg, ["cooling_factor", "cooling", "cooling-factor"])

        partition_split_raw = pick_first(cfg, ["partition_split", "partition-split", "split", "partition"])
        partition_split = normalize_partition_split(partition_split_raw)

        explicit_param = pick_first(cfg, ["parameter", "param", "hyperparam", "hyperparameter"])

        dataset = pick_first(cfg, ["dataset", "data", "dataset_name"])
        model = pick_first(cfg, ["model", "model_name"])

        try:
            num_clients = int(num_clients) if num_clients is not None else None
        except Exception:
            num_clients = None
        try:
            alpha_f = float(alpha) if alpha is not None else None
        except Exception:
            alpha_f = None
        try:
            cooling_f = float(cooling) if cooling is not None else None
        except Exception:
            cooling_f = None

        parameter = derive_parameter(partition_split, alpha_f, explicit_param)

        return ConfigSignature(
            num_clients=num_clients,
            dataset_split_alpha=alpha_f,
            cooling_factor=cooling_f,
            partition_split=partition_split,
            parameter=parameter,
            dataset=str(dataset) if dataset is not None else None,
            model=str(model) if model is not None else None,
        )


# --------------------------
# Loading
# --------------------------

def load_results(paths: List[str], metric_key: str) -> pd.DataFrame:
    """
    Returns a long DataFrame with columns:
      file, strategy, seed, round,
      metric, metric_var,
      num_clients, dataset_split_alpha, cooling_factor,
      partition_split, parameter,
      dataset, model
    """
    metric_var_key = f"{metric_key}_var"
    rows: List[Dict[str, Any]] = []

    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)

        cfg = data.get("run_config", {}) or {}
        sig = ConfigSignature.from_run_config(cfg)

        strategy_raw = cfg.get("strategy", None)
        strategy = normalize_strategy(strategy_raw, sig.cooling_factor)

        seed = cfg.get("seed", None)
        try:
            seed_i = int(seed) if seed is not None else None
        except Exception:
            seed_i = None

        for rr in data.get("round_res", []) or []:
            metrics = (rr.get("evaluate_metrics_clientapp") or {})
            rows.append(
                {
                    "file": Path(p).name,
                    "strategy": strategy,
                    "seed": seed_i,
                    "round": int(rr.get("round")),
                    "metric": safe_float(metrics.get(metric_key)),
                    "metric_var": safe_float(metrics.get(metric_var_key)),
                    "num_clients": sig.num_clients,
                    "dataset_split_alpha": sig.dataset_split_alpha,  # kept for filtering only
                    "cooling_factor": sig.cooling_factor,
                    "partition_split": sig.partition_split,
                    "parameter": sig.parameter,
                    "dataset": sig.dataset,
                    "model": sig.model,
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["strategy", "seed", "round"], how="any")
    df["round"] = df["round"].astype(int)
    df["seed"] = df["seed"].astype(int)
    return df


def filter_df(
    df: pd.DataFrame,
    clients: Optional[int],
    alpha: Optional[float],
    cooling_factor: Optional[float],
) -> pd.DataFrame:
    out = df.copy()
    if clients is not None:
        out = out[out["num_clients"] == clients]
    if alpha is not None:
        a = pd.to_numeric(out["dataset_split_alpha"], errors="coerce")
        out = out[np.isclose(a, alpha, atol=1e-12, rtol=0)]
    if cooling_factor is not None:
        c = pd.to_numeric(out["cooling_factor"], errors="coerce")
        out = out[np.isclose(c, cooling_factor, atol=1e-12, rtol=0)]
    return out


# --------------------------
# Duplicate detection by trajectory
# --------------------------

def trajectory_groups(df_seed_round: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg_cols = ["partition_split", "parameter", "num_clients", "cooling_factor", "dataset", "model"]
    base_cols = ["strategy", "seed"] + cfg_cols

    records: List[Dict[str, Any]] = []

    for key, g in df_seed_round.groupby(base_cols, dropna=False):
        g = g.sort_values("round")
        rounds = g["round"].to_numpy()
        a1 = g["metric"].to_numpy()
        a2 = g["metric_var"].to_numpy()
        sig = (tuple(rounds.tolist()), tuple(a1.tolist()), tuple(a2.tolist()))
        records.append({**dict(zip(base_cols, key)), "trajectory_sig": sig})

    sig_df = pd.DataFrame(records)

    if sig_df.empty:
        dup_cols = base_cols + ["trajectory_id", "n_identical_trajectories"]
        uniq_cols = base_cols + ["trajectory_id", "trajectory_sig"]
        return pd.DataFrame(columns=dup_cols), pd.DataFrame(columns=uniq_cols)

    sig_df["trajectory_id"] = (
        sig_df.groupby(base_cols, dropna=False)["trajectory_sig"]
        .transform(lambda s: pd.factorize(s)[0] + 1)
    )

    dup_report = (
        sig_df.groupby(base_cols + ["trajectory_id"], dropna=False)
        .size()
        .reset_index(name="n_identical_trajectories")
        .sort_values(base_cols + ["trajectory_id"])
    )

    unique_first_keys = (
        sig_df.drop_duplicates(subset=base_cols + ["trajectory_id"], keep="first")
        .sort_values(base_cols + ["trajectory_id"])
        .reset_index(drop=True)
    )

    return dup_report, unique_first_keys


# --------------------------
# Summaries
# --------------------------

def per_seed_summary(df_seed_round: pd.DataFrame, mode: str) -> pd.DataFrame:
    cfg_cols = ["partition_split", "parameter", "num_clients", "cooling_factor", "dataset", "model"]
    key_cols = ["strategy", "seed"] + cfg_cols

    rows = []
    for key, g in df_seed_round.groupby(key_cols, dropna=False):
        g = g.sort_values("round")
        final = g.iloc[-1]

        metric_arr = g["metric"].to_numpy()
        if np.all(np.isnan(metric_arr)):
            best_idx = 0
        else:
            best_idx = int(np.nanargmax(metric_arr) if mode == "max" else np.nanargmin(metric_arr))
        best = g.iloc[best_idx]

        rows.append(
            {
                **dict(zip(key_cols, key)),
                "final_round": int(final["round"]),
                "metric@final": float(final["metric"]),
                "metric_var@final": float(final["metric_var"]),
                "best_round": int(best["round"]),
                "best metric": float(best["metric"]),
                "metric_var@best": float(best["metric_var"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["partition_split", "parameter", "strategy", "seed"])


def final_round_mean_std(df_seed_round: pd.DataFrame) -> pd.DataFrame:
    cfg_cols = ["partition_split", "parameter", "num_clients", "cooling_factor", "dataset", "model"]
    last_round = int(df_seed_round["round"].max())
    last_df = df_seed_round[df_seed_round["round"] == last_round].copy()

    group_cols = ["strategy"] + cfg_cols
    stats = (
        last_df.groupby(group_cols, dropna=False)[["metric", "metric_var"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = [
        c[0] if c[1] == "" else f"{c[0]}_{c[1]}"
        for c in stats.columns.to_flat_index()
    ]
    stats["round"] = last_round

    for c in ["metric_mean", "metric_std", "metric_var_mean", "metric_var_std"]:
        if c in stats.columns:
            stats[c] = stats[c].map(ceil2)

    return stats.sort_values(["partition_split", "parameter", "strategy"])


# --------------------------
# Plotting (NO SHADING) + TITLES
# --------------------------

def plot_per_seed(
    sub: pd.DataFrame,
    metric_key: str,
    save_dir: Optional[Path],
    tag: str,
    partition_split: str,
    parameter: str,
) -> None:
    """
    Two figures: <metric> and <metric>_var per round, for each strategy+seed.
    Titles include dataset/num_clients/partition_split (+alpha for Dirichlet).
    """
    dfp = sub.copy()
    dfp["label"] = dfp.apply(lambda r: f'{r["strategy"]} (seed {int(r["seed"])})', axis=1)

    # metric
    plt.figure(figsize=(11, 5.5))
    for label, g in dfp.groupby("label", dropna=False):
        g = g.sort_values("round")
        plt.plot(g["round"], g["metric"], label=label)
    plt.xlabel("Round")
    plt.ylabel(metric_key)
    plt.title(build_fig_title(dfp, partition_split, parameter, metric_label=metric_key))
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"per_seed_{metric_key}__{tag}.png", dpi=200)
    plt.show()

    # metric_var
    plt.figure(figsize=(11, 5.5))
    for label, g in dfp.groupby("label", dropna=False):
        g = g.sort_values("round")
        plt.plot(g["round"], g["metric_var"], label=label)
    plt.xlabel("Round")
    plt.ylabel(f"{metric_key}_var")
    plt.title(build_fig_title(dfp, partition_split, parameter, metric_label=f"{metric_key}_var"))
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"per_seed_{metric_key}_var__{tag}.png", dpi=200)
    plt.show()


def plot_mean_across_seeds(
    sub: pd.DataFrame,
    metric_key: str,
    save_dir: Optional[Path],
    tag: str,
    partition_split: str,
    parameter: str,
) -> None:
    """
    Two figures:
      - mean <metric> across seeds per strategy per round (NO shading)
      - mean <metric>_var across seeds per strategy per round (NO shading)
    Titles include dataset/num_clients/partition_split (+alpha for Dirichlet).
    """
    avg = (
        sub.groupby(["strategy", "round"], as_index=False, dropna=False)[["metric", "metric_var"]]
        .mean()
        .rename(columns={"metric": "metric_mean", "metric_var": "metric_var_mean"})
    )

    # mean metric
    plt.figure(figsize=(10.5, 5.5))
    for strategy, g in avg.groupby("strategy", dropna=False):
        g = g.sort_values("round")
        plt.plot(g["round"], g["metric_mean"], label=strategy)
    plt.xlabel("Round")
    plt.ylabel(f"{metric_key} (mean across seeds)")
    plt.title(build_fig_title(sub, partition_split, parameter, metric_label=f"{metric_key}_mean"))
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"mean_across_seeds_{metric_key}__{tag}.png", dpi=200)
    plt.show()

    # mean metric_var
    plt.figure(figsize=(10.5, 5.5))
    for strategy, g in avg.groupby("strategy", dropna=False):
        g = g.sort_values("round")
        plt.plot(g["round"], g["metric_var_mean"], label=strategy)
    plt.xlabel("Round")
    plt.ylabel(f"{metric_key}_var (mean across seeds)")
    plt.title(build_fig_title(sub, partition_split, parameter, metric_label=f"{metric_key}_var_mean"))
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"mean_across_seeds_{metric_key}_var__{tag}.png", dpi=200)
    plt.show()


# --------------------------
# Main
# --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True, help='File glob, e.g. "results/**/result-*.json"')

    ap.add_argument(
        "--metric",
        type=str,
        default="eval_acc",
        help=(
            "Metric key to extract from evaluate_metrics_clientapp (e.g., eval_acc, eval_auroc, rmse, loss). "
            "Variance key is assumed to be <metric>_var."
        ),
    )
    ap.add_argument(
        "--metric-mode",
        type=str,
        choices=["auto", "max", "min"],
        default="auto",
        help="How to define 'best' over rounds for the chosen metric.",
    )

    ap.add_argument("--clients", type=int, default=None, help="Filter: num_clients (e.g., 6)")
    ap.add_argument("--alpha", type=float, default=None, help="Filter: dataset_split_alpha (Dirichlet alpha)")
    ap.add_argument("--cooling-factor", type=float, default=None, help="Filter: cooling_factor (e.g., 25)")
    ap.add_argument("--save-dir", type=str, default=None, help="Optional directory to save plots/tables")
    ap.add_argument("--print-tables", action="store_true", help="Print tables to stdout as markdown")
    args = ap.parse_args()

    metric_key = args.metric.strip()
    mode = metric_mode_auto(metric_key) if args.metric_mode == "auto" else args.metric_mode

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched glob: {args.glob}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(paths, metric_key=metric_key)

    df = filter_df(df, args.clients, args.alpha, args.cooling_factor)
    if df.empty:
        raise SystemExit("After filtering, no rows remain. Check filters or run_config keys in your JSON.")

    # Collapse duplicates per (strategy, seed, config, round)
    cfg_cols = ["partition_split", "parameter", "num_clients", "cooling_factor", "dataset", "model"]
    group_cols = ["strategy", "seed"] + cfg_cols + ["round"]
    df_seed_round = (
        df.groupby(group_cols, as_index=False, dropna=False)[["metric", "metric_var"]]
        .mean()
        .sort_values(["partition_split", "parameter", "strategy", "seed", "round"])
    )

    if df_seed_round.empty:
        raise SystemExit(
            "df_seed_round is empty after grouping. Likely missing 'strategy'/'seed' or missing round_res/metrics."
        )

    # Global tables (across all partition_split/parameter)
    dup_report, _ = trajectory_groups(df_seed_round)
    seed_summary = per_seed_summary(df_seed_round, mode=mode)
    final_stats = final_round_mean_std(df_seed_round)

    seed_cov = (
        df_seed_round.groupby(["partition_split", "parameter", "strategy"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            seeds=("seed", lambda s: ", ".join(map(str, sorted(set(s.astype(int)))))),
            n_models=("model", "nunique"),
            models=("model", lambda s: ", ".join(sorted(set([str(x) for x in s.dropna().unique()])))),
            n_clients=("num_clients", lambda s: ", ".join(sorted(set([str(x) for x in s.dropna().unique()])))),
            datasets=("dataset", lambda s: ", ".join(sorted(set([str(x) for x in s.dropna().unique()])))),
        )
        .reset_index()
        .sort_values(["partition_split", "parameter", "strategy"])
    )

    # Display formatting (variance as x100)
    seed_summary_disp = seed_summary.copy()
    seed_summary_disp["metric@final"] = seed_summary_disp["metric@final"].map(ceil2)
    seed_summary_disp["best metric"] = seed_summary_disp["best metric"].map(ceil2)
    seed_summary_disp[f"{metric_key}_var@final (x100)"] = (seed_summary_disp["metric_var@final"] * 100).map(ceil2)
    seed_summary_disp[f"{metric_key}_var@best (x100)"] = (seed_summary_disp["metric_var@best"] * 100).map(ceil2)
    seed_summary_disp = seed_summary_disp.drop(columns=["metric_var@final", "metric_var@best"])
    seed_summary_disp = seed_summary_disp.rename(
        columns={
            "metric@final": f"{metric_key}@final",
            "best metric": f"best {metric_key}",
        }
    )

    # Save tables
    if save_dir:
        seed_summary_disp.to_csv(save_dir / f"per_seed_summary_{metric_key}.csv", index=False)
        dup_report.to_csv(save_dir / f"duplicate_trajectory_report_{metric_key}.csv", index=False)
        seed_cov.to_csv(save_dir / "seed_coverage.csv", index=False)
        final_stats.to_csv(save_dir / f"final_round_mean_std_{metric_key}.csv", index=False)

    # Print tables
    if args.print_tables:
        print(f"\n## Seed coverage (metric={metric_key}, best mode={mode}) [grouped by partition_split, parameter]")
        print(seed_cov.to_markdown(index=False))

        print(f"\n## Per-seed summary (metric={metric_key}, best mode={mode}; variance shown as x100)")
        print(seed_summary_disp.to_markdown(index=False))

        print(f"\n## Final round mean/std across seeds (metric={metric_key})")
        print(final_stats.to_markdown(index=False))

        print(f"\n## Duplicate trajectory report (metric={metric_key})")
        print(dup_report.to_markdown(index=False))

    # Plots PER (partition_split, parameter)
    combos = (
        df_seed_round[["partition_split", "parameter"]]
        .drop_duplicates()
        .sort_values(["partition_split", "parameter"])
        .itertuples(index=False, name=None)
    )

    for partition_split, parameter in combos:
        sub = df_seed_round[
            (df_seed_round["partition_split"] == partition_split) &
            (df_seed_round["parameter"] == parameter)
        ].copy()

        if sub.empty:
            continue

        tag = _safe_slug(f"ps={partition_split}__param={parameter}")
        plot_per_seed(
            sub,
            metric_key=metric_key,
            save_dir=save_dir,
            tag=tag,
            partition_split=partition_split,
            parameter=parameter,
        )
        plot_mean_across_seeds(
            sub,
            metric_key=metric_key,
            save_dir=save_dir,
            tag=tag,
            partition_split=partition_split,
            parameter=parameter,
        )


if __name__ == "__main__":
    main()
