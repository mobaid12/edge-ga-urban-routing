"""
utils.py
========
Preprocessing, normalisation, and graph-building utilities.

Paper: "Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing"
Repository: https://github.com/MahmoudObaid-AAUP/edge-ga-urban-routing

Contents
--------
- build_route_graph()     : Parse a SUMO .net.xml and build an adjacency-dict graph
- normalise_objectives()  : Min-max normalisation (equation 2 of the paper)
- load_vehicle_trajectories() : Load and preprocess vehicle_trajectories.csv
- load_signal_logs()      : Load and preprocess signal_logs.csv
- aggregate_time_slice()  : Aggregate link data into 5-minute rolling slices
- compute_od_matrix()     : Build O-D demand matrix from trajectory data
- statistical_tests()     : Paired t-test and two-way ANOVA helpers (Section III-D)
- plot_pareto_front()     : 3-D Pareto front visualisation
- plot_convergence()      : Objective convergence curves
"""

from __future__ import annotations

import os
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Graph construction from SUMO .net.xml
# ---------------------------------------------------------------------------

def build_route_graph(
    network_file: str,
    default_capacity_vph: float = 1800.0,
    default_speed_kmh: float = 50.0,
) -> Tuple[Dict, List[Tuple[str, str]], int]:
    """
    Parse a SUMO network file and build a weighted directed adjacency dict.

    Parameters
    ----------
    network_file          : path to SUMO .net.xml file
    default_capacity_vph  : fallback link capacity (vehicles/hour)
    default_speed_kmh     : fallback free-flow speed (km/h)

    Returns
    -------
    graph               : {node_id: {neighbour_id: {attrs}}}
    od_pairs            : list of (origin, destination) node pairs
    num_intersections   : number of signalised junctions
    """
    try:
        import sumolib
        net = sumolib.net.readNet(network_file)
        graph: Dict[str, Dict] = {}

        for edge in net.getEdges():
            u = edge.getFromNode().getID()
            v = edge.getToNode().getID()
            length    = edge.getLength()                            # metres
            ffspeed   = edge.getSpeed()                            # m/s
            num_lanes = edge.getLaneNumber()
            capacity  = num_lanes * default_capacity_vph           # veh/h (per lane)

            if u not in graph:
                graph[u] = {}
            graph[u][v] = {
                "edge_id":        edge.getID(),
                "length":         round(length, 2),
                "free_flow_speed": round(ffspeed, 3),
                "capacity":       capacity,
                "free_flow_time": round(length / max(ffspeed, 0.5), 3),
                "num_lanes":      num_lanes,
                "weight":         round(length / max(ffspeed, 0.5), 3),
            }

        # Collect signalised junctions
        tl_junctions = [j for j in net.getNodes() if j.getType() == "traffic_light"]
        num_intersections = len(tl_junctions)

        # Build O-D pairs from unique node pairs
        nodes = list(graph.keys())
        od_pairs = _sample_od_pairs(nodes, n_pairs=min(50, len(nodes) * (len(nodes) - 1)))

        print(
            f"[build_route_graph] Nodes={len(graph)} | "
            f"Edges={sum(len(v) for v in graph.values())} | "
            f"TL junctions={num_intersections}"
        )
        return graph, od_pairs, num_intersections

    except ImportError:
        warnings.warn("sumolib not found; using mock graph for Bethlehem network.")
        return _build_mock_bethlehem_graph()


def _sample_od_pairs(
    nodes: List[str], n_pairs: int
) -> List[Tuple[str, str]]:
    """Sample n_pairs unique O-D pairs from the node list."""
    import random
    pairs: set = set()
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        o = random.choice(nodes)
        d = random.choice(nodes)
        if o != d:
            pairs.add((o, d))
        attempts += 1
    return list(pairs)


def _build_mock_bethlehem_graph() -> Tuple[Dict, List[Tuple[str, str]], int]:
    """
    Construct a mock adjacency graph approximating the 15-intersection
    Bethlehem road network for use when sumolib is unavailable.
    """
    intersection_ids = [f"INT_{i:02d}" for i in range(1, 16)]
    # Simple grid-like connectivity
    connections = [
        ("INT_01", "INT_02"), ("INT_02", "INT_03"), ("INT_03", "INT_04"),
        ("INT_01", "INT_05"), ("INT_05", "INT_06"), ("INT_06", "INT_07"),
        ("INT_07", "INT_08"), ("INT_08", "INT_09"), ("INT_09", "INT_10"),
        ("INT_02", "INT_06"), ("INT_03", "INT_07"), ("INT_04", "INT_08"),
        ("INT_10", "INT_11"), ("INT_11", "INT_12"), ("INT_12", "INT_13"),
        ("INT_13", "INT_14"), ("INT_14", "INT_15"), ("INT_15", "INT_01"),
        ("INT_05", "INT_11"), ("INT_09", "INT_13"),
    ]
    graph: Dict = {n: {} for n in intersection_ids}
    for u, v in connections:
        length   = float(np.random.uniform(300, 2000))
        ffspeed  = float(np.random.uniform(8.3, 16.7))   # 30–60 km/h
        capacity = float(np.random.choice([900, 1200, 1800]))
        attrs = {
            "edge_id":         f"{u}_{v}",
            "length":          round(length, 2),
            "free_flow_speed": round(ffspeed, 3),
            "capacity":        capacity,
            "free_flow_time":  round(length / ffspeed, 3),
            "num_lanes":       int(capacity / 900),
            "weight":          round(length / ffspeed, 3),
        }
        graph[u][v] = attrs
        # Make bidirectional
        graph[v][u] = {**attrs, "edge_id": f"{v}_{u}"}

    od_pairs = [(intersection_ids[i], intersection_ids[j])
                for i in range(15) for j in range(15) if i != j]
    return graph, od_pairs, 15  # 15 signalised intersections


# ---------------------------------------------------------------------------
# Objective normalisation
# ---------------------------------------------------------------------------

def normalise_objectives(
    obj_matrix: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Min-max normalisation across a matrix of objective values (equation 2 of the paper):

        f_norm = (f - f_min) / (f_max - f_min + eps)

    Parameters
    ----------
    obj_matrix : shape (N, M) – N solutions, M objectives
    eps        : small constant to avoid division by zero

    Returns
    -------
    Normalised array of shape (N, M), values in [0, 1]
    """
    f_min = obj_matrix.min(axis=0)
    f_max = obj_matrix.max(axis=0)
    return (obj_matrix - f_min) / (f_max - f_min + eps)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_vehicle_trajectories(
    csv_path: str = "dataset/vehicle_trajectories.csv",
    parse_timestamps: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess the vehicle trajectory dataset.

    Returns a DataFrame with 200,000 rows and typed columns.
    """
    dtype_map = {
        "vehicle_id":          "str",
        "day_of_week":         "category",
        "origin_intersection": "category",
        "dest_intersection":   "category",
        "vehicle_type":        "category",
        "period":              "category",
    }
    df = pd.read_csv(csv_path, dtype=dtype_map)

    if parse_timestamps and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Derived features
    df["hour"] = df["timestamp"].dt.hour if parse_timestamps else 0
    df["is_peak"] = df["period"] == "peak"

    print(
        f"[load_vehicle_trajectories] {len(df):,} rows | "
        f"{df.columns.tolist()}"
    )
    return df


def load_signal_logs(
    csv_path: str = "dataset/signal_logs.csv",
    parse_timestamps: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess the signal log dataset.

    Filters out rows where sensor_active == 0 (sensor outage).
    Returns a DataFrame with 5-minute slot records.
    """
    df = pd.read_csv(csv_path)

    if parse_timestamps and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Keep only active sensor records
    if "sensor_active" in df.columns:
        n_before = len(df)
        df = df[df["sensor_active"] == 1].copy()
        print(
            f"[load_signal_logs] {len(df):,} active rows "
            f"(dropped {n_before - len(df):,} outage rows)"
        )
    return df


# ---------------------------------------------------------------------------
# Time-slice aggregation (Δt = 5 min, as used in SUMO TraCI interface)
# ---------------------------------------------------------------------------

def aggregate_time_slice(
    df_trajectories: pd.DataFrame,
    delta_t_minutes: int = 5,
) -> pd.DataFrame:
    """
    Aggregate vehicle trajectory data into rolling Δt-minute time slices,
    producing the link-level inputs used to update GA fitness values.

    Returns a DataFrame with columns:
        time_slot | origin_intersection | dest_intersection |
        mean_travel_time_s | mean_congestion | vehicle_count
    """
    df = df_trajectories.copy()
    df["time_slot"] = df["timestamp"].dt.floor(f"{delta_t_minutes}min")

    agg = (
        df.groupby(["time_slot", "origin_intersection", "dest_intersection"])
        .agg(
            mean_travel_time_s=("travel_time_s", "mean"),
            mean_waiting_time_s=("waiting_time_s", "mean"),
            mean_congestion=("congestion_index", "mean"),
            vehicle_count=("vehicle_id", "count"),
        )
        .reset_index()
    )
    return agg


def compute_od_matrix(
    df_trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an origin-destination demand matrix from the trajectory dataset.

    Returns a pivot table (intersections × intersections) with vehicle counts.
    """
    counts = (
        df_trajectories
        .groupby(["origin_intersection", "dest_intersection"])["vehicle_id"]
        .count()
        .reset_index(name="demand")
    )
    pivot = counts.pivot(
        index="origin_intersection",
        columns="dest_intersection",
        values="demand",
    ).fillna(0).astype(int)
    return pivot


# ---------------------------------------------------------------------------
# Statistical analysis helpers (Section III-D of the paper)
# ---------------------------------------------------------------------------

def paired_t_test(
    ga_values: List[float],
    dijkstra_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Paired t-test comparing GA vs. Dijkstra performance metrics.

    Parameters
    ----------
    ga_values        : list of per-run GA metric values (e.g. travel times)
    dijkstra_values  : list of per-run Dijkstra metric values (same ordering)
    alpha            : significance level (default 0.05)

    Returns
    -------
    dict with t_statistic, p_value, significant, mean_diff, ci_lower, ci_upper
    """
    ga  = np.array(ga_values, dtype=float)
    dij = np.array(dijkstra_values, dtype=float)
    t_stat, p_val = stats.ttest_rel(ga, dij)
    diff = ga - dij
    se = diff.std(ddof=1) / math.sqrt(len(diff))
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(diff) - 1)

    return {
        "t_statistic":  round(float(t_stat), 4),
        "p_value":      round(float(p_val), 6),
        "significant":  bool(p_val < alpha),
        "mean_diff":    round(float(diff.mean()), 4),
        "ci_lower":     round(float(diff.mean() - t_crit * se), 4),
        "ci_upper":     round(float(diff.mean() + t_crit * se), 4),
    }


def two_way_anova(
    data: pd.DataFrame,
    metric_col: str,
    factor1_col: str = "algorithm",     # "GA" / "Dijkstra"
    factor2_col: str = "congestion",    # "low" / "medium" / "high"
) -> Dict[str, Any]:
    """
    Two-way ANOVA on a performance metric as a function of algorithm type
    and congestion level (replicating the analysis in Section III-D).

    Parameters
    ----------
    data        : DataFrame with columns [metric_col, factor1_col, factor2_col]
    metric_col  : name of the dependent variable column

    Returns
    -------
    dict with F-statistics and p-values for main effects and interaction
    """
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        warnings.warn("statsmodels not installed; returning dummy ANOVA result.")
        return {"error": "statsmodels not installed"}

    formula = f"{metric_col} ~ C({factor1_col}) + C({factor2_col}) + C({factor1_col}):C({factor2_col})"
    model  = ols(formula, data=data).fit()
    table  = anova_lm(model, typ=2)

    return {
        f"F_{factor1_col}":     round(float(table.loc[f"C({factor1_col})", "F"]), 4),
        f"p_{factor1_col}":     round(float(table.loc[f"C({factor1_col})", "PR(>F)"]), 6),
        f"F_{factor2_col}":     round(float(table.loc[f"C({factor2_col})", "F"]), 4),
        f"p_{factor2_col}":     round(float(table.loc[f"C({factor2_col})", "PR(>F)"]), 6),
        "F_interaction":        round(float(table.loc[f"C({factor1_col}):C({factor2_col})", "F"]), 4),
        "p_interaction":        round(float(table.loc[f"C({factor1_col}):C({factor2_col})", "PR(>F)"]), 6),
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_pareto_front(
    pareto_front,
    output_path: str = "results/pareto_front.png",
    title: str = "NSGA-II Pareto Front",
    knee_solution=None,
) -> None:
    """
    Render a 3-D scatter plot of the Pareto front objectives.
    Highlights the knee-point solution if provided.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

        obj = np.array([ind.objectives for ind in pareto_front])
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")

        ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2],
                   c="steelblue", s=60, alpha=0.8, label="Pareto front")

        if knee_solution is not None:
            kp = knee_solution.objectives
            ax.scatter([kp[0]], [kp[1]], [kp[2]],
                       c="red", s=120, marker="*", label="Knee-point", zorder=5)

        ax.set_xlabel("f₁ – Travel time (s)", labelpad=10)
        ax.set_ylabel("f₂ – Congestion index", labelpad=10)
        ax.set_zlabel("f₃ – Network delay (s)", labelpad=10)
        ax.set_title(title)
        ax.legend()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[plot_pareto_front] Saved → {output_path}")
    except ImportError:
        warnings.warn("matplotlib not installed; skipping Pareto front plot.")


def plot_convergence(
    history: List[List[float]],
    output_path: str = "results/convergence.png",
    title: str = "NSGA-II Convergence – Mean Objectives per Generation",
) -> None:
    """
    Plot mean objective values per generation to illustrate NSGA-II convergence.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr  = np.array(history)
        gens = np.arange(1, len(arr) + 1)
        labels = ["f₁ Travel time (s)", "f₂ Congestion index", "f₃ Delay (s)"]
        colours = ["steelblue", "darkorange", "forestgreen"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for m, (ax, label, colour) in enumerate(zip(axes, labels, colours)):
            ax.plot(gens, arr[:, m], color=colour, linewidth=1.5)
            ax.set_xlabel("Generation")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(title, fontsize=13)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[plot_convergence] Saved → {output_path}")
    except ImportError:
        warnings.warn("matplotlib not installed; skipping convergence plot.")


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def load_scenarios(
    csv_path: str = "dataset/simulation_scenarios.csv",
) -> pd.DataFrame:
    """
    Load SUMO scenario configurations.

    Returns a DataFrame with 60 rows (scenarios) and their GA/Dijkstra results.
    """
    df = pd.read_csv(csv_path)
    df["travel_time_reduction_pct"] = (
        (df["dijkstra_travel_time_s"] - df["ga_travel_time_s"])
        / df["dijkstra_travel_time_s"] * 100
    ).round(2)
    df["waiting_time_reduction_pct"] = (
        (df["dijkstra_waiting_time_s"] - df["ga_waiting_time_s"])
        / df["dijkstra_waiting_time_s"] * 100
    ).round(2)
    return df


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== utils.py sanity checks ===")

    # Normalisation
    obj = np.array([[300, 0.6, 5000], [400, 0.8, 7000], [350, 0.7, 6000]])
    norm = normalise_objectives(obj)
    print(f"Normalised objectives:\n{norm}")
    assert norm.max() <= 1.0 and norm.min() >= 0.0, "Normalisation out of [0,1]"

    # Mock graph
    graph, od_pairs, n_int = _build_mock_bethlehem_graph()
    print(f"Mock graph: {len(graph)} nodes | {len(od_pairs)} O-D pairs | {n_int} TL")

    # t-test mock
    ga_times  = list(np.random.normal(320, 30, 20))
    dij_times = list(np.random.normal(380, 35, 20))
    result    = paired_t_test(ga_times, dij_times)
    print(f"Paired t-test: {result}")

    print("All checks passed.")
