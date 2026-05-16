"""
objectives.py
=============
Multi-objective fitness functions for the NSGA-II urban routing framework.

Paper: "Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing"
Repository: https://github.com/MahmoudObaid-AAUP/edge-ga-urban-routing

Three objectives are minimised simultaneously (equation 1 of the paper):

    F = { f1(x), f2(x), f3(x) }

  f1(x) – Average travel time (seconds)
         Mean link traversal time, weighted by current congestion.

  f2(x) – Congestion intensity (normalised vehicle density per link [0, 1])
         Computed as mean occupancy across all route links.

  f3(x) – Total network delay (seconds)
         Sum of intersection waiting times along the route.

All objectives are normalised to [0, 1] via min-max scaling (equation 2 of the
paper) before being presented to the NSGA-II dominance comparator, to prevent
any single objective from dominating due to scale differences.
"""

from __future__ import annotations

import math
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sumo_interface import SumoInterface


# ---------------------------------------------------------------------------
# BPR (Bureau of Public Roads) travel-time function
# ---------------------------------------------------------------------------

def bpr_travel_time(
    free_flow_time: float,
    volume: float,
    capacity: float,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float:
    """
    BPR link performance function:

        t = t0 * (1 + alpha * (v/c)^beta)

    Parameters
    ----------
    free_flow_time : seconds at zero congestion
    volume         : current volume (vehicles / hour)
    capacity       : link capacity (vehicles / hour)
    alpha, beta    : BPR calibration parameters (default: 0.15, 4.0)

    Returns
    -------
    Congested travel time in seconds.
    """
    if capacity <= 0:
        return free_flow_time
    vc_ratio = volume / capacity
    return free_flow_time * (1.0 + alpha * (vc_ratio ** beta))


# ---------------------------------------------------------------------------
# Per-link congestion index
# ---------------------------------------------------------------------------

def link_congestion_index(
    current_speed: float, free_flow_speed: float
) -> float:
    """
    Congestion index for a single link:

        CI = 1 - (v_current / v_free_flow)   ∈ [0, 1]

    CI = 0  → free flow
    CI = 1  → gridlock
    """
    if free_flow_speed <= 0:
        return 0.0
    return float(np.clip(1.0 - current_speed / free_flow_speed, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Route-level objective evaluators (analytical / graph-based)
# ---------------------------------------------------------------------------

def f1_route_travel_time(
    route: List[str],
    graph: dict,
    congestion_index: dict,
    delta_t: float = 5.0,
) -> float:
    """
    f1 – Average travel time (seconds) along a single node-sequence route.

    Uses BPR function applied to each link in the route.  Link volumes are
    estimated from the TraCI congestion index via:
        v = CI * capacity

    Parameters
    ----------
    route           : ordered list of node IDs
    graph           : adjacency dict {u: {v: {length, capacity, free_flow_speed, ...}}}
    congestion_index: {edge_id: CI_value}
    delta_t         : rolling-time-slice width (minutes), default 5 min
    """
    total_time = 0.0
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        attrs = graph.get(u, {}).get(v, {})
        length       = attrs.get("length", 500.0)           # metres
        capacity     = attrs.get("capacity", 1800.0)        # veh/h
        ffspeed      = attrs.get("free_flow_speed", 13.9)   # m/s  (≈ 50 km/h)
        free_flow_t  = length / max(ffspeed, 0.5)           # seconds

        edge_id      = attrs.get("edge_id", f"{u}_{v}")
        ci           = congestion_index.get(edge_id, 0.0)
        volume       = ci * capacity                         # veh/h (estimate)

        link_time    = bpr_travel_time(free_flow_t, volume, capacity)
        total_time  += link_time

    return total_time


def f2_route_congestion(
    route: List[str],
    graph: dict,
    congestion_index: dict,
) -> float:
    """
    f2 – Mean congestion intensity along a route [0, 1].

    Computed as the length-weighted mean of per-link congestion indices.
    """
    total_length = 0.0
    weighted_ci  = 0.0
    for i in range(len(route) - 1):
        u, v     = route[i], route[i + 1]
        attrs    = graph.get(u, {}).get(v, {})
        length   = attrs.get("length", 500.0)
        edge_id  = attrs.get("edge_id", f"{u}_{v}")
        ci       = congestion_index.get(edge_id, 0.0)

        weighted_ci  += ci * length
        total_length += length

    if total_length == 0:
        return 0.0
    return float(np.clip(weighted_ci / total_length, 0.0, 1.0))


def f3_route_delay(
    route: List[str],
    junction_waiting: dict,
) -> float:
    """
    f3 – Total network delay (seconds) along a route.

    Summed as the mean waiting time at each intermediate junction.
    End nodes (origin, destination) are excluded.
    """
    total_delay = 0.0
    for node in route[1:-1]:       # intermediate intersections only
        total_delay += junction_waiting.get(node, 0.0)
    return total_delay


# ---------------------------------------------------------------------------
# Population-level objective computation (called from NSGA-II loop)
# ---------------------------------------------------------------------------

def compute_objectives(
    sumo,                          # SumoInterface instance
    routes: List[List[str]],
    signal_timings: List[float],
    graph: dict = None,
    congestion_index: dict = None,
    junction_waiting: dict = None,
) -> Tuple[float, float, float]:
    """
    Evaluate all three objectives for one chromosome.

    Two modes:
    A) Live TraCI mode  – if `graph` / `congestion_index` / `junction_waiting`
       are None, fetches metrics directly from the running SUMO simulation.
       One simulation mini-step is performed to collect fresh data.

    B) Graph mode       – uses pre-fetched network state dicts.
       Used inside the RHO loop after fetching the state snapshot once per window.

    Returns
    -------
    (f1, f2, f3) : tuple of raw (un-normalised) objective values
    """
    # ── Mode A: live TraCI ──────────────────────────────────────────────
    if graph is None or congestion_index is None or junction_waiting is None:
        return _compute_from_sumo(sumo, routes, signal_timings)

    # ── Mode B: graph-based analytical ──────────────────────────────────
    return _compute_from_graph(routes, graph, congestion_index, junction_waiting)


def _compute_from_sumo(
    sumo,
    routes: List[List[str]],
    signal_timings: List[float],
) -> Tuple[float, float, float]:
    """
    Fetch objective values directly from a live SUMO instance.
    Applies signal timings, steps the simulation, then reads metrics.
    """
    # Apply signal timings to SUMO
    sumo.apply_solution(routes, signal_timings)

    # Advance one 5-minute slice (300 steps at step_length=1s)
    sumo.step(n=300)

    f1 = sumo.get_mean_travel_time()
    f2 = sumo.get_mean_congestion()
    f3 = sumo.get_total_delay()

    return (
        max(0.0, f1),
        float(np.clip(f2, 0.0, 1.0)),
        max(0.0, f3),
    )


def _compute_from_graph(
    routes: List[List[str]],
    graph: dict,
    congestion_index: dict,
    junction_waiting: dict,
) -> Tuple[float, float, float]:
    """
    Compute objectives analytically using pre-fetched network-state dicts.
    Aggregates across all route genes in the chromosome.
    """
    f1_vals, f2_vals, f3_vals = [], [], []

    for route in routes:
        if len(route) < 2:
            continue
        f1_vals.append(f1_route_travel_time(route, graph, congestion_index))
        f2_vals.append(f2_route_congestion(route, graph, congestion_index))
        f3_vals.append(f3_route_delay(route, junction_waiting))

    f1 = float(np.mean(f1_vals)) if f1_vals else 0.0
    f2 = float(np.mean(f2_vals)) if f2_vals else 0.0
    f3 = float(np.sum(f3_vals))  if f3_vals else 0.0

    return max(0.0, f1), float(np.clip(f2, 0.0, 1.0)), max(0.0, f3)


# ---------------------------------------------------------------------------
# Objective normalisation (equation 2 of the paper)
# ---------------------------------------------------------------------------

def normalise_population_objectives(
    population,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply min-max normalisation across the population:

        f_norm = (f - f_min) / (f_max - f_min + eps)

    Parameters
    ----------
    population : list of Individual objects with .objectives attribute
    eps        : small constant to avoid division by zero

    Returns
    -------
    numpy array of shape (N, M) with normalised objective values
    """
    obj_matrix = np.array([ind.objectives for ind in population], dtype=float)
    f_min = obj_matrix.min(axis=0)
    f_max = obj_matrix.max(axis=0)
    normalised = (obj_matrix - f_min) / (f_max - f_min + eps)
    return normalised


# ---------------------------------------------------------------------------
# Pareto front metrics
# ---------------------------------------------------------------------------

def hypervolume_indicator(
    pareto_front,
    reference_point: List[float] = None,
) -> float:
    """
    Compute a 3-D hypervolume indicator for the Pareto front
    (used for convergence analysis in supplementary experiments).

    Uses a simple sweep algorithm suitable for M=3 objectives.
    Reference point defaults to [1.2 × max_f1, 1.2 × max_f2, 1.2 × max_f3].

    Parameters
    ----------
    pareto_front    : list of Individual objects (Rank-1 solutions)
    reference_point : worst-point reference [f1_ref, f2_ref, f3_ref]

    Returns
    -------
    Hypervolume value (higher is better – larger dominated space).
    """
    if not pareto_front:
        return 0.0

    obj_matrix = np.array([ind.objectives for ind in pareto_front], dtype=float)

    if reference_point is None:
        reference_point = (obj_matrix.max(axis=0) * 1.2).tolist()

    ref = np.array(reference_point)

    # Filter out solutions that don't dominate the reference point
    dominated = obj_matrix[np.all(obj_matrix < ref, axis=1)]
    if len(dominated) == 0:
        return 0.0

    # Sort by f1
    dominated = dominated[dominated[:, 0].argsort()]

    # Sweep line (simplified 3-D HV via layer decomposition)
    hv = 0.0
    prev_f1 = ref[0]
    for i in range(len(dominated) - 1, -1, -1):
        point = dominated[i]
        hv += (prev_f1 - point[0]) * (ref[1] - point[1]) * (ref[2] - point[2])
        prev_f1 = point[0]

    return float(max(0.0, hv))


def spacing_metric(pareto_front) -> float:
    """
    Compute the spacing metric S (smaller is better – uniform distribution).

        S = sqrt( (1/(n-1)) * sum_i (d_i - d_mean)^2 )

    where d_i = min Euclidean distance from solution i to all other solutions.
    """
    if len(pareto_front) < 2:
        return 0.0

    obj_matrix = np.array([ind.objectives for ind in pareto_front], dtype=float)
    n = len(obj_matrix)
    d_vals = []
    for i in range(n):
        dists = [
            np.linalg.norm(obj_matrix[i] - obj_matrix[j])
            for j in range(n) if j != i
        ]
        d_vals.append(min(dists))

    d_mean = np.mean(d_vals)
    spacing = math.sqrt(np.mean([(d - d_mean) ** 2 for d in d_vals]))
    return float(spacing)
