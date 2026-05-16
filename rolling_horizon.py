"""
rolling_horizon.py
==================
Rolling-Horizon Optimization (RHO) wrapper for NSGA-II urban routing.

Paper: "Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing"
Repository: https://github.com/MahmoudObaid-AAUP/edge-ga-urban-routing

The horizon is divided into overlapping windows of length H (minutes).
At each window:
  1. Fetch the current network state from SUMO via TraCI.
  2. Re-initialise the GA population (warm-started from the previous Pareto front).
  3. Run NSGA-II for a fixed budget of generations.
  4. Apply the knee-point solution to SUMO for the next window.
  5. Slide the window forward by step_size minutes.

This ensures near-real-time adaptation to dynamic congestion events
(accidents, demand surges, weather) as described in Section III-C of the paper.
"""

import time
import csv
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from nsga2_routing import (
    Individual,
    nsga2,
    select_knee_point,
    fast_non_dominated_sort,
    crowding_distance_assignment,
    initialise_population,
)
from sumo_interface import SumoInterface
from objectives import compute_objectives
from utils import build_route_graph, normalise_objectives


# ---------------------------------------------------------------------------
# RHO Configuration
# ---------------------------------------------------------------------------

@dataclass
class RHOConfig:
    """Configuration parameters for Rolling-Horizon Optimization."""
    horizon_minutes: int = 15        # Length of each optimisation window (Δt)
    step_minutes: int = 5            # Slide step between consecutive windows
    pop_size: int = 100              # NSGA-II population size per window
    generations_per_window: int = 50 # GA budget per window (reduced vs. full run)
    warm_start_ratio: float = 0.5    # Fraction of population seeded from previous Pareto front
    cx_prob: float = 0.9
    mut_prob: float = 0.1
    total_sim_minutes: int = 60      # Total simulation horizon
    output_dir: str = "results/"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Network state snapshot
# ---------------------------------------------------------------------------

@dataclass
class NetworkState:
    """Snapshot of the SUMO network state at a given simulation step."""
    sim_time: float                         # Current simulation time (seconds)
    edge_speeds: Dict[str, float]           # edge_id → mean speed (m/s)
    edge_occupancies: Dict[str, float]      # edge_id → occupancy [0,1]
    edge_vehicle_counts: Dict[str, int]     # edge_id → vehicle count
    junction_waiting: Dict[str, float]      # junction_id → mean waiting time (s)
    congestion_index: Dict[str, float]      # edge_id → congestion index [0,1]
    timestamp: str = ""

    @classmethod
    def from_sumo(cls, sumo: SumoInterface) -> "NetworkState":
        """Fetch the current network state from a live SUMO simulation."""
        return sumo.get_network_state()


# ---------------------------------------------------------------------------
# Link weight update
# ---------------------------------------------------------------------------

def update_link_weights(graph: Dict, state: NetworkState) -> Dict:
    """
    Update edge travel-time weights in the route graph based on the
    current network state.  Uses the BPR (Bureau of Public Roads) function:

        t = t0 * (1 + alpha * (v / c)^beta)

    where v = current volume, c = capacity, alpha=0.15, beta=4.
    Falls back to speed-based estimate when capacity is unknown.
    """
    alpha, beta = 0.15, 4.0
    updated_graph = {}
    for u, neighbours in graph.items():
        updated_graph[u] = {}
        for v, attrs in neighbours.items():
            edge_id = attrs.get("edge_id", f"{u}_{v}")
            t0 = attrs.get("free_flow_time", 30.0)
            capacity = attrs.get("capacity", 1800.0)
            volume = state.edge_vehicle_counts.get(edge_id, 0) * 12  # scale 5-min to hourly
            congestion = state.congestion_index.get(edge_id, 0.0)

            # BPR formula
            t_bpr = t0 * (1.0 + alpha * (volume / max(capacity, 1)) ** beta)

            # Speed-based fallback
            speed = state.edge_speeds.get(edge_id, attrs.get("free_flow_speed", 13.9))
            length = attrs.get("length", 500.0)
            t_speed = length / max(speed, 0.5)

            # Weighted blend (favour BPR when congestion is high)
            weight = congestion * t_bpr + (1 - congestion) * t_speed

            updated_graph[u][v] = {**attrs, "weight": round(weight, 3)}
    return updated_graph


# ---------------------------------------------------------------------------
# Warm-start initialisation
# ---------------------------------------------------------------------------

def warm_start_population(
    previous_pareto: List[Individual],
    graph: Dict,
    od_pairs: List[Tuple[str, str]],
    num_intersections: int,
    pop_size: int,
    warm_ratio: float = 0.5,
) -> List[Individual]:
    """
    Initialise a new population by seeding with individuals from the
    previous window's Pareto front (warm start) and filling the remainder
    with random individuals.
    """
    warm_count = min(int(pop_size * warm_ratio), len(previous_pareto))
    import copy
    population = [copy.deepcopy(random.choice(previous_pareto)) for _ in range(warm_count)]

    # Random fill
    random_fill = initialise_population(
        pop_size - warm_count, graph, od_pairs, num_intersections
    )
    population.extend(random_fill)
    random.shuffle(population)
    return population


# ---------------------------------------------------------------------------
# RHO main controller
# ---------------------------------------------------------------------------

class RollingHorizonOptimiser:
    """
    Orchestrates repeated NSGA-II runs over sliding time windows,
    re-optimising routing and signal timing as the simulation progresses.
    """

    def __init__(self, sumo: SumoInterface, config: RHOConfig):
        self.sumo = sumo
        self.cfg = config
        self.graph: Dict = {}
        self.od_pairs: List[Tuple[str, str]] = []
        self.num_intersections: int = 0
        self.previous_pareto: Optional[List[Individual]] = None

        # Result logs
        self.window_results: List[Dict[str, Any]] = []

        os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def initialise(self, network_file: str) -> None:
        """Build the route graph from the SUMO network file."""
        self.graph, self.od_pairs, self.num_intersections = build_route_graph(network_file)
        if self.cfg.verbose:
            print(
                f"[RHO] Graph: {len(self.graph)} nodes | "
                f"{len(self.od_pairs)} O-D pairs | "
                f"{self.num_intersections} intersections"
            )

    # ------------------------------------------------------------------
    def _run_window(self, window_id: int, sim_time_s: float) -> Individual:
        """Run one optimisation window and return the selected knee-point solution."""
        if self.cfg.verbose:
            print(
                f"\n[RHO] Window {window_id:03d} | "
                f"sim_time={sim_time_s/60:.1f} min | "
                f"horizon={self.cfg.horizon_minutes} min"
            )

        # 1. Fetch current network state and update link weights
        state = NetworkState.from_sumo(self.sumo)
        updated_graph = update_link_weights(self.graph, state)

        # 2. Initialise population (warm-start if previous Pareto front exists)
        if self.previous_pareto:
            population = warm_start_population(
                self.previous_pareto,
                updated_graph,
                self.od_pairs,
                self.num_intersections,
                self.cfg.pop_size,
                self.cfg.warm_start_ratio,
            )
        else:
            population = initialise_population(
                self.cfg.pop_size,
                updated_graph,
                self.od_pairs,
                self.num_intersections,
            )

        # 3. Evaluate initial population with current link weights
        for ind in population:
            raw = compute_objectives(self.sumo, ind.routes, ind.signal_timings)
            ind.objectives = list(raw)

        # 4. Run NSGA-II for reduced generation budget
        pareto_front, history = nsga2(
            sumo=self.sumo,
            graph=updated_graph,
            od_pairs=self.od_pairs,
            num_intersections=self.num_intersections,
            pop_size=self.cfg.pop_size,
            num_generations=self.cfg.generations_per_window,
            cx_prob=self.cfg.cx_prob,
            mut_prob=self.cfg.mut_prob,
            verbose=self.cfg.verbose,
        )

        self.previous_pareto = pareto_front

        # 5. Select compromise solution
        knee = select_knee_point(pareto_front)

        # Log window results
        mean_objs = np.mean([ind.objectives for ind in pareto_front], axis=0).tolist()
        self.window_results.append({
            "window_id": window_id,
            "sim_time_min": round(sim_time_s / 60, 2),
            "pareto_size": len(pareto_front),
            "knee_f1_travel_time_s": round(knee.objectives[0], 2),
            "knee_f2_congestion": round(knee.objectives[1], 4),
            "knee_f3_delay_s": round(knee.objectives[2], 2),
            "mean_f1": round(mean_objs[0], 2),
            "mean_f2": round(mean_objs[1], 4),
            "mean_f3": round(mean_objs[2], 2),
            "congestion_events": len([
                v for v in state.congestion_index.values() if v > 0.7
            ]),
        })

        if self.cfg.verbose:
            print(
                f"[RHO] Window {window_id:03d} knee: "
                f"f1={knee.objectives[0]:.1f}s  "
                f"f2={knee.objectives[1]:.4f}  "
                f"f3={knee.objectives[2]:.1f}s  "
                f"| Pareto size={len(pareto_front)}"
            )

        return knee

    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the full rolling-horizon loop over the simulation period.
        Returns a list of per-window result dictionaries.
        """
        total_seconds = self.cfg.total_sim_minutes * 60
        horizon_seconds = self.cfg.horizon_minutes * 60
        step_seconds = self.cfg.step_minutes * 60

        if self.cfg.verbose:
            print(
                f"\n{'='*60}\n"
                f"  Rolling-Horizon Optimisation\n"
                f"  Total: {self.cfg.total_sim_minutes} min  |  "
                f"Window: {self.cfg.horizon_minutes} min  |  "
                f"Step: {self.cfg.step_minutes} min\n"
                f"{'='*60}"
            )

        window_id = 1
        sim_time = 0.0

        while sim_time + horizon_seconds <= total_seconds:
            t_start = time.perf_counter()

            # Advance SUMO to current sim_time
            self.sumo.advance_to(sim_time)

            # Run optimisation window
            knee_solution = self._run_window(window_id, sim_time)

            # Apply solution to SUMO for the next step
            self.sumo.apply_solution(knee_solution.routes, knee_solution.signal_timings)

            elapsed = time.perf_counter() - t_start
            if self.cfg.verbose:
                print(f"[RHO] Window {window_id:03d} wall-clock: {elapsed:.1f}s")

            sim_time += step_seconds
            window_id += 1

        # Save results
        self._save_results()
        return self.window_results

    # ------------------------------------------------------------------
    def _save_results(self) -> None:
        """Persist window-by-window results to CSV."""
        out_path = os.path.join(self.cfg.output_dir, "rho_window_results.csv")
        if not self.window_results:
            return
        fieldnames = list(self.window_results[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.window_results)
        print(f"\n[RHO] Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_rho(
    sumo: SumoInterface,
    network_file: str,
    config: Optional[RHOConfig] = None,
) -> List[Dict[str, Any]]:
    """
    High-level entry point: initialise and run the Rolling-Horizon Optimiser.

    Parameters
    ----------
    sumo         : Active SumoInterface instance
    network_file : Path to the SUMO .net.xml network file
    config       : RHOConfig (uses defaults if None)

    Returns
    -------
    List of per-window result dicts
    """
    if config is None:
        config = RHOConfig()
    rho = RollingHorizonOptimiser(sumo, config)
    rho.initialise(network_file)
    return rho.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rolling-Horizon NSGA-II Optimiser")
    parser.add_argument("--network",      required=True)
    parser.add_argument("--config",       required=True, help="SUMO .sumocfg")
    parser.add_argument("--routes",       required=True)
    parser.add_argument("--horizon",      type=int, default=15)
    parser.add_argument("--step",         type=int, default=5)
    parser.add_argument("--total",        type=int, default=60)
    parser.add_argument("--pop_size",     type=int, default=100)
    parser.add_argument("--generations",  type=int, default=50)
    parser.add_argument("--output",       default="results/")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sumo_instance = SumoInterface(
        config_file=args.config,
        network_file=args.network,
        route_file=args.routes,
    )
    sumo_instance.start()

    cfg = RHOConfig(
        horizon_minutes=args.horizon,
        step_minutes=args.step,
        total_sim_minutes=args.total,
        pop_size=args.pop_size,
        generations_per_window=args.generations,
        output_dir=args.output,
    )

    results = run_rho(sumo_instance, args.network, cfg)
    sumo_instance.close()

    print(f"\nCompleted {len(results)} optimisation windows.")
