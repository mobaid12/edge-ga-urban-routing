"""
nsga2_routing.py
================
NSGA-II Multi-Objective Genetic Algorithm for Urban Traffic Routing.

Paper: "Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing"
Authors: Odeh, Obaid, Lasari, Alrajab, Khattab, Qarariyah, Ziou
Repository: https://github.com/MahmoudObaid-AAUP/edge-ga-urban-routing

Objectives (minimise simultaneously):
  f1 – Average travel time (seconds)
  f2 – Congestion intensity (vehicle density per link, normalised [0,1])
  f3 – Total network delay (seconds)

Usage:
  python nsga2_routing.py \
      --network sumo_scenarios/bethlehem_network.net.xml \
      --routes  sumo_scenarios/routes_baseline.rou.xml \
      --config  sumo_scenarios/SCN_001_baseline.sumocfg \
      --pop_size 100 --generations 200 --horizon 15
"""

import argparse
import copy
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from objectives import compute_objectives
from sumo_interface import SumoInterface
from utils import normalise_objectives, build_route_graph

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """A candidate solution (chromosome) in the NSGA-II population."""
    routes: List[List[str]]          # List of node-sequence routes per vehicle class
    signal_timings: List[float]      # Green-phase durations per intersection (seconds)
    objectives: List[float] = field(default_factory=lambda: [float("inf")] * 3)
    rank: int = 0
    crowding_distance: float = 0.0

    def clone(self) -> "Individual":
        return Individual(
            routes=copy.deepcopy(self.routes),
            signal_timings=copy.deepcopy(self.signal_timings),
        )


# ---------------------------------------------------------------------------
# NSGA-II core operators
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    """
    Fast non-dominated sorting (Deb et al., 2002).
    Returns a list of Pareto fronts, each front being a list of individual indices.
    Complexity: O(M * N^2), M = number of objectives, N = population size.
    """
    n = len(population)
    domination_count = [0] * n       # number of solutions that dominate individual i
    dominated_by: List[List[int]] = [[] for _ in range(n)]  # solutions dominated by i
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(population[i], population[j]):
                dominated_by[i].append(j)
            elif _dominates(population[j], population[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            population[i].rank = 1
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front: List[int] = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 2
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _dominates(a: Individual, b: Individual) -> bool:
    """Return True if individual a Pareto-dominates individual b."""
    better_in_at_least_one = False
    for oa, ob in zip(a.objectives, b.objectives):
        if oa > ob:
            return False
        if oa < ob:
            better_in_at_least_one = True
    return better_in_at_least_one


def crowding_distance_assignment(
    front: List[int], population: List[Individual], num_objectives: int = 3
) -> None:
    """Assign crowding distance to each individual in a Pareto front."""
    n = len(front)
    if n == 0:
        return
    for idx in front:
        population[idx].crowding_distance = 0.0

    for m in range(num_objectives):
        front.sort(key=lambda i: population[i].objectives[m])
        population[front[0]].crowding_distance = float("inf")
        population[front[-1]].crowding_distance = float("inf")
        obj_range = (
            population[front[-1]].objectives[m] - population[front[0]].objectives[m]
        )
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            population[front[k]].crowding_distance += (
                population[front[k + 1]].objectives[m]
                - population[front[k - 1]].objectives[m]
            ) / obj_range


def tournament_selection(
    population: List[Individual], tournament_size: int = 2
) -> Individual:
    """Binary tournament selection based on rank and crowding distance."""
    candidates = random.sample(population, min(tournament_size, len(population)))
    candidates.sort(
        key=lambda ind: (ind.rank, -ind.crowding_distance)
    )
    return candidates[0]


def crossover(
    parent1: Individual, parent2: Individual, cx_prob: float = 0.9
) -> Tuple[Individual, Individual]:
    """
    Single-point crossover on signal timings;
    uniform crossover on route gene segments.
    """
    child1, child2 = parent1.clone(), parent2.clone()
    if random.random() < cx_prob:
        # Signal timing crossover (single-point)
        n = len(child1.signal_timings)
        if n > 1:
            point = random.randint(1, n - 1)
            child1.signal_timings[point:], child2.signal_timings[point:] = (
                child2.signal_timings[point:],
                child1.signal_timings[point:],
            )
        # Route crossover (uniform)
        for r in range(min(len(child1.routes), len(child2.routes))):
            if random.random() < 0.5:
                child1.routes[r], child2.routes[r] = child2.routes[r], child1.routes[r]
    return child1, child2


def mutate(
    individual: Individual,
    graph,
    mut_prob: float = 0.1,
    timing_sigma: float = 5.0,
    green_min: float = 10.0,
    green_max: float = 60.0,
) -> Individual:
    """
    Gaussian mutation on signal timings;
    random re-routing on a randomly selected route gene.
    """
    child = individual.clone()

    # Mutate signal timings
    for i in range(len(child.signal_timings)):
        if random.random() < mut_prob:
            child.signal_timings[i] = float(
                np.clip(
                    child.signal_timings[i] + np.random.normal(0, timing_sigma),
                    green_min,
                    green_max,
                )
            )

    # Mutate one route gene
    if child.routes and random.random() < mut_prob:
        idx = random.randint(0, len(child.routes) - 1)
        route = child.routes[idx]
        if len(route) >= 2:
            new_route = _random_route(graph, route[0], route[-1])
            if new_route:
                child.routes[idx] = new_route

    return child


def _random_route(graph, origin: str, destination: str, max_hops: int = 20) -> List[str]:
    """Generate a random feasible route from origin to destination via DFS with backtracking."""
    stack = [(origin, [origin])]
    visited: set = set()
    while stack:
        node, path = stack.pop()
        if node == destination:
            return path
        if len(path) > max_hops or node in visited:
            continue
        visited.add(node)
        neighbours = list(graph.get(node, {}).keys())
        random.shuffle(neighbours)
        for nb in neighbours:
            stack.append((nb, path + [nb]))
    return []


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def initialise_population(
    pop_size: int,
    graph,
    od_pairs: List[Tuple[str, str]],
    num_intersections: int,
    green_min: float = 10.0,
    green_max: float = 60.0,
) -> List[Individual]:
    """Create an initial population of feasible individuals."""
    population: List[Individual] = []
    for _ in range(pop_size):
        routes = []
        for origin, dest in od_pairs:
            route = _random_route(graph, origin, dest)
            routes.append(route if route else [origin, dest])
        signal_timings = [
            random.uniform(green_min, green_max) for _ in range(num_intersections)
        ]
        population.append(Individual(routes=routes, signal_timings=signal_timings))
    return population


# ---------------------------------------------------------------------------
# NSGA-II main loop
# ---------------------------------------------------------------------------

def nsga2(
    sumo: SumoInterface,
    graph,
    od_pairs: List[Tuple[str, str]],
    num_intersections: int,
    pop_size: int = 100,
    num_generations: int = 200,
    cx_prob: float = 0.9,
    mut_prob: float = 0.1,
    verbose: bool = True,
) -> Tuple[List[Individual], List[List[float]]]:
    """
    Run NSGA-II optimisation.

    Returns
    -------
    pareto_front : list of non-dominated Individual objects (Rank-1 solutions)
    history      : per-generation list of [mean_f1, mean_f2, mean_f3]
    """
    population = initialise_population(pop_size, graph, od_pairs, num_intersections)

    # Evaluate initial population
    for ind in population:
        raw = compute_objectives(sumo, ind.routes, ind.signal_timings)
        ind.objectives = list(raw)

    history: List[List[float]] = []

    for gen in range(1, num_generations + 1):
        t0 = time.perf_counter()

        # --- Offspring generation ---
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = crossover(p1, p2, cx_prob)
            c1 = mutate(c1, graph, mut_prob)
            c2 = mutate(c2, graph, mut_prob)
            offspring.extend([c1, c2])
        offspring = offspring[:pop_size]

        # Evaluate offspring
        for ind in offspring:
            raw = compute_objectives(sumo, ind.routes, ind.signal_timings)
            ind.objectives = list(raw)

        # --- Environmental selection ---
        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        for front in fronts:
            crowding_distance_assignment(front, combined)

        new_population: List[Individual] = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                new_population.extend([combined[i] for i in front])
            else:
                remaining = pop_size - len(new_population)
                front.sort(
                    key=lambda i: -combined[i].crowding_distance
                )
                new_population.extend([combined[i] for i in front[:remaining]])
                break

        population = new_population

        # Logging
        obj_array = np.array([ind.objectives for ind in population])
        mean_obj = obj_array.mean(axis=0).tolist()
        history.append(mean_obj)
        elapsed = time.perf_counter() - t0
        if verbose and (gen % 10 == 0 or gen == 1):
            print(
                f"Gen {gen:4d}/{num_generations} | "
                f"f1={mean_obj[0]:.1f}s  f2={mean_obj[1]:.4f}  f3={mean_obj[2]:.1f}s | "
                f"time={elapsed:.2f}s"
            )

    # Extract Pareto front (rank-1 solutions)
    fronts = fast_non_dominated_sort(population)
    pareto_front = [population[i] for i in fronts[0]] if fronts else population

    return pareto_front, history


# ---------------------------------------------------------------------------
# Knee-point selection (compromise solution)
# ---------------------------------------------------------------------------

def select_knee_point(pareto_front: List[Individual]) -> Individual:
    """
    Select the compromise (knee-point) solution from the Pareto front using
    the minimum distance to the ideal point after min-max normalisation.
    """
    if len(pareto_front) == 1:
        return pareto_front[0]

    obj_matrix = np.array([ind.objectives for ind in pareto_front])
    norm = normalise_objectives(obj_matrix)

    # Ideal point is origin [0,0,0] in normalised space
    distances = np.linalg.norm(norm, axis=1)
    knee_idx = int(np.argmin(distances))
    return pareto_front[knee_idx]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSGA-II Urban Routing Optimiser")
    p.add_argument("--network",     required=True, help="SUMO .net.xml file")
    p.add_argument("--routes",      required=True, help="SUMO .rou.xml file")
    p.add_argument("--config",      required=True, help="SUMO .sumocfg file")
    p.add_argument("--pop_size",    type=int,   default=100)
    p.add_argument("--generations", type=int,   default=200)
    p.add_argument("--horizon",     type=int,   default=15,
                   help="Rolling-horizon window (minutes)")
    p.add_argument("--cx_prob",     type=float, default=0.9)
    p.add_argument("--mut_prob",    type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--output",      default="results/",
                   help="Directory to write Pareto front CSV and plots")
    p.add_argument("--gui",         action="store_true",
                   help="Launch SUMO-GUI instead of sumo (headless)")
    return p.parse_args()


def main() -> None:
    import os
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  NSGA-II Urban Routing Optimiser")
    print(f"  Network : {args.network}")
    print(f"  Pop size: {args.pop_size}  |  Generations: {args.generations}")
    print(f"  Horizon : {args.horizon} min  |  Seed: {args.seed}")
    print("=" * 60)

    sumo = SumoInterface(
        config_file=args.config,
        network_file=args.network,
        route_file=args.routes,
        use_gui=args.gui,
    )
    sumo.start()

    graph, od_pairs, num_intersections = build_route_graph(args.network)

    pareto_front, history = nsga2(
        sumo=sumo,
        graph=graph,
        od_pairs=od_pairs,
        num_intersections=num_intersections,
        pop_size=args.pop_size,
        num_generations=args.generations,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        verbose=True,
    )

    sumo.close()

    # Save Pareto front
    import csv
    out_csv = os.path.join(args.output, "pareto_front.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution_id", "f1_travel_time_s", "f2_congestion", "f3_delay_s"])
        for i, ind in enumerate(pareto_front):
            writer.writerow([i + 1] + [f"{v:.4f}" for v in ind.objectives])
    print(f"\nPareto front ({len(pareto_front)} solutions) saved → {out_csv}")

    knee = select_knee_point(pareto_front)
    print(
        f"Knee-point solution: f1={knee.objectives[0]:.1f}s  "
        f"f2={knee.objectives[1]:.4f}  f3={knee.objectives[2]:.1f}s"
    )

    # Save convergence history
    hist_csv = os.path.join(args.output, "convergence_history.csv")
    with open(hist_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_f1", "mean_f2", "mean_f3"])
        for gen, row in enumerate(history, 1):
            writer.writerow([gen] + [f"{v:.4f}" for v in row])
    print(f"Convergence history saved → {hist_csv}")


if __name__ == "__main__":
    main()
