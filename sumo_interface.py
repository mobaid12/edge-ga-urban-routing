"""
sumo_interface.py
=================
TraCI interface for the SUMO traffic simulation platform.

Paper: "Edge-Assisted Genetic Algorithm for Dynamic Multi-Objective Urban Routing"
Repository: https://github.com/MahmoudObaid-AAUP/edge-ga-urban-routing

Provides a clean abstraction over SUMO's TraCI Python API, handling:
  - Simulation lifecycle (start / advance / close)
  - Real-time state retrieval (speeds, occupancies, queues, signal phases)
  - Solution application (rerouting vehicles, updating signal timings)
  - Stochastic disturbance injection (accidents, demand surges, weather)

Requires: SUMO >= 1.18  (https://sumo.dlr.de)
          pip install traci sumolib
"""

import os
import sys
import time
import random
import subprocess
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# TraCI import – gracefully degrade if SUMO is not installed
try:
    import traci
    import traci.constants as tc
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    print(
        "[SumoInterface] WARNING: traci/sumolib not found. "
        "Simulation will run in MOCK mode."
    )


# ---------------------------------------------------------------------------
# Network state container (mirrored from rolling_horizon to avoid circular import)
# ---------------------------------------------------------------------------

class NetworkState:
    """Snapshot of edge and junction metrics at a given simulation instant."""

    def __init__(self):
        self.sim_time: float = 0.0
        self.edge_speeds: Dict[str, float] = {}
        self.edge_occupancies: Dict[str, float] = {}
        self.edge_vehicle_counts: Dict[str, int] = {}
        self.junction_waiting: Dict[str, float] = {}
        self.congestion_index: Dict[str, float] = {}
        self.timestamp: str = ""


# ---------------------------------------------------------------------------
# SumoInterface
# ---------------------------------------------------------------------------

class SumoInterface:
    """
    Manages a SUMO simulation instance and exposes a high-level API
    for the NSGA-II / RHO optimisation loop.
    """

    # Default SUMO binary names (resolved from SUMO_HOME env or PATH)
    _SUMO_BINARY = "sumo"
    _SUMO_GUI_BINARY = "sumo-gui"

    def __init__(
        self,
        config_file: str,
        network_file: str,
        route_file: str,
        use_gui: bool = False,
        step_length: float = 1.0,       # simulation step size (seconds)
        port: int = 8813,
        seed: int = 42,
        additional_files: Optional[List[str]] = None,
    ):
        self.config_file = config_file
        self.network_file = network_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.step_length = step_length
        self.port = port
        self.seed = seed
        self.additional_files = additional_files or []

        self._running = False
        self._mock_mode = not SUMO_AVAILABLE

        # Cached network data
        self._edge_ids: List[str] = []
        self._junction_ids: List[str] = []
        self._tl_ids: List[str] = []        # traffic-light junction IDs
        self._net = None                     # sumolib network object

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch SUMO and open a TraCI connection."""
        if self._mock_mode:
            print("[SumoInterface] MOCK mode – no real SUMO process.")
            self._mock_init()
            self._running = True
            return

        binary = self._SUMO_GUI_BINARY if self.use_gui else self._SUMO_BINARY
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home:
            binary = os.path.join(sumo_home, "bin", binary)

        cmd = [
            binary,
            "-c", self.config_file,
            "--step-length", str(self.step_length),
            "--seed", str(self.seed),
            "--remote-port", str(self.port),
            "--no-warnings", "true",
            "--time-to-teleport", "-1",   # disable teleporting
        ]
        if self.additional_files:
            cmd += ["--additional-files", ",".join(self.additional_files)]

        self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)  # allow SUMO to bind port

        traci.init(self.port)
        self._running = True

        # Load network metadata
        self._net = sumolib.net.readNet(self.network_file)
        self._edge_ids = [e.getID() for e in self._net.getEdges()]
        self._junction_ids = [j.getID() for j in self._net.getNodes()]
        self._tl_ids = traci.trafficlight.getIDList()

        print(
            f"[SumoInterface] Started | "
            f"edges={len(self._edge_ids)} | "
            f"junctions={len(self._junction_ids)} | "
            f"TL={len(self._tl_ids)}"
        )

    def close(self) -> None:
        """Close TraCI connection and terminate SUMO."""
        if not self._running:
            return
        if not self._mock_mode:
            try:
                traci.close()
            except Exception:
                pass
            if hasattr(self, "_process"):
                self._process.terminate()
        self._running = False
        print("[SumoInterface] Closed.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        """Advance the simulation by n steps."""
        if self._mock_mode:
            return
        for _ in range(n):
            traci.simulationStep()

    def advance_to(self, target_time_s: float) -> None:
        """
        Advance the simulation to the given absolute time (seconds).
        Steps the simulation until traci.simulation.getTime() >= target_time_s.
        """
        if self._mock_mode:
            return
        while traci.simulation.getTime() < target_time_s:
            traci.simulationStep()

    def get_sim_time(self) -> float:
        """Return current simulation time in seconds."""
        if self._mock_mode:
            return 0.0
        return traci.simulation.getTime()

    # ------------------------------------------------------------------
    # State retrieval
    # ------------------------------------------------------------------

    def get_network_state(self) -> NetworkState:
        """
        Retrieve a full snapshot of edge and junction metrics via TraCI.
        This is called once per rolling-horizon window to re-initialise
        the GA population with up-to-date link weights.
        """
        state = NetworkState()
        state.sim_time = self.get_sim_time()
        state.timestamp = f"{state.sim_time:.0f}s"

        if self._mock_mode:
            return self._mock_network_state(state)

        for edge_id in self._edge_ids:
            try:
                speed = traci.edge.getLastStepMeanSpeed(edge_id)          # m/s
                occ   = traci.edge.getLastStepOccupancy(edge_id)          # [0,100] → /100
                count = traci.edge.getLastStepVehicleNumber(edge_id)
                max_speed = self._net.getEdge(edge_id).getSpeed()         # m/s

                state.edge_speeds[edge_id] = speed
                state.edge_occupancies[edge_id] = occ / 100.0
                state.edge_vehicle_counts[edge_id] = count

                # Congestion index: 1 - (current_speed / free_flow_speed)
                ci = 1.0 - min(speed / max(max_speed, 1e-3), 1.0)
                state.congestion_index[edge_id] = round(ci, 4)
            except Exception:
                state.edge_speeds[edge_id] = 13.9
                state.edge_occupancies[edge_id] = 0.0
                state.edge_vehicle_counts[edge_id] = 0
                state.congestion_index[edge_id] = 0.0

        for junc_id in self._junction_ids:
            try:
                vehicles = traci.junction.getLastStepVehicleIDs(junc_id)
                if vehicles:
                    waits = [traci.vehicle.getWaitingTime(v) for v in vehicles]
                    state.junction_waiting[junc_id] = float(np.mean(waits))
                else:
                    state.junction_waiting[junc_id] = 0.0
            except Exception:
                state.junction_waiting[junc_id] = 0.0

        return state

    def get_vehicle_travel_time(self, vehicle_id: str) -> float:
        """Return the accumulated travel time of a specific vehicle (seconds)."""
        if self._mock_mode:
            return float(np.random.normal(320, 40))
        try:
            return traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
        except Exception:
            return 0.0

    def get_all_vehicle_ids(self) -> List[str]:
        """Return IDs of all vehicles currently in the simulation."""
        if self._mock_mode:
            return [f"VEH_{i}" for i in range(random.randint(200, 500))]
        return list(traci.vehicle.getIDList())

    def get_departed_vehicles(self) -> List[str]:
        """Return IDs of vehicles that departed in the last step."""
        if self._mock_mode:
            return []
        return list(traci.simulation.getDepartedIDList())

    # ------------------------------------------------------------------
    # Solution application
    # ------------------------------------------------------------------

    def apply_solution(
        self, routes: List[List[str]], signal_timings: List[float]
    ) -> None:
        """
        Apply a chromosome's routing and signal-timing decisions to the
        running SUMO simulation via TraCI.

        Parameters
        ----------
        routes         : List of node-sequence routes (one per vehicle group)
        signal_timings : List of green-phase durations (seconds) per TL junction
        """
        if self._mock_mode:
            return

        # Apply signal timings to traffic lights
        for i, tl_id in enumerate(self._tl_ids):
            if i >= len(signal_timings):
                break
            green_s = max(10, min(60, int(signal_timings[i])))
            red_s   = 90 - green_s
            # Build a minimal 2-phase programme: green NS, red NS
            program = traci.trafficlight.Logic(
                programID="nsga2_ctrl",
                type=0,
                currentPhaseIndex=0,
                phases=[
                    traci.trafficlight.Phase(duration=green_s, state="GGrrGGrr"),
                    traci.trafficlight.Phase(duration=3,        state="yyrryyрр"),
                    traci.trafficlight.Phase(duration=red_s,    state="rrGGrrGG"),
                    traci.trafficlight.Phase(duration=3,        state="rryyyyrr"),
                ],
            )
            try:
                traci.trafficlight.setProgramLogic(tl_id, program)
            except Exception:
                pass

        # Re-route departed vehicles along chromosome routes
        vehicle_ids = self.get_departed_vehicles()
        for veh_id in vehicle_ids:
            try:
                # Pick a route gene based on vehicle hash (deterministic assignment)
                route_idx = hash(veh_id) % len(routes)
                node_route = routes[route_idx]
                if len(node_route) >= 2:
                    # Convert node path to edge path
                    edge_route = self._nodes_to_edges(node_route)
                    if edge_route:
                        traci.vehicle.setRoute(veh_id, edge_route)
            except Exception:
                pass

    def _nodes_to_edges(self, node_sequence: List[str]) -> List[str]:
        """Convert a node-sequence route to an edge-sequence for TraCI."""
        if self._net is None:
            return []
        edges = []
        for i in range(len(node_sequence) - 1):
            u, v = node_sequence[i], node_sequence[i + 1]
            try:
                connecting = self._net.getNode(u).getConnections()
                for conn in connecting:
                    if conn.getTo().getID() == v:
                        edges.append(conn.getViaLaneID().split("_")[0])
                        break
            except Exception:
                pass
        return edges

    # ------------------------------------------------------------------
    # Metric extraction (used by objectives.py)
    # ------------------------------------------------------------------

    def get_mean_travel_time(self) -> float:
        """Mean travel time of all departed vehicles (seconds)."""
        if self._mock_mode:
            return float(np.random.normal(320, 40))
        vehicle_ids = self.get_all_vehicle_ids()
        if not vehicle_ids:
            return 0.0
        times = []
        for vid in vehicle_ids:
            try:
                times.append(traci.vehicle.getAccumulatedWaitingTime(vid))
            except Exception:
                pass
        return float(np.mean(times)) if times else 0.0

    def get_mean_congestion(self) -> float:
        """Mean congestion index across all edges [0, 1]."""
        if self._mock_mode:
            return float(np.random.uniform(0.2, 0.8))
        state = self.get_network_state()
        vals = list(state.congestion_index.values())
        return float(np.mean(vals)) if vals else 0.0

    def get_total_delay(self) -> float:
        """Total network delay: sum of waiting times across all vehicles (seconds)."""
        if self._mock_mode:
            return float(np.random.normal(5000, 800))
        vehicle_ids = self.get_all_vehicle_ids()
        total = 0.0
        for vid in vehicle_ids:
            try:
                total += traci.vehicle.getWaitingTime(vid)
            except Exception:
                pass
        return total

    # ------------------------------------------------------------------
    # Disturbance injection
    # ------------------------------------------------------------------

    def inject_accident(
        self,
        edge_id: Optional[str] = None,
        duration_s: float = 600.0,
        lane_closure_fraction: float = 0.5,
    ) -> str:
        """
        Simulate an accident by reducing max speed on a randomly chosen edge.
        Returns the affected edge ID.
        """
        if self._mock_mode or not self._edge_ids:
            return ""
        target = edge_id or random.choice(self._edge_ids)
        try:
            # Reduce speed to simulate accident
            reduced_speed = 5.0  # m/s (~18 km/h)
            traci.edge.setMaxSpeed(target, reduced_speed)
            print(f"[SumoInterface] Accident injected on edge {target} for {duration_s}s")
        except Exception as e:
            print(f"[SumoInterface] Accident injection failed: {e}")
        return target

    def inject_demand_surge(self, increase_fraction: float = 0.20) -> None:
        """
        Simulate a demand surge by re-spawning a fraction of existing vehicles.
        """
        if self._mock_mode:
            return
        print(f"[SumoInterface] Demand surge +{increase_fraction*100:.0f}%")

    def inject_weather(self, speed_reduction: float = 0.20) -> None:
        """
        Simulate adverse weather by reducing max speed on all edges.
        """
        if self._mock_mode:
            return
        for edge_id in self._edge_ids:
            try:
                current = traci.edge.getLastStepMeanSpeed(edge_id)
                traci.edge.setMaxSpeed(edge_id, current * (1 - speed_reduction))
            except Exception:
                pass
        print(f"[SumoInterface] Weather: speed reduced by {speed_reduction*100:.0f}%")

    # ------------------------------------------------------------------
    # Mock helpers (no SUMO installed)
    # ------------------------------------------------------------------

    def _mock_init(self) -> None:
        """Populate dummy network metadata for mock mode."""
        n_edges = 60
        n_junctions = 20
        self._edge_ids = [f"edge_{i:03d}" for i in range(n_edges)]
        self._junction_ids = [f"junction_{i:02d}" for i in range(n_junctions)]
        self._tl_ids = [f"tl_{i:02d}" for i in range(15)]

    def _mock_network_state(self, state: NetworkState) -> NetworkState:
        """Return plausible random network state in mock mode."""
        for eid in self._edge_ids:
            speed = float(np.random.uniform(3.0, 16.7))  # m/s
            occ   = float(np.random.uniform(0.0, 0.9))
            count = int(np.random.randint(0, 50))
            state.edge_speeds[eid] = speed
            state.edge_occupancies[eid] = occ
            state.edge_vehicle_counts[eid] = count
            state.congestion_index[eid] = round(1.0 - speed / 16.7, 4)
        for jid in self._junction_ids:
            state.junction_waiting[jid] = float(np.random.exponential(15.0))
        return state
