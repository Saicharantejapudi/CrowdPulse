"""
CrowdPulse: Multi-Agent Spatial Coordination Broker (src/agent_coordinator.py)
Implements autonomous spatial agents for surveillance grid monitoring:
  - ZoneAgent: Represents an individual zone tracking count, risk tier, and status.
  - AgentCoordinator: Brokers inter-zone messaging across a spatial adjacency graph,
    detects multi-zone coordinated bottlenecks, and generates dynamic rerouting directives.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ZoneAgent:
    """
    Autonomous agent responsible for monitoring a single spatial zone patch.
    """
    zone_id: str
    row: int
    col: int
    current_count: float = 0.0
    risk_tier: str = "Green"
    alert_status: str = "NORMAL"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        estimated_count: float,
        green_cutoff: float = 10.0,
        red_cutoff: float = 25.0,
    ) -> str:
        """
        Updates zone count and recalculates risk tier and alert status.
        """
        self.current_count = max(0.0, float(estimated_count))

        if self.current_count < green_cutoff:
            self.risk_tier = "Green"
            self.alert_status = "NORMAL"
        elif self.current_count < red_cutoff:
            self.risk_tier = "Yellow"
            self.alert_status = "CAUTION"
        else:
            self.risk_tier = "Red"
            self.alert_status = "ALERT"

        self.history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "count": self.current_count,
            "tier": self.risk_tier,
        })
        # Keep recent history
        if len(self.history) > 50:
            self.history = self.history[-50:]

        return self.risk_tier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "row": self.row,
            "col": self.col,
            "current_count": round(self.current_count, 1),
            "risk_tier": self.risk_tier,
            "alert_status": self.alert_status,
        }


class AgentCoordinator:
    """
    Broker and orchestrator for spatial grid multi-agent system.
    Models zone adjacency graph, listens for agent telemetry, identifies
    localized vs. coordinated bottlenecks, and issues dynamic rerouting directives.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (3, 3),
        allow_diagonal_adjacency: bool = True,
        green_cutoff: float = 10.0,
        red_cutoff: float = 25.0,
    ) -> None:
        self.grid_size = grid_size
        self.n_rows, self.n_cols = grid_size
        self.allow_diagonal = allow_diagonal_adjacency
        self.green_cutoff = green_cutoff
        self.red_cutoff = red_cutoff

        # Initialize agents
        self.agents: Dict[str, ZoneAgent] = {}
        self._init_agents()

        # Build adjacency graph
        self.adjacency_graph: Dict[str, List[str]] = {}
        self._build_adjacency_graph()

        # Message broker log
        self.message_log: List[Dict[str, str]] = []

    def _init_agents(self) -> None:
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                zid = f"Zone_{r}_{c}"
                self.agents[zid] = ZoneAgent(zone_id=zid, row=r, col=c)

    def _build_adjacency_graph(self) -> None:
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                zid = f"Zone_{r}_{c}"
                neighbors = []
                # Orthogonal
                orthogonal_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                # Diagonal
                diagonal_steps = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                steps = orthogonal_steps + (diagonal_steps if self.allow_diagonal else [])

                for dr, dc in steps:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                        neighbors.append(f"Zone_{nr}_{nc}")

                self.adjacency_graph[zid] = neighbors

    def log_message(self, sender: str, recipient: str, message: str, level: str = "INFO") -> None:
        """Appends an event message to the broker communication stream."""
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "sender": sender,
            "recipient": recipient,
            "message": message,
            "level": level,
        }
        self.message_log.append(entry)

    def update_zone_counts(self, counts_dict: Dict[str, float]) -> None:
        """
        Updates counts for all registered zone agents.
        counts_dict can map 'Zone_r_c' -> count or (r, c) -> count.
        """
        for k, count in counts_dict.items():
            if isinstance(k, tuple):
                zid = f"Zone_{k[0]}_{k[1]}"
            else:
                zid = str(k)

            if zid in self.agents:
                old_tier = self.agents[zid].risk_tier
                new_tier = self.agents[zid].update(
                    count,
                    green_cutoff=self.green_cutoff,
                    red_cutoff=self.red_cutoff,
                )
                if old_tier != new_tier:
                    self.log_message(
                        sender=zid,
                        recipient="Coordinator",
                        message=f"Status transition: {old_tier} -> {new_tier} (count: {count:.1f})",
                        level="WARNING" if new_tier in ["Yellow", "Red"] else "INFO",
                    )

    def evaluate_system_state(self) -> Dict[str, Any]:
        """
        Evaluates current global multi-agent state:
          - Identifies isolated cautions
          - Checks for spatially adjacent Red bottlenecks
          - Computes rerouting vectors if critical bottlenecks exist
        """
        red_zones = [zid for zid, agent in self.agents.items() if agent.risk_tier == "Red"]
        yellow_zones = [zid for zid, agent in self.agents.items() if agent.risk_tier == "Yellow"]
        green_zones = [zid for zid, agent in self.agents.items() if agent.risk_tier == "Green"]

        # Check for adjacent pairs of Red zones
        adjacent_red_pairs: List[Tuple[str, str]] = []
        red_set = set(red_zones)
        seen_pairs: Set[Tuple[str, str]] = set()

        for r_zone in red_zones:
            for neighbor in self.adjacency_graph.get(r_zone, []):
                if neighbor in red_set:
                    pair = tuple(sorted([r_zone, neighbor]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        adjacent_red_pairs.append(pair)

        # Determine Global Alert Level
        if len(adjacent_red_pairs) > 0:
            system_status = "CRITICAL COORDINATED BOTTLENECK ALERT"
            status_level = "CRITICAL"
        elif len(red_zones) > 0:
            system_status = "LOCALIZED CONGESTION CAUTION"
            status_level = "WARNING"
        elif len(yellow_zones) > 0:
            system_status = "ELEVATED TRAFFIC MONITORING"
            status_level = "CAUTION"
        else:
            system_status = "NORMAL FLOW"
            status_level = "NORMAL"

        # Generate dynamic rerouting directives
        rerouting_directives: List[str] = []
        if status_level == "CRITICAL":
            self.log_message(
                sender="Coordinator",
                recipient="BROADCAST",
                message=f"CRITICAL BOTTLENECK DETECTED! Adjacent congested zones: {adjacent_red_pairs}. Initiating diversion protocols.",
                level="CRITICAL",
            )
            for pair in adjacent_red_pairs:
                z1, z2 = pair
                # Find available safe neighboring corridors (Green or lowest count)
                candidate_exits = [
                    nbr for nbr in set(self.adjacency_graph[z1] + self.adjacency_graph[z2])
                    if nbr not in red_set
                ]
                candidate_exits.sort(key=lambda zid: self.agents[zid].current_count)

                if candidate_exits:
                    primary_exit = candidate_exits[0]
                    exit_count = self.agents[primary_exit].current_count
                    directive = (
                        f"Divert pedestrian influx from {z1} & {z2} "
                        f"towards relief corridor {primary_exit} (Occupancy: {exit_count:.1f})."
                    )
                else:
                    directive = f"Throttle intake upstream of {z1} & {z2}; no immediate neighboring capacity."

                rerouting_directives.append(directive)
                self.log_message(
                    sender="Coordinator",
                    recipient=f"{z1},{z2}",
                    message=f"DIRECTIVE: {directive}",
                    level="CRITICAL",
                )
        elif status_level == "WARNING":
            for r_zone in red_zones:
                self.log_message(
                    sender="Coordinator",
                    recipient=r_zone,
                    message=f"Localized high count ({self.agents[r_zone].current_count:.1f}). Issuing pacing warning.",
                    level="WARNING",
                )
                rerouting_directives.append(f"Monitor entry gates leading to {r_zone}.")

        total_crowd = sum(a.current_count for a in self.agents.values())

        return {
            "system_status": system_status,
            "status_level": status_level,
            "total_estimated_count": round(total_crowd, 1),
            "red_zones": red_zones,
            "yellow_zones": yellow_zones,
            "green_zones": green_zones,
            "adjacent_red_pairs": adjacent_red_pairs,
            "rerouting_directives": rerouting_directives,
            "agent_summaries": {zid: a.to_dict() for zid, a in self.agents.items()},
        }

