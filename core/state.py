"""
State definition for Phase 3.

Responsibility boundaries:
- Represents an immutable snapshot of the simulation state at time `t`.
- Defines properties that agents cannot mutate directly.

Mutation constraints:
- State definitions here must be treated as Read-Only by all modules
  EXCEPT the Environment, which creates new state objects per step.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, List, Tuple
from graph.graph_manager import GraphManager
from vulnerabilities.vulnerability_registry import VulnerabilityRegistry


class BaseState(ABC):
    """
    Immutable state container pattern.
    State objects must not be mutated directly by agents.
    """

    @property
    @abstractmethod
    def timestamp(self) -> int:
        """Return the current step/timestamp of the state."""
        pass

    @abstractmethod
    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Return a safe, read-only projection of the state for a specific agent.
        """
        pass


class SimulationState(BaseState):
    """
    Mutable state container strictly owned by the EnvironmentEngine.
    """

    def __init__(self, graph_manager: GraphManager, vulnerability_registry: VulnerabilityRegistry):
        super().__init__()
        self._graph_manager = graph_manager
        self._vulnerability_registry = vulnerability_registry
        self._timestamp = 0
        self._done = False

        # Phase 4 Epistemic State
        self._attacker_scanned: Set[str] = set()
        self._attacker_compromised: Set[str] = set()
        self._defender_detected_nodes: Set[str] = set()
        
        # Phase 5B Stochastic Detection state
        self._detection_queue: List[Tuple[str, int]] = []
        self._detection_probability = 0.7  # Defaults
        self._detection_delay = 2
        self._false_positive_rate = 0.05

    def compute_state_hash(self) -> int:
        """
        Produce a deterministic hash of the current global state.
        Includes graph topology, vulnerability status, and epistemic sets.
        """
        graph_hash = self._graph_manager.compute_topology_hash()
        vuln_hash = self._vulnerability_registry.compute_state_hash()
        
        # Epistemic hashes
        atk_scan_hash = hash(tuple(sorted(list(self._attacker_scanned))))
        atk_comp_hash = hash(tuple(sorted(list(self._attacker_compromised))))
        def_det_hash = hash(tuple(sorted(list(self._defender_detected_nodes))))
        
        # Detection queue hash
        queue_hash = hash(tuple(sorted(self._detection_queue)))

        return hash((
            graph_hash, 
            vuln_hash, 
            self._timestamp, 
            atk_scan_hash, 
            atk_comp_hash, 
            def_det_hash,
            queue_hash
        ))

    @property
    def graph_manager(self) -> GraphManager:
        return self._graph_manager

    @property
    def vulnerability_registry(self) -> VulnerabilityRegistry:
        return self._vulnerability_registry

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def episode_done(self) -> bool:
        return self._done

    @property
    def attacker_scanned(self) -> Set[str]:
        return set(self._attacker_scanned)

    @property
    def attacker_compromised(self) -> Set[str]:
        return set(self._attacker_compromised)

    @property
    def defender_detected_nodes(self) -> Set[str]:
        return set(self._defender_detected_nodes)
        
    @property
    def detection_probability(self) -> float:
        return self._detection_probability

    @property
    def detection_delay(self) -> int:
        return self._detection_delay

    @property
    def false_positive_rate(self) -> float:
        return self._false_positive_rate

    def add_scanned_node(self, node_id: str) -> None:
        self._attacker_scanned.add(node_id)

    def add_compromised_node(self, node_id: str) -> None:
        self._attacker_compromised.add(node_id)

    def add_detected_node(self, node_id: str) -> None:
        self._defender_detected_nodes.add(node_id)
        
    def add_to_detection_queue(self, node_id: str, trigger_time: int) -> None:
        """Add a detection event to be triggered in the future."""
        self._detection_queue.append((node_id, trigger_time))
        
    def process_detection_queue(self, current_time: int) -> List[str]:
        """Identify triggered detections and remove them from the queue."""
        triggered = [n for n, t in self._detection_queue if t <= current_time]
        self._detection_queue = [(n, t) for n, t in self._detection_queue if t > current_time]
        return triggered

    def set_ids_parameters(self, prob: float, prompt_delay: int, fp_rate: float) -> None:
        """Configure IDS parameters."""
        self._detection_probability = prob
        self._detection_delay = prompt_delay
        self._false_positive_rate = fp_rate

    def increment_time(self) -> None:
        """Advance simulation clock by one step."""
        self._timestamp += 1

    def mark_done(self) -> None:
        """Mark the episode as terminal."""
        self._episode_done = True

    def compute_state_hash(self) -> int:
        """Compute a deterministic hash of the current state structure."""
        return hash((
            self._timestep,
            self._graph_manager.number_of_nodes(),
            self._graph_manager.number_of_edges(),
            self._vulnerability_registry.get_node_count(),
            tuple(sorted(self._attacker_scanned)),
            tuple(sorted(self._attacker_compromised)),
            tuple(sorted(self._defender_detected_nodes))
        ))

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Return a safe, read-only projection of the state for a specific agent.
        """
        return {
            "timestep": self._timestep,
            "done": self._episode_done,
        }
