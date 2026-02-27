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
from typing import Any, Dict
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
        self._graph_manager = graph_manager
        self._vulnerability_registry = vulnerability_registry
        self._timestep = 0
        self._episode_done = False

    @property
    def graph_manager(self) -> GraphManager:
        return self._graph_manager

    @property
    def vulnerability_registry(self) -> VulnerabilityRegistry:
        return self._vulnerability_registry

    @property
    def timestamp(self) -> int:
        return self._timestep

    @property
    def episode_done(self) -> bool:
        return self._episode_done

    def increment_time(self) -> None:
        """Advance simulation clock by one step."""
        self._timestep += 1

    def mark_done(self) -> None:
        """Mark the episode as terminal."""
        self._episode_done = True

    def compute_state_hash(self) -> int:
        """Compute a deterministic hash of the current state structure."""
        return hash((
            self._timestep,
            self._graph_manager.number_of_nodes(),
            self._graph_manager.number_of_edges(),
            self._vulnerability_registry.get_node_count()  # Use public method
        ))

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Return a safe, read-only projection of the state for a specific agent.
        """
        return {
            "timestep": self._timestep,
            "done": self._episode_done,
        }
