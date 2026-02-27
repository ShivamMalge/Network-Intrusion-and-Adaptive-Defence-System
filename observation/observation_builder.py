"""
Observation Builder Phase 0.

Responsibility boundaries:
- Constructs agent-specific partial views for fog-of-war constraints.
- Takes the global environment state and filters it based on agent type.

Mutation constraints:
- Produces entirely new output dicts/objects. No references to live environment state
  should be passed out. Every emitted observation is a deep copy slice.
"""

from typing import Any, Dict
from core.state import BaseState
from agents.base_agent import BaseAgent


class ObservationBuilder:
    """
    Constructs partial graph observations based on agent capabilities.
    """

    def build_observation(self, state: BaseState, agent: BaseAgent) -> Dict[str, Any]:
        """
        Create a customized view of the environment for the specified agent.

        Args:
            state: The globally consistent, immutable simulation state snapshot.
            agent: The inquiring adversary or defender.
            
        Returns:
            A sanitized dictionary describing the observed game state.
        """
        pass
