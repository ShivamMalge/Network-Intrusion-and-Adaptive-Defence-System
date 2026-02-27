"""
Base Agent definition for Phase 0.

Responsibility boundaries:
- Receives observations and emits actions.
- Maintains internal memory strictly logically isolated from the environment's state.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Abstract representation of an active entity in the simulation.
    """

    def __init__(self, agent_id: str) -> None:
        """Initialize with an ID."""
        self.agent_id = agent_id

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Any:
        """
        Process an observation and return a chosen action.
        
        Args:
            observation: A partial view of `BaseState` formatted for this agent.
            
        Returns:
            The chosen action payload to be executed by the environment.
        """
        pass
