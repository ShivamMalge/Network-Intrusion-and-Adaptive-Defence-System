"""
Random Defender Agent Phase 5A.

Responsibility boundaries:
- Randomly selects defense actions from the full network observation.
"""

import random
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from core.actions import DefenderAction, ActionType


class RandomDefender(BaseAgent):
    """
    A defender that selects defensive actions randomly.
    """

    def __init__(self, agent_id: str, seed: int = 88):
        super().__init__(agent_id)
        self._rng = random.Random(seed)

    def act(self, observation: Dict[str, Any]) -> DefenderAction:
        """
        Process observation and return a random valid defense action.
        """
        nodes = observation.get("nodes", [])
        if not nodes:
            return DefenderAction(self.agent_id, ActionType.DEFENDER_NO_OP)

        valid_actions: List[DefenderAction] = []
        valid_actions.append(DefenderAction(self.agent_id, ActionType.DEFENDER_NO_OP))

        for node in nodes:
            node_id = node["node_id"]
            
            # PATCH: Valid if vullnerabilities exist
            if node.get("vulnerabilities"):
                valid_actions.append(DefenderAction(self.agent_id, ActionType.PATCH, target_node=node_id))
            
            # ISOLATE: Valid on any node (usually)
            valid_actions.append(DefenderAction(self.agent_id, ActionType.ISOLATE, target_node=node_id))
            
            # RESET_PRIVILEGE: Valid on any node
            valid_actions.append(DefenderAction(self.agent_id, ActionType.RESET_PRIVILEGE, target_node=node_id))

        return self._rng.choice(valid_actions)
