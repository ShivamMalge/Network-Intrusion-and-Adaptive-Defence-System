"""
Greedy Defender Agent Phase 5A.

Responsibility boundaries:
- Prioritizes ISOLATION on detection and PATCHING visible vulnerabilities.
"""

from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from core.actions import DefenderAction, ActionType


class GreedyDefender(BaseAgent):
    """
    A defender that prioritizes threats and critical vulnerabilities.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def act(self, observation: Dict[str, Any]) -> DefenderAction:
        """
        Process observation and return the highest priority defense action.
        """
        nodes = observation.get("nodes", [])
        if not nodes:
            return DefenderAction(self.agent_id, ActionType.DEFENDER_NO_OP)

        # Priority 1: ISOLATE detected threats
        detections = observation.get("detections", [])
        if detections and __import__("random").random() < 0.6:
            # Pick the first detection and isolate it
            target = detections[0]
            # Verify node exists in observation to be safe
            if any(n["node_id"] == target for n in nodes):
                 return DefenderAction(self.agent_id, ActionType.ISOLATE, target_node=target)

        # Priority 2: PATCH visible vulnerabilities only if detected or randomly with low probability
        # For Phase 5A, we just pick the first node with vulns that is detected
        for node in nodes:
            if node.get("vulnerabilities") and node.get("detected_threat"):
                return DefenderAction(self.agent_id, ActionType.PATCH, target_node=node["node_id"])

        # Default: NO_OP
        return DefenderAction(self.agent_id, ActionType.DEFENDER_NO_OP)
