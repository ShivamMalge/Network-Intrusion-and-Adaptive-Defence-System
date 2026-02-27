"""
Greedy Attacker Agent Phase 5A.

Responsibility boundaries:
- Prioritizes actions based on a greedy heuristic.
- EXPLOIT > SCAN > MOVE_LATERAL > NO_OP.
"""

from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from core.actions import AttackerAction, ActionType


class GreedyAttacker(BaseAgent):
    """
    An attacker that prioritizes exploitation and scanning over random movement.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def act(self, observation: Dict[str, Any]) -> AttackerAction:
        """
        Process partial observation and return the highest priority valid action.
        """
        nodes = observation.get("nodes", [])
        if not nodes:
            return AttackerAction(self.agent_id, ActionType.ATTACKER_NO_OP)

        # Priority 1: EXPLOIT visible vulnerabilities
        # We look for the first available vulnerability on a known node
        for node in nodes:
            node_id = node["node_id"]
            status = node.get("status")
            if status in ["KNOWN", "COMPROMISED"]:
                vulns = node.get("vulnerabilities", [])
                # If compromised, maybe we don't need to exploit again unless for escalation (not in phase 5A)
                # But for now, we exploit if we find a vuln on a node we haven't maxed out? 
                # Let's say exploit if not yet ROOT (if privilege info available) or just exploit any visible
                if status == "KNOWN": # Prioritize compromising unknown nodes
                    for vuln_id in vulns:
                        return AttackerAction(
                            self.agent_id, 
                            ActionType.EXPLOIT, 
                            target_node=node_id, 
                            metadata={"vuln_id": vuln_id, "probability": 1.0} # Ensure progress for baseline
                        )

        # Priority 2: SCAN unscanned neighbors
        for node in nodes:
            if node.get("status") == "DISCOVERED":
                return AttackerAction(self.agent_id, ActionType.SCAN, target_node=node["node_id"])

        # Priority 3: MOVE_LATERAL (Discovery)
        for node in nodes:
             if node.get("status") == "DISCOVERED":
                return AttackerAction(self.agent_id, ActionType.MOVE_LATERAL, target_node=node["node_id"])

        # Default: NO_OP
        return AttackerAction(self.agent_id, ActionType.ATTACKER_NO_OP)
