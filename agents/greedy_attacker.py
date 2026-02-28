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
        self.moved_to = set()
        self.exploit_attempts = {}

    def act(self, observation: Dict[str, Any]) -> AttackerAction:
        """
        Process partial observation and return the highest priority valid action.
        """
        nodes = observation.get("nodes", [])
        if not nodes:
            return AttackerAction(self.agent_id, ActionType.ATTACKER_NO_OP)
            
        # Recovery logic: If we are stuck trying to exploit a DISCOVERED node and failing,
        # it means our lateral movement didn't give us the required privileges (e.g. Defender reset).
        # We clear moved_to so we try moving laterally again.
        for node in nodes:
            node_id = node["node_id"]
            if node_id in self.moved_to and node.get("status") == "DISCOVERED":
                self.exploit_attempts[node_id] = self.exploit_attempts.get(node_id, 0) + 1
                if self.exploit_attempts[node_id] > 3:
                    self.moved_to.remove(node_id)
                    self.exploit_attempts[node_id] = 0

        # Priority 1: MOVE_LATERAL to new DISCOVERED nodes before exploiting them
        # (This stages the privilege necessary for the exploit)
        for node in nodes:
             node_id = node["node_id"]
             status = node.get("status")
             if status == "DISCOVERED" and node_id not in self.moved_to:
                 self.moved_to.add(node_id)
                 return AttackerAction(self.agent_id, ActionType.MOVE_LATERAL, target_node=node_id)

        # Priority 2: EXPLOIT visible vulnerabilities on moved-to or compromised nodes
        for node in nodes:
            node_id = node["node_id"]
            status = node.get("status")
            if status in ["DISCOVERED", "COMPROMISED"] and node_id in self.moved_to:
                vulns = node.get("vulnerabilities", [])
                if status == "DISCOVERED": # Prioritize compromising unknown nodes
                    for vuln_id in vulns:
                        return AttackerAction(
                            self.agent_id, 
                            ActionType.EXPLOIT, 
                            target_node=node_id, 
                            metadata={"vuln_id": vuln_id, "probability": 1.0}
                        )

        # Priority 3: SCAN unscanned discovered neighbors
        for node in nodes:
            if node.get("status") == "DISCOVERED":
                return AttackerAction(self.agent_id, ActionType.SCAN, target_node=node["node_id"])

        # Default: NO_OP
        return AttackerAction(self.agent_id, ActionType.ATTACKER_NO_OP)
