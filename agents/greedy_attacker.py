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
            is_compromised = node.get("compromised", False)
            if node_id in self.moved_to and not is_compromised:
                self.exploit_attempts[node_id] = self.exploit_attempts.get(node_id, 0) + 1
                if self.exploit_attempts[node_id] > 3:
                    self.moved_to.remove(node_id)
                    self.exploit_attempts[node_id] = 0

        # Priority 1: MOVE_LATERAL to new DISCOVERED nodes before exploiting them
        # (This stages the privilege necessary for the exploit)
        
        compromised_nodes = {n["node_id"] for n in nodes if n.get("compromised", False)}
        edges = observation.get("edges", [])
        
        valid_lateral_targets = set()
        for edge in edges:
            if edge["source"] in compromised_nodes:
                valid_lateral_targets.add(edge["target"])
                
        for node in nodes:
             node_id = node["node_id"]
             is_compromised = node.get("compromised", False)
             
             if not is_compromised and node_id not in self.moved_to:
                 # Initial entry node doesn't require adjacent compromised node
                 if len(compromised_nodes) == 0 or node_id in valid_lateral_targets:
                     self.moved_to.add(node_id)
                     return AttackerAction(self.agent_id, ActionType.MOVE_LATERAL, target_node=node_id)

        # Priority 2: EXPLOIT visible vulnerabilities on moved-to or compromised nodes
        for node in nodes:
            node_id = node["node_id"]
            is_compromised = node.get("compromised", False)
            node_type = node.get("node_type", "UNKNOWN")
            privilege = node.get("privilege", "NONE")
            
            if node_id in self.moved_to:
                vulns = node.get("vulnerabilities", [])
                
                # We want to exploit if we aren't compromised yet, OR if it's a critical asset and we aren't ROOT
                needs_exploit = False
                if not is_compromised:
                    needs_exploit = True
                elif node_type == "CRITICAL_ASSET" and privilege != "ROOT":
                    needs_exploit = True
                    
                if needs_exploit:
                    for vuln_id in vulns:
                        return AttackerAction(
                            self.agent_id, 
                            ActionType.EXPLOIT, 
                            target_node=node_id, 
                            metadata={"vuln_id": vuln_id, "probability": 1.0}
                        )

        # Priority 3: SCAN unscanned neighbors (we assume anything uncompromised with no vulns visible needs a scan)
        for node in nodes:
            is_compromised = node.get("compromised", False)
            vulns = node.get("vulnerabilities", [])
            if not is_compromised and not vulns:
                return AttackerAction(self.agent_id, ActionType.SCAN, target_node=node["node_id"])

        # Default: NO_OP
        return AttackerAction(self.agent_id, ActionType.ATTACKER_NO_OP)
