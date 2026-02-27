"""
Random Attacker Agent Phase 5A.

Responsibility boundaries:
- Randomly selects valid actions based on partial observation.
- Does not possess internal state or direct environment access.
"""

import random
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from core.actions import AttackerAction, ActionType


class RandomAttacker(BaseAgent):
    """
    An attacker that selects actions randomly from its current knowledge.
    """

    def __init__(self, agent_id: str, seed: int = 42):
        super().__init__(agent_id)
        self._rng = random.Random(seed)

    def act(self, observation: Dict[str, Any]) -> AttackerAction:
        """
        Process partial observation and return a random valid action.
        """
        nodes = observation.get("nodes", [])
        if not nodes:
            return AttackerAction(self.agent_id, ActionType.ATTACKER_NO_OP)

        valid_actions: List[AttackerAction] = []

        # 1. NO_OP is always valid
        valid_actions.append(AttackerAction(self.agent_id, ActionType.ATTACKER_NO_OP))

        # 2. Identify potential targets from observation
        # Attacker knows about:
        # - Scanned nodes (status == 'KNOWN')
        # - Compromised nodes (status == 'COMPROMISED')
        # - Discovered neighbors (status == 'DISCOVERED')

        known_nodes = [n["node_id"] for n in nodes if n.get("status") in ["KNOWN", "COMPROMISED"]]
        discovered_nodes = [n["node_id"] for n in nodes if n.get("status") == "DISCOVERED"]
        compromised_nodes = [n["node_id"] for n in nodes if n.get("status") == "COMPROMISED"]

        # SCAN: Target discovered neighbors or known nodes
        for node_id in discovered_nodes:
            valid_actions.append(AttackerAction(self.agent_id, ActionType.SCAN, target_node=node_id))

        # EXPLOIT: Target known nodes with visible vulnerabilities
        for node in nodes:
            node_id = node["node_id"]
            if node_id in known_nodes:
                vulns = node.get("vulnerabilities", [])
                for vuln_id in vulns:
                    valid_actions.append(
                        AttackerAction(
                            self.agent_id, 
                            ActionType.EXPLOIT, 
                            target_node=node_id, 
                            metadata={"vuln_id": vuln_id}
                        )
                    )

        # MOVE_LATERAL: Placeholder, currently treated as NO_OP or move to neighbor
        # In this phase, we'll keep it simple: if compromised, can move to neighbors
        for node_id in discovered_nodes:
            valid_actions.append(AttackerAction(self.agent_id, ActionType.MOVE_LATERAL, target_node=node_id))

        # Randomly select one
        return self._rng.choice(valid_actions)
