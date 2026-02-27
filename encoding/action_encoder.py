"""
Action Encoder Phase 5C.

Responsibility boundaries:
- Maps discrete action indices to AttackerAction/DefenderAction objects.
- Generates action masks to enforce validity and partial observability.
- Maintains fixed-dimensional action space for DQN compatibility.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from core.actions import AttackerAction, DefenderAction, ActionType


class ActionEncoder:
    """
    Encoder for a fixed-size discrete action space.
    """

    def __init__(self, max_nodes: int = 32):
        self.max_nodes = max_nodes
        # Ordered action types for consistent indexing
        self.attacker_types = [ActionType.SCAN, ActionType.EXPLOIT, ActionType.MOVE_LATERAL]
        self.defender_types = [ActionType.PATCH, ActionType.ISOLATE, ActionType.RESET_PRIVILEGE]
        
        # Total actions = (3 types * max_nodes) + 1 (NO_OP) = 3 * max_nodes + 1 for each role separately
        # Or unified? The prompt says: [SCAN] ... [RESET] ... [NO_OP]
        # This implies a unified space or role-specific. I'll provide a mapping that handles both.
        self.all_types = self.attacker_types + self.defender_types
        self.action_dim = len(self.all_types) * self.max_nodes + 1
        self.no_op_index = self.action_dim - 1

    def _get_no_op(self, agent_id: str) -> Any:
        """Internal helper to return role-appropriate NO_OP."""
        if agent_id.startswith("atk"):
            return AttackerAction(agent_id, ActionType.ATTACKER_NO_OP)
        else:
            return DefenderAction(agent_id, ActionType.DEFENDER_NO_OP)

    def encode_action(self, action: Any, sorted_node_ids: List[str]) -> int:
        """
        Converts an Action object to a discrete index.
        """
        if action.action_type == ActionType.NO_OP or action.action_type.name.endswith("NO_OP"):
            return self.no_op_index
            
        type_idx = self.all_types.index(action.action_type)
        if action.target_node in sorted_node_ids:
            node_idx = sorted_node_ids.index(action.target_node)
            return type_idx * self.max_nodes + node_idx
            
        return self.no_op_index

    def decode_action(self, index: int, sorted_node_ids: List[str], agent_id: str) -> Any:
        """
        Converts a discrete index to an Action object.
        """
        if index >= self.no_op_index:
            return self._get_no_op(agent_id)
                
        type_idx = index // self.max_nodes
        node_idx = index % self.max_nodes
        
        # If node_idx is out of bounds for current observation, return NO_OP
        if node_idx >= len(sorted_node_ids):
             return self._get_no_op(agent_id)
        
        action_type = self.all_types[type_idx]
        target_node = sorted_node_ids[node_idx]
        
        if agent_id.startswith("atk"):
            # For EXPLOIT, we need a vuln_id. Baseline greedy use first available.
            metadata = {}
            if action_type == ActionType.EXPLOIT:
                metadata["vuln_id"] = "UNKNOWN" # Logic handled elsewhere or placeholder
            return AttackerAction(agent_id, action_type, target_node, metadata)
        else:
            return DefenderAction(agent_id, action_type, target_node)

    def generate_action_mask(self, observation: Dict[str, Any], role: str) -> np.ndarray:
        """
        Generates a binary mask of shape [action_dim].
        1.0 for valid actions, 0.0 for invalid.
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        mask[self.no_op_index] = 1.0 # NO_OP is always allowed
        
        observed_nodes = observation.get("nodes", [])
        sorted_nodes = sorted(observed_nodes, key=lambda x: x["node_id"])
        sorted_ids = [n["node_id"] for n in sorted_nodes]
        
        if role == "attacker":
            # Mask for SCAN, EXPLOIT, MOVE_LATERAL
            compromised = [n["node_id"] for n in sorted_nodes if n.get("status") == "COMPROMISED"]
            
            for i, node_data in enumerate(sorted_nodes):
                if i >= self.max_nodes: break
                node_id = node_data["node_id"]
                
                # SCAN: Valid if node is DISCOVERED but not KNOWN (in our obs logic, KNOWN means scanned/comp)
                # Actually, in simplified terms, allow SCAN on any known node that isn't compromised
                if node_data.get("status") != "COMPROMISED":
                    mask[self.all_types.index(ActionType.SCAN) * self.max_nodes + i] = 1.0
                
                # EXPLOIT: Valid if node has visible vulnerabilities and is not compromised
                if len(node_data.get("vulnerabilities", [])) > 0 and node_data.get("status") != "COMPROMISED":
                    mask[self.all_types.index(ActionType.EXPLOIT) * self.max_nodes + i] = 1.0
                    
                # MOVE_LATERAL: Valid if node is a neighbor of a compromised node (DISCOVERED status)
                # and not already compromised
                if node_data.get("status") == "DISCOVERED":
                    mask[self.all_types.index(ActionType.MOVE_LATERAL) * self.max_nodes + i] = 1.0
                    
        elif role == "defender":
            # Mask for PATCH, ISOLATE, RESET_PRIVILEGE
            for i, node_data in enumerate(sorted_nodes):
                if i >= self.max_nodes: break
                node_id = node_data["node_id"]
                
                # PATCH: Valid if vulnerabilities exist
                if len(node_data.get("vulnerabilities", [])) > 0:
                     mask[self.all_types.index(ActionType.PATCH) * self.max_nodes + i] = 1.0
                
                # ISOLATE: Valid if node is not already isolated (status ACTIVE or COMPROMISED)
                if node_data.get("status") != "ISOLATED":
                     mask[self.all_types.index(ActionType.ISOLATE) * self.max_nodes + i] = 1.0
                     
                # RESET_PRIVILEGE: Valid if detected threat or suspicious
                if node_data.get("detected_threat"):
                     mask[self.all_types.index(ActionType.RESET_PRIVILEGE) * self.max_nodes + i] = 1.0
                     
        return mask
