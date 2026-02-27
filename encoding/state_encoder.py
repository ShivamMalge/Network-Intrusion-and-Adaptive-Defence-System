"""
State Encoder Phase 5C.

Responsibility boundaries:
- Converts variable-graph observation dictionaries into fixed-size numeric vectors.
- Suitable for Deep Q-Learning (DQN) input layers.
- Maintains strict partial observability: only encodes what is in the observation.
"""

import numpy as np
from typing import Dict, Any, List


class StateEncoder:
    """
    Encoder for converting environment observations into fixed-dimensional tensors.
    """

    def __init__(self, max_nodes: int = 32):
        self.max_nodes = max_nodes
        # NodeType mapping
        self.node_types = ["SERVER", "WORKSTATION", "ROUTER", "HONEYPOT", "INSIDER_TERMINAL", "CRITICAL_ASSET"]
        self.node_statuses = ["ACTIVE", "ISOLATED", "DOWN"]
        
        # NODE_FEATURE_DIM = 1 (known) + 1 (compromised) + 1 (detected) + 6 (type) + 1 (vuln_count) + 3 (status) = 13
        self.node_feature_dim = 13
        self.global_feature_dim = 2 # timestep, detection_count

    def encode(self, observation: Dict[str, Any], role: str) -> np.ndarray:
        """
        Encodes an observation into a flattened numpy array.
        
        Output shape: [MAX_NODES * NODE_FEATURE_DIM + GLOBAL_FEATURE_DIM]
        """
        # 1. Initialize node matrix [MAX_NODES, NODE_FEATURE_DIM]
        node_matrix = np.zeros((self.max_nodes, self.node_feature_dim), dtype=np.float32)
        
        # 2. Extract and sort observed nodes for deterministic mapping
        observed_nodes = observation.get("nodes", [])
        # Deterministic slot assignment by ID
        sorted_nodes = sorted(observed_nodes, key=lambda x: x["node_id"])
        
        for i, node_data in enumerate(sorted_nodes):
            if i >= self.max_nodes:
                break
            
            node_id = node_data["node_id"]
            
            # Feature extraction
            # [0] is_known (Always 1 if in observation nodes list)
            node_matrix[i, 0] = 1.0
            
            # [1] is_compromised
            node_matrix[i, 1] = 1.0 if node_data.get("status") == "COMPROMISED" else 0.0
            
            # [2] is_detected
            node_matrix[i, 2] = 1.0 if node_data.get("detected_threat") else 0.0
            
            # [3:9] node_type_one_hot
            node_type = node_data.get("node_type", "WORKSTATION")
            if node_type in self.node_types:
                type_idx = self.node_types.index(node_type)
                node_matrix[i, 3 + type_idx] = 1.0
                
            # [9] vulnerability_count (normalized by a reasonable max, e.g., 5)
            vuln_count = len(node_data.get("vulnerabilities", []))
            node_matrix[i, 9] = min(vuln_count / 5.0, 1.0)
            
            # [10:13] node_status_one_hot
            status = node_data.get("status", "ACTIVE")
            if status in self.node_statuses:
                status_idx = self.node_statuses.index(status)
                node_matrix[i, 10 + status_idx] = 1.0
            elif status == "COMPROMISED":
                # For encoding status, COMPROMISED nodes are still ACTIVE in status enum terms
                node_matrix[i, 10] = 1.0
                
        # 3. Global features
        global_vec = np.zeros(self.global_feature_dim, dtype=np.float32)
        
        # normalized timestep (assuming max 100 steps)
        global_vec[0] = min(observation.get("timestep", 0) / 100.0, 1.0)
        
        # detection count
        global_vec[1] = len(observation.get("detections", [])) / self.max_nodes
        
        # 4. Concatenate and return
        return np.concatenate([node_matrix.flatten(), global_vec])

    @property
    def observation_dim(self) -> int:
        return self.max_nodes * self.node_feature_dim + self.global_feature_dim
