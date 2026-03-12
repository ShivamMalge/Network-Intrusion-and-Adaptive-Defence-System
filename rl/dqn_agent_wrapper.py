"""
DQN Agent Wrapper for Phase 11 Regression Audit.

Wraps a saved .pt checkpoint into the BaseAgent interface for deterministic evaluation.
"""

import torch
import numpy as np
import random
from typing import Any, Dict

from agents.base_agent import BaseAgent
from core.actions import AttackerAction, DefenderAction, ActionType
from rl.q_network import QNetwork
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder

class DQNAgentWrapper(BaseAgent):
    def __init__(self, agent_id: str, model_path: str, is_attacker: bool, device: str = "cpu"):
        super().__init__(agent_id)
        self.is_attacker = is_attacker
        self.device = torch.device(device)
        
        # Load encoders (assumed to be static or matching the checkpoint)
        # We use standard encoders from the project
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
        
        # Initialize network
        # We need to know state/action dims. Standard dims from Phase 6/7:
        # State: ~150-200 (depends on graph size, but let's assume 32 nodes)
        # Action: 32 nodes * actions per node
        # We can dynamically determine this from the checkpoint or a dummy env.
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine dimensions from weights
        # weight shape [out_features, in_features]
        first_layer_weight = checkpoint.get("fc1.weight")
        if first_layer_weight is None:
             # Try alternate names if any
             first_layer_weight = next(iter(checkpoint.values()))
             
        state_dim = first_layer_weight.shape[1]
        action_dim = list(checkpoint.values())[-1].shape[0] # Output layer
        
        self.model = QNetwork(state_dim, action_dim).to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
    def act(self, observation: Dict[str, Any]) -> Any:
        with torch.no_grad():
            role = "attacker" if self.is_attacker else "defender"
            state_tensor = self.state_encoder.encode(observation, role)
            state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
            
            # Action mask
            mask = self.action_encoder.generate_action_mask(observation, role)
            mask_tensor = torch.FloatTensor(mask).to(self.device)
            
            q_values = self.model(state_tensor)
            
            # Apply mask (infinite penalty to invalid actions)
            masked_q = q_values - (1.0 - mask_tensor) * 1e9
            action_idx = masked_q.argmax(dim=1).item()
            
            # Decode action
            nodes = observation.get("nodes", [])
            sorted_node_ids = [n["node_id"] for n in sorted(nodes, key=lambda x: x["node_id"])]
            action_obj = self.action_encoder.decode_action(action_idx, sorted_node_ids, self.agent_id)
            return action_obj
