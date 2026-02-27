"""
Q-Network Architecture Phase 6A.2.

Responsibility boundaries:
- Defines the neural network architecture for Deep Q-Learning.
- Handles epsilon-greedy action selection with masking.
- Decoupled from environment logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional


class QNetwork(nn.Module):
    """
    MLP-based Q-Network for discrete action spaces in adversarial simulation.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization for layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, state_dim)
        Output: (batch_size, action_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(
        self, 
        state: torch.Tensor, 
        action_mask: torch.Tensor, 
        epsilon: float, 
        device: torch.device
    ) -> int:
        """
        Performs masked epsilon-greedy action selection.
        
        Args:
            state: Tensor of shape (1, state_dim)
            action_mask: Binary tensor of shape (action_dim,)
            epsilon: Exploration rate in [0, 1]
            device: Device to run computation on
            
        Returns:
            The selected discrete action index.
        """
        # 1. Epsilon-greedy exploration
        if random.random() < epsilon:
            # Pick a random valid action
            valid_indices = torch.where(action_mask > 0)[0].tolist()
            if not valid_indices:
                # Fallback to absolute last index (NO_OP) if no mask provided
                return self.action_dim - 1
            return random.choice(valid_indices)
            
        # 2. Greedy selection
        self.eval()
        with torch.no_grad():
            q_values = self.forward(state.to(device))
            
            # Apply masking: Set invalid actions to a very low value
            masked_q = q_values.clone().squeeze(0)
            masked_q[action_mask.to(device) == 0] = -1e9
            
            # Select max index
            action_idx = torch.argmax(masked_q).item()
            
        self.train()
        return int(action_idx)
