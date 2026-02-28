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
import numpy as np
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

    def select_action(self, state, mask, epsilon, device):
        """
        state: (1, state_dim)
        mask: (action_dim,) float tensor (1.0 valid, 0.0 invalid)
        """
        
        mask = mask.to(device)
        
        # Exploration
        if np.random.rand() < epsilon:
            valid_actions = torch.where(mask == 1.0)[0]
            if len(valid_actions) == 0:
                return 0  # safety fallback
            idx = torch.randint(0, len(valid_actions), (1,))
            return valid_actions[idx].item()
            
        # Exploitation
        with torch.no_grad():
            q_values = self.forward(state).squeeze(0)
            
            masked_q = q_values.clone()
            masked_q[mask == 0] = -1e9
            
            return torch.argmax(masked_q).item()
