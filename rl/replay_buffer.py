"""
Replay Buffer and Epsilon Scheduler Phase 6A.3.

Responsibility boundaries:
- Experience Replay Buffer for stable DQN training.
- Linear Epsilon-Greedy Scheduler for exploration.
- Uses preallocated PyTorch tensors for performance.
"""

import torch
import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-memory circular buffer for storing simulation transitions.
    Uses preallocated tensors for memory efficiency and faster sampling.
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        
        # Preallocate tensors
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)
        
        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Adds a transition to the buffer.
        """
        self.states[self.position] = torch.from_numpy(state).float()
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.from_numpy(next_state).float()
        self.dones[self.position] = 1.0 if done else 0.0
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly samples a batch of transitions.
        Returns tensors on the configured device.
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Return detached tensors on the target device
        return (
            self.states[indices].to(self.device).detach(),
            self.actions[indices].to(self.device).detach(),
            self.rewards[indices].to(self.device).detach(),
            self.next_states[indices].to(self.device).detach(),
            self.dones[indices].to(self.device).detach()
        )

    def __len__(self) -> int:
        return self.size


class EpsilonScheduler:
    """
    Linear decay scheduler for the exploration-exploitation trade-off.
    """

    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def value(self, step: int) -> float:
        """
        Calculates epsilon for a given step.
        """
        if step >= self.decay_steps:
            return self.end
        
        # Linear decay: start -> end over decay_steps
        epsilon = self.start - (self.start - self.end) * (step / self.decay_steps)
        return max(self.end, epsilon)
