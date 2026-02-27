"""
Verification script for Phase 6A.2: Q-Network Architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from rl.q_network import QNetwork


def test_q_network():
    print("--- Phase 6A.2 Q-Network Test ---")
    
    state_dim = 106
    action_dim = 49
    hidden_dim = 128
    
    # 1. Initialization
    device = torch.device("cpu")
    q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    
    # Check weight init (Linear layers)
    for m in q_net.modules():
        if isinstance(m, nn.Linear):
            # Biases should be 0
            assert torch.all(m.bias == 0)
    print("Weight initialization verified (Biases=0)")

    # 2. Forward Pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, state_dim)
    output = q_net(dummy_input)
    
    print(f"Forward pass output shape: {output.shape}")
    assert output.shape == (batch_size, action_dim)
    
    # 3. Masked Action Selection (Greedy)
    state = torch.randn(1, state_dim)
    # Mask: Disable all actions except the absolute last one (NO_OP)
    mask = torch.zeros(action_dim)
    mask[-1] = 1.0
    
    # epsilon=0 means purely greedy
    action_idx = q_net.select_action(state, mask, epsilon=0.0, device=device)
    
    print(f"Selected action index (Greedy, Masked): {action_idx}")
    assert action_idx == action_dim - 1
    
    # 4. Target Network Synchronization
    target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    for p1, p2 in zip(q_net.parameters(), target_net.parameters()):
        assert torch.equal(p1, p2)
    print("Target network synchronization verified")
    
    print("\nQ-Network verification SUCCESS")


if __name__ == "__main__":
    if torch.__version__:
        test_q_network()
