"""
Phase 6C Step 1: Cross-Checkpoint Evaluation Matrix.
Evaluates all (Attacker_i, Defender_j) pairs for Cycles 0-3.
"""

import os
import sys
import torch
import numpy as np
import json
from typing import Dict, List, Any

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.q_network import QNetwork

class DQNAgent:
    """A wrapper to use a Q-Network as a fixed policy for evaluation."""
    def __init__(self, model, action_encoder, role="attacker"):
        self.model = model
        self.action_encoder = action_encoder
        self.role = role
        self.agent_id = "atk_1" if role == "attacker" else "def_1"
        self.state_enc = StateEncoder(max_nodes=32)

    def act(self, observation):
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # 1. Encode state
        encoded_state = self.state_enc.encode(observation, self.role)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)
        
        # 2. Generate mask
        mask = self.action_encoder.generate_action_mask(observation, self.role)
        mask_tensor = torch.from_numpy(mask).float().to(device)
        
        # 3. Select greedy action (Epsilon=0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            masked_q = q_values.clone()
            # Handle batch dimension [1, action_dim]
            masked_q[0, mask_tensor == 0] = -float('inf')
            action_idx = masked_q.argmax(dim=1).item()
            
        # 4. Decode to Action object
        nodes = sorted(observation.get("nodes", []), key=lambda x: x["node_id"])
        node_ids = [n["node_id"] for n in nodes]
        
        return self.action_encoder.decode_action(action_idx, node_ids, self.agent_id)

def run_evaluation(base_env, attacker_policy, defender_policy, episodes=100):
    """Run evaluation matches."""
    wins = 0
    rewards = []
    lengths = []
    
    for _ in range(episodes):
        obs_dict = base_env.reset()
        done = False
        ep_reward_atk = 0
        steps = 0
        
        while not done:
            atk_obs = base_env.get_observation_by_id("atk_1")
            def_obs = base_env.get_observation_by_id("def_1")
            
            a_atk = attacker_policy.act(atk_obs)
            a_def = defender_policy.act(def_obs)
            
            obs_dict, rewards_dict, done, info = base_env.step(a_atk, a_def)
            ep_reward_atk += rewards_dict.get("atk_1", 0)
            steps += 1
            if steps >= 50:
                done = True
        
        if info.get("termination_reason") == "CRITICAL_COMPROMISED":
            wins += 1
        rewards.append(ep_reward_atk)
        lengths.append(steps)
        
    return {
        "win_rate": (wins / episodes) * 100,
        "avg_reward_atk": float(np.mean(rewards)),
        "avg_length": float(np.mean(lengths))
    }

def main(episodes_per_pair=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_nodes = 32
    state_enc = StateEncoder(max_nodes=max_nodes)
    act_enc = ActionEncoder(max_nodes=max_nodes)
    state_dim = state_enc.observation_dim
    action_dim = act_enc.action_dim
    hidden_dim = 256
    
    # Dynamic checkpoint discovery
    atk_checkpoints = {"A0": "dqn_attacker.pt"}
    def_checkpoints = {"D0": "dqn_defender.pt"}
    
    # Scan for cycle checkpoints
    for i in range(1, 100):
        a_path = f"dqn_attacker_cycle_{i}.pt"
        d_path = f"dqn_defender_cycle_{i}.pt"
        if os.path.exists(a_path):
            atk_checkpoints[f"A{i}"] = a_path
        if os.path.exists(d_path):
            def_checkpoints[f"D{i}"] = d_path
        if not os.path.exists(a_path) and not os.path.exists(d_path) and i > 5:
            break
    
    print(f"Starting Cross-Checkpoint Evaluation Matrix ({episodes_per_pair} episodes/pair)")
    print(f"Attackers: {list(atk_checkpoints.keys())}")
    print(f"Defenders: {list(def_checkpoints.keys())}\n")
    
    rng = CentralizedRNG(seed=42)
    base_env = EnvironmentEngine(rng)
    
    matrix = {}
    
    for a_name, a_path in atk_checkpoints.items():
        matrix[a_name] = {}
        atk_model = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        atk_model.load_state_dict(torch.load(a_path, map_location=device))
        atk_agent = DQNAgent(atk_model, act_enc, role="attacker")
        
        for d_name, d_path in def_checkpoints.items():
            print(f"Evaluating {a_name} vs {d_name}...")
            def_model = QNetwork(state_dim, action_dim, hidden_dim).to(device)
            def_model.load_state_dict(torch.load(d_path, map_location=device))
            def_agent = DQNAgent(def_model, act_enc, role="defender")
            
            result = run_evaluation(base_env, atk_agent, def_agent, episodes=episodes_per_pair)
            matrix[a_name][d_name] = result
            print(f"  Result: Win Rate {result['win_rate']:.1f}% | Avg Len {result['avg_length']:.1f}")

    # Output JSON
    with open("evaluation_matrix.json", "w") as f:
        json.dump(matrix, f, indent=4)
    
    # Print Formatted Table
    print("\n" + "="*50)
    print("WIN RATE MATRIX (%)")
    print("="*50)
    
    header = "Atk\\Def | " + " | ".join(def_checkpoints.keys())
    print(header)
    print("-" * len(header))
    
    for a_name in atk_checkpoints.keys():
        row = f"{a_name:7} | "
        values = []
        for d_name in def_checkpoints.keys():
            wr = matrix[a_name][d_name]['win_rate']
            values.append(f"{wr:4.1f}")
        row += " | ".join(values)
        print(row)
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    
    main(episodes_per_pair=args.episodes)
