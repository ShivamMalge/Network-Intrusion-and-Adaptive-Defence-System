"""
Phase 6C Step B: Final Equilibrium Stress Test (A5 vs D5).
Evaluates 1000 deterministic episodes to confirm stability.
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
        
        # 3. Select greedy action (Deterministic, Epsilon=0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            masked_q = q_values.clone()
            masked_q[0, mask_tensor == 0] = -float('inf')
            action_idx = masked_q.argmax(dim=1).item()
            
        # 4. Decode to Action object
        nodes = sorted(observation.get("nodes", []), key=lambda x: x["node_id"])
        node_ids = [n["node_id"] for n in nodes]
        
        return self.action_encoder.decode_action(action_idx, node_ids, self.agent_id)

def run_stress_test(episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_nodes = 32
    state_enc = StateEncoder(max_nodes=max_nodes)
    act_enc = ActionEncoder(max_nodes=max_nodes)
    state_dim = state_enc.observation_dim
    action_dim = act_enc.action_dim
    hidden_dim = 256
    
    # Load A7 and D7
    atk_path = "dqn_attacker_cycle_7.pt"
    def_path = "dqn_defender_cycle_7.pt"
    
    if not os.path.exists(atk_path) or not os.path.exists(def_path):
        print("Error: Cycle 7 checkpoints not found.")
        return

    atk_model = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    atk_model.load_state_dict(torch.load(atk_path, map_location=device))
    atk_agent = DQNAgent(atk_model, act_enc, role="attacker")
    
    def_model = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    def_model.load_state_dict(torch.load(def_path, map_location=device))
    def_agent = DQNAgent(def_model, act_enc, role="defender")
    
    print(f"Starting Stress Test: A7 vs D7 ({episodes} episodes)")
    
    rng = CentralizedRNG(seed=999) # Separate seed for stress test
    base_env = EnvironmentEngine(rng)
    
    results = []
    wins = 0
    current_win_streak = 0
    current_loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    
    for i in range(episodes):
        obs_dict = base_env.reset()
        done = False
        ep_reward_atk = 0
        steps = 0
        
        while not done:
            atk_obs = base_env.get_observation_by_id("atk_1")
            def_obs = base_env.get_observation_by_id("def_1")
            
            a_atk = atk_agent.act(atk_obs)
            a_def = def_agent.act(def_obs)
            
            obs_dict, rewards_dict, done, info = base_env.step(a_atk, a_def)
            ep_reward_atk += rewards_dict.get("atk_1", 0)
            steps += 1
            if steps >= 50:
                done = True
        
        is_win = info.get("termination_reason") == "CRITICAL_COMPROMISED"
        if is_win:
            wins += 1
            current_win_streak += 1
            current_loss_streak = 0
        else:
            current_loss_streak += 1
            current_win_streak = 0
            
        max_win_streak = max(max_win_streak, current_win_streak)
        max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        results.append({
            "win": is_win,
            "reward": ep_reward_atk,
            "length": steps
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{episodes} | Win Rate: {(wins/(i+1))*100:.1f}%")

    # Compute Final Stats
    win_rate = (wins / episodes) * 100
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    avg_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    
    # Verdict Logic
    is_stable = (60.0 <= win_rate <= 68.0) and (max_loss_streak <= 10)
    verdict = "STABLE" if is_stable else "UNSTABLE"
    
    summary = {
        "episodes": episodes,
        "attacker_win_rate": win_rate,
        "avg_length": avg_length,
        "avg_reward": avg_reward,
        "reward_std": std_reward,
        "length_std": std_length,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "verdict": verdict
    }
    
    with open("equilibrium_stress_test.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*30)
    print("STRESS TEST SUMMARY")
    print("="*30)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.2f}")
        else:
            print(f"{k:20}: {v}")
    print("="*30)

if __name__ == "__main__":
    run_stress_test(1000)
