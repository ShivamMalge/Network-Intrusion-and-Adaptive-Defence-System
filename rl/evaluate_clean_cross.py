"""
Clean Cross-Agent Equilibrium Evaluation for Phase 8 Part C.

Responsibility boundaries:
- Orchestrates deterministic evaluation of dqn_attacker_clean vs dqn_defender_clean.
- Completely bypassing wrapper logic strictly using step pipeline and encoders.
- Computes advanced streak and variance metrics.
"""

import os
import sys
import torch
import numpy as np
import json
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.q_network import QNetwork
from core.actions import ActionType

def evaluate_equilibrium(episodes=1000, max_steps=50):
    print(f"Starting Clean Cross-Agent Evaluation (Episodes: {episodes})")
    
    # 1. Setup Environment
    rng = CentralizedRNG(seed=999) # Different evaluation seed
    env = EnvironmentEngine(rng)
    
    max_nodes = 32
    state_enc = StateEncoder(max_nodes=max_nodes)
    act_enc = ActionEncoder(max_nodes=max_nodes)
    
    state_dim = state_enc.observation_dim
    action_dim = act_enc.action_dim
    hidden_dim = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Models
    atk_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    def_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    
    atk_path = "dqn_attacker_clean.pt"
    def_path = "dqn_defender_clean.pt"
    
    if os.path.exists(atk_path):
        atk_net.load_state_dict(torch.load(atk_path, map_location=device))
        print(f"Loaded {atk_path}")
    else:
        print(f"ERROR: {atk_path} missing!")
        return
        
    if os.path.exists(def_path):
        def_net.load_state_dict(torch.load(def_path, map_location=device))
        print(f"Loaded {def_path}")
    else:
        print(f"ERROR: {def_path} missing!")
        return
        
    atk_net.eval()
    def_net.eval()
    
    # 3. Tracking Metrics
    results = {
        "episodes": episodes,
        "attacker_wins": 0,
        "defender_wins": 0, # Timeout
        "episode_lengths": [],
        "attacker_rewards": [],
        "defender_rewards": []
    }
    
    current_atk_streak = 0
    current_def_streak = 0
    max_atk_streak = 0
    max_def_streak = 0
    max_atk_loss_streak = 0
    max_def_loss_streak = 0
    
    # 4. Evaluation Loop
    for episode in range(1, episodes + 1):
        env.reset()
        done = False
        steps = 0
        ep_atk_reward = 0
        ep_def_reward = 0
        
        while not done and steps < max_steps:
            # --- Get Observations ---
            atk_obs = env.get_observation_by_id("atk_1")
            def_obs = env.get_observation_by_id("def_1")
            
            atk_vec = state_enc.encode(atk_obs, "attacker")
            def_vec = state_enc.encode(def_obs, "defender")
            
            atk_mask = act_enc.generate_action_mask(atk_obs, "attacker")
            def_mask = act_enc.generate_action_mask(def_obs, "defender")
            
            # Attacker Action
            atk_tensor = torch.from_numpy(atk_vec).float().unsqueeze(0).to(device)
            atk_mask_tensor = torch.from_numpy(atk_mask).float().to(device)
            atk_idx = atk_net.select_action(atk_tensor, atk_mask_tensor, epsilon=0.0, device=device)
            atk_sorted_nodes = sorted(atk_obs.get("nodes", []), key=lambda x: x["node_id"])
            atk_sorted_ids = [n["node_id"] for n in atk_sorted_nodes]
            atk_action = act_enc.decode_action(atk_idx, atk_sorted_ids, "atk_1")
            
            # Defender Action
            def_tensor = torch.from_numpy(def_vec).float().unsqueeze(0).to(device)
            def_mask_tensor = torch.from_numpy(def_mask).float().to(device)
            def_idx = def_net.select_action(def_tensor, def_mask_tensor, epsilon=0.0, device=device)
            def_sorted_nodes = sorted(def_obs.get("nodes", []), key=lambda x: x["node_id"])
            def_sorted_ids = [n["node_id"] for n in def_sorted_nodes]
            def_action = act_enc.decode_action(def_idx, def_sorted_ids, "def_1")
            
            if episode == 1:
                print(f"\n--- STEP {steps} ---")
                print('Attacker Scanned:', env._state.attacker_scanned)
                print('Attacker Compromised:', env._state.attacker_compromised)
                print('Defender Detections:', env._state.defender_detected_nodes)
                print('Defender Queue:', env._state._detection_queue)
                print('Cooldowns:', env._state._patch_cooldowns)
                print('Attacker Action:', atk_action.action_type.name, getattr(atk_action, 'target_node', 'N/A'))
                print('Defender Action:', def_action.action_type.name, getattr(def_action, 'target_node', 'N/A'))
            
            # --- Step Environment ---
            _, rewards, terminated, info = env.step(atk_action, def_action)
            ep_atk_reward += rewards.get("atk_1", 0.0)
            ep_def_reward += rewards.get("def_1", 0.0)
            steps += 1
            done = terminated
            
        # Determine winner
        # Attacker wins if CRITICAL_ASSET has ROOT privilege
        attacker_won = False
        for node in env._state.graph_manager.get_all_nodes():
            if node.node_type.name == "CRITICAL_ASSET":
                try:
                    if env._state.vulnerability_registry.get_privilege(node.node_id).value == 3: # ROOT
                        attacker_won = True
                        break
                except Exception:
                    pass
                    
        if attacker_won:
            results["attacker_wins"] += 1
            current_atk_streak += 1
            current_def_streak = 0
            if current_atk_streak > max_atk_streak:
                max_atk_streak = current_atk_streak
            if max_def_loss_streak < current_atk_streak:
                max_def_loss_streak = current_atk_streak
        else:
            results["defender_wins"] += 1
            current_def_streak += 1
            current_atk_streak = 0
            if current_def_streak > max_def_streak:
                max_def_streak = current_def_streak
            if max_atk_loss_streak < current_def_streak:
                max_atk_loss_streak = current_def_streak
                
        results["episode_lengths"].append(steps)
        results["attacker_rewards"].append(ep_atk_reward)
        results["defender_rewards"].append(ep_def_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode} | A_WinRate: {(results['attacker_wins'] / episode) * 100:.1f}% | AvgLen: {np.mean(results['episode_lengths'][-100:]):.1f}")
            
    # Compile Final Stats
    final_stats = {
        "episodes": episodes,
        "attacker_win_rate": (results["attacker_wins"] / episodes) * 100,
        "defender_win_rate": (results["defender_wins"] / episodes) * 100,
        "avg_episode_length": float(np.mean(results["episode_lengths"])),
        "avg_attacker_reward": float(np.mean(results["attacker_rewards"])),
        "reward_std": float(np.std(results["attacker_rewards"])),
        "length_std": float(np.std(results["episode_lengths"])),
        "max_win_streak_attacker": max_atk_streak,
        "max_win_streak_defender": max_def_streak,
        "max_loss_streak_attacker": max_atk_loss_streak,
        "max_loss_streak_defender": max_def_loss_streak
    }
    
    with open("clean_cross_eval.json", "w") as f:
        json.dump(final_stats, f, indent=4)
        
    print("\nEvaluation Complete.")
    print(json.dumps(final_stats, indent=2))

if __name__ == "__main__":
    evaluate_equilibrium(1000)
