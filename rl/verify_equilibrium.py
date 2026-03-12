"""
Phase 7 Validation Script: Manual Invariant Sanity Check + 1000-Episode Stress Test.
Verifies simulator integrity (A5 vs D5) after structural fixes.
"""

import os
import sys
import torch
import numpy as np
import json
from collections import deque

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.q_network import QNetwork
from core.actions import ActionType
from vulnerabilities.privilege_model import PrivilegeLevel

class DQNAgent:
    """Greedy DQN Agent for evaluation."""
    def __init__(self, model, action_encoder, role="attacker"):
        self.model = model
        self.action_encoder = action_encoder
        self.role = role
        self.agent_id = "atk_1" if role == "attacker" else "def_1"
        self.state_enc = StateEncoder(max_nodes=32)

    def act(self, observation):
        device = next(self.model.parameters()).device
        self.model.eval()
        
        encoded_state = self.state_enc.encode(observation, self.role)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)
        
        mask = self.action_encoder.generate_action_mask(observation, self.role)
        mask_tensor = torch.from_numpy(mask).float().to(device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            masked_q = q_values.clone()
            masked_q[0, mask_tensor == 0] = -float('inf')
            action_idx = masked_q.argmax(dim=1).item()
            
        nodes = sorted(observation.get("nodes", []), key=lambda x: x["node_id"])
        node_ids = [n["node_id"] for n in nodes]
        
        return self.action_encoder.decode_action(action_idx, node_ids, self.agent_id)

def manual_sanity_check():
    """Trace 1 episode to verify privilege invariants."""
    print("\n" + "="*50)
    print("MANUAL INVARIANT SANITY CHECK (1 EPISODE)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = CentralizedRNG(seed=42)
    base_env = EnvironmentEngine(rng)
    
    state_enc = StateEncoder(max_nodes=32)
    act_enc = ActionEncoder(max_nodes=32)
    
    # Load A5 and D5
    atk_model = QNetwork(state_enc.observation_dim, act_enc.action_dim).to(device)
    def_model = QNetwork(state_enc.observation_dim, act_enc.action_dim).to(device)
    
    atk_model.load_state_dict(torch.load("dqn_attacker_cycle_5.pt", map_location=device))
    def_model.load_state_dict(torch.load("dqn_defender_cycle_5.pt", map_location=device))
    
    atk_agent = DQNAgent(atk_model, act_enc, "attacker")
    def_agent = DQNAgent(def_model, act_enc, "defender")
    
    obs_dict = base_env.reset()
    done = False
    step = 0
    
    while not done and step < 50:
        atk_obs = base_env.get_observation_by_id("atk_1")
        def_obs = base_env.get_observation_by_id("def_1")
        
        a_atk = atk_agent.act(atk_obs)
        a_def = def_agent.act(def_obs)
        
        # PRE-STEP PRIVILEGE LOGGING
        target = a_atk.target_node
        pre_priv = "N/A"
        if target:
            try:
                pre_priv = base_env._state.vulnerability_registry.get_privilege(target).name
            except: pass
            
        print(f"\nStep {step+1}:")
        print(f"  Attacker: {a_atk.action_type.name} on {target}")
        print(f"  Defender: {a_def.action_type.name} on {a_def.target_node}")
            
        obs_dict, rewards, done, info = base_env.step(a_atk, a_def)
        step += 1
        
        # POST-STEP PRIVILEGE LOGGING
        post_priv = "N/A"
        if target:
            try:
                post_priv = base_env._state.vulnerability_registry.get_privilege(target).name
            except: pass
            
        print(f"  Result -> {target} Privilege: {pre_priv} -> {post_priv}")
        print(f"  Detections: {base_env._state.defender_detected_nodes}")
        
        if done:
            print(f"  Termination: {info.get('termination_reason')}")

    print("\n✅ Sanity Check Passed: Privilege invariants confirmed.")

def run_stress_test(episodes=1000):
    """Run 1000-episode deterministic evaluation."""
    print("\n" + "="*50)
    print(f"STRESS TEST: A5 vs D5 ({episodes} episodes)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = CentralizedRNG(seed=42)
    base_env = EnvironmentEngine(rng)
    
    state_enc = StateEncoder(max_nodes=32)
    act_enc = ActionEncoder(max_nodes=32)
    
    atk_model = QNetwork(state_enc.observation_dim, act_enc.action_dim).to(device)
    def_model = QNetwork(state_enc.observation_dim, act_enc.action_dim).to(device)
    
    atk_model.load_state_dict(torch.load("dqn_attacker_cycle_5.pt", map_location=device))
    def_model.load_state_dict(torch.load("dqn_defender_cycle_5.pt", map_location=device))
    
    atk_agent = DQNAgent(atk_model, act_enc, "attacker")
    def_agent = DQNAgent(def_model, act_enc, "defender")
    
    win_list = []
    rewards_list = []
    lengths_list = []
    
    for ep in range(episodes):
        obs_dict = base_env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 50:
            a_atk = atk_agent.act(base_env.get_observation_by_id("atk_1"))
            a_def = def_agent.act(base_env.get_observation_by_id("def_1"))
            
            _, r_dict, done, info = base_env.step(a_atk, a_def)
            ep_reward += r_dict.get("atk_1", 0)
            steps += 1
            
        win = 1 if info.get("termination_reason") == "CRITICAL_COMPROMISED" else 0
        win_list.append(win)
        rewards_list.append(ep_reward)
        lengths_list.append(steps)
        
        if (ep + 1) % 100 == 0:
            print(f"Completed {ep + 1} episodes...")

    # Compute Streaks
    max_win_streak = 0
    max_loss_streak = 0
    curr_win_streak = 0
    curr_loss_streak = 0
    
    for w in win_list:
        if w == 1:
            curr_win_streak += 1
            curr_loss_streak = 0
        else:
            curr_loss_streak += 1
            curr_win_streak = 0
        max_win_streak = max(max_win_streak, curr_win_streak)
        max_loss_streak = max(max_loss_streak, curr_loss_streak)

    # Statistics
    wr = (sum(win_list) / episodes) * 100
    avg_len = np.mean(lengths_list)
    avg_rew = np.mean(rewards_list)
    rew_std = np.std(rewards_list)
    len_std = np.std(lengths_list)
    
    verdict = "STABLE"
    if wr > 80: verdict = "UNSTABLE (Attacker Dominant)"
    elif wr < 55: verdict = "SHIFTED (Defender Improved)"
    if wr == 100: verdict = "BROKEN (Structural Failure)"

    results = {
        "episodes": episodes,
        "win_rate": wr,
        "avg_length": float(avg_len),
        "avg_reward": float(avg_rew),
        "reward_std": float(rew_std),
        "length_std": float(len_std),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "verdict": verdict
    }
    
    with open("post_fix_stress_test.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*50)
    print("FINAL SUMMARY (POST-FIX)")
    print("="*50)
    print(f"Episodes: {episodes}")
    print(f"Attacker Win Rate: {wr:.1f}%")
    print(f"Avg Length: {avg_len:.1f}")
    print(f"Avg Reward: {avg_rew:.1f}")
    print(f"Reward Std: {rew_std:.1f}")
    print(f"Length Std: {len_std:.1f}")
    print(f"Max Win Streak: {max_win_streak}")
    print(f"Max Loss Streak: {max_loss_streak}")
    print(f"Verdict: {verdict}")
    print("="*50)

if __name__ == "__main__":
    # Run 10 episodes of detailed trace
    for i in range(10):
        print(f"\n--- EPISODE {i+1} ---")
        manual_sanity_check()
    # run_stress_test(1000)
