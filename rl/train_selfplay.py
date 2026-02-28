"""
Stable Self-Play Training Phase 6C.
Alternating Freeze-Train Cycles for DQN Attacker vs DQN Defender.

Safeguards:
- Replay Buffer Isolation (reset per phase).
- Frozen Opponent Determinism (eval mode, epsilon=0, no_grad).
- Short Cycles (300-500 episodes).
- Distinct Target Networks per phase.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from collections import deque

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.gym_wrapper import CyberAttackEnv
from rl.defender_gym_wrapper import DefenderEnv
from rl.q_network import QNetwork
from rl.replay_buffer import ReplayBuffer, EpsilonScheduler

class DQNAgent:
    """A wrapper to use a Q-Network as a fixed policy for the opponent."""
    def __init__(self, model, action_encoder, role="attacker"):
        self.model = model
        self.action_encoder = action_encoder
        self.role = role
        self.agent_id = "atk_1" if role == "attacker" else "def_1"

    def act(self, observation):
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # 1. Encode state
        encoded_state = StateEncoder(max_nodes=32).encode(observation, self.role)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)
        
        # 2. Generate mask
        mask = self.action_encoder.generate_action_mask(observation, self.role)
        mask_tensor = torch.from_numpy(mask).float().to(device)
        
        # 3. Select greedy action (Epsilon=0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            masked_q = q_values.clone()
            masked_q[0, mask_tensor == 0] = -float('inf')
            action_idx = masked_q.argmax(dim=1).item()
            
        # 4. Decode to Action object
        nodes = sorted(observation.get("nodes", []), key=lambda x: x["node_id"])
        node_ids = [n["node_id"] for n in nodes]
        
        return self.action_encoder.decode_action(action_idx, node_ids, self.agent_id)

def evaluate(base_env, attacker_policy, defender_policy, episodes=100):
    """Run evaluation matches using base_env directly."""
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
        "avg_reward_atk": np.mean(rewards),
        "avg_length": np.mean(lengths)
    }

def train_phase(env, policy_net, target_net, episodes, device, lr=5e-4, gamma=0.99, batch_size=64):
    """Generic training loop for one agent against a frozen opponent."""
    # Reset Optimization State for each phase
    policy_net.train()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(50000, policy_net.state_dim, policy_net.action_dim, device)
    scheduler = EpsilonScheduler(start=0.3, end=0.05, decay_steps=episodes * 20)
    
    global_step = 0
    target_update_freq = 500
    
    for ep in range(episodes):
        state, info = env.reset()
        mask = info["action_mask"]
        done = False
        
        while not done:
            mask_tensor = torch.from_numpy(mask).float().to(device)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # Select action
            epsilon = scheduler.value(global_step)
            action_idx = policy_net.select_action(state_tensor, mask_tensor, epsilon, device)
            
            # Step
            next_state, reward, term, trunc, n_info = env.step(action_idx)
            done = term or trunc
            next_mask = n_info["action_mask"]
            
            # Add to buffer
            buffer.add(state, action_idx, reward, next_state, next_mask, done)
            
            state = next_state
            mask = next_mask
            global_step += 1
            
            # Optimization
            if len(buffer) > 1000:
                b_s, b_a, b_r, b_ns, b_nm, b_d = buffer.sample(batch_size)
                
                # Q(s, a)
                q_vals = policy_net(b_s)
                curr_q = q_vals.gather(1, b_a.unsqueeze(1)).squeeze()
                
                # Target: r + gamma * max Q'(s', a') with mask
                with torch.no_grad():
                    n_q_vals = target_net(b_ns)
                    masked_n_q = n_q_vals.clone()
                    masked_n_q[b_nm == 0] = -1e9
                    max_n_q = masked_n_q.max(dim=1)[0]
                    target_q = b_r + gamma * max_n_q * (1 - b_d)
                    
                loss = F.mse_loss(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()
                
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

def run_self_play(max_cycles=3, episodes_per_phase=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_nodes = 32
    state_enc = StateEncoder(max_nodes=max_nodes)
    act_enc = ActionEncoder(max_nodes=max_nodes)
    state_dim = state_enc.observation_dim
    action_dim = act_enc.action_dim
    hidden_dim = 256
    
    # 1. Models
    print("Loading initial checkpoints...")
    atk_policy = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    atk_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    atk_policy.load_state_dict(torch.load("dqn_attacker.pt", map_location=device))
    
    def_policy = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    def_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    def_policy.load_state_dict(torch.load("dqn_defender.pt", map_location=device))
    
    rng = CentralizedRNG(seed=42)
    base_env = EnvironmentEngine(rng)
    
    # Static evaluators (wrappers around the networks)
    atk_fixed_policy = DQNAgent(atk_policy, act_enc, "attacker")
    def_fixed_policy = DQNAgent(def_policy, act_enc, "defender")
    
    metrics = []
    
    # 2. Stage A: Baseline
    print("\nStage A: Baseline Evaluation")
    baseline = evaluate(base_env, atk_fixed_policy, def_fixed_policy, episodes=100)
    print(f"Baseline -> Win Rate: {baseline['win_rate']:.1f}%, Avg Len: {baseline['avg_length']:.1f}")
    metrics.append({"cycle": 0, "stage": "baseline", **baseline})
    
    # 3. Stage B: Cycles
    for cycle in range(1, max_cycles + 1):
        print(f"\n--- Cycle {cycle} ---")
        
        # Attacker Train Phase
        print(f"Cycle {cycle} Phase 1: Training Attacker...")
        atk_env = CyberAttackEnv(base_env, state_enc, act_enc, def_fixed_policy)
        train_phase(atk_env, atk_policy, atk_target, episodes_per_phase, device)
        
        # Defender Train Phase
        print(f"Cycle {cycle} Phase 2: Training Defender...")
        def_env = DefenderEnv(base_env, state_enc, act_enc, atk_fixed_policy)
        train_phase(def_env, def_policy, def_target, episodes_per_phase, device)
        
        # End of Cycle Eval
        print(f"Cycle {cycle} Evaluation...")
        result = evaluate(base_env, atk_fixed_policy, def_fixed_policy, episodes=100)
        print(f"Result -> Win Rate: {result['win_rate']:.1f}%, Avg Len: {result['avg_length']:.1f}")
        metrics.append({"cycle": cycle, "stage": "self-play", **result})
        
        # Save checkpoints
        torch.save(atk_policy.state_dict(), f"dqn_attacker_cycle_{cycle}.pt")
        torch.save(def_policy.state_dict(), f"dqn_defender_cycle_{cycle}.pt")

    # Final save
    torch.save(atk_policy.state_dict(), "dqn_attacker_selfplay.pt")
    torch.save(def_policy.state_dict(), "dqn_defender_selfplay.pt")
    
    with open("selfplay_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\nTraining Complete. Metrics saved to selfplay_metrics.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    
    run_self_play(max_cycles=args.cycles, episodes_per_phase=args.episodes)
