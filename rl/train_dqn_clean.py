"""
Clean DQN Training Loop for Phase 8.

Responsibility boundaries:
- Orchestrates the training process for the DQN Attacker from SCRATCH.
- Integrates Gym Wrapper, Q-Network, Replay Buffer, and Scheduler.
- Bypasses any historical tainted checkpoints.
- Uses strictly enforced invariant rules in the updated simulator.
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
from agents.greedy_defender import GreedyDefender
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.gym_wrapper import CyberAttackEnv
from rl.q_network import QNetwork
from rl.replay_buffer import ReplayBuffer, EpsilonScheduler

def evaluate(env, policy_net, device, episodes=100) -> dict:
    """Evaluate current policy deterministically without exploration."""
    wins = 0
    rewards = []
    lengths = []
    
    policy_net.eval()
    
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            mask = info["action_mask"]
            mask_tensor = torch.from_numpy(mask).float().to(device)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # Epsilon = 0 for deterministic evaluation
            action_idx = policy_net.select_action(state_tensor, mask_tensor, epsilon=0.0, device=device)
            
            state, reward, term, trunc, info = env.step(action_idx)
            done = term or trunc
            ep_reward += reward
            steps += 1
            
        # Check if the terminal state was a successful compromise
        if env.base_env._state.timestamp < 50 and any(
            v == env.base_env._state.vulnerability_registry.get_privilege(n.node_id).name == "ROOT"
            for n in env.base_env.graph_manager.get_all_nodes() if n.node_type.name == "CRITICAL_ASSET" for v in ["ROOT"]
        ):
            # Simplification: We can check Termination reason if available in base_env
            pass # Use generic win check below
            
        termination = info.get("termination_reason")
        # In our gym wrapper, we passed info directly through from base_env's step pipeline
        # Actually base_env.step returns info, but the wrapper currently just injects action_mask
        # Let's peek at the base_env state for a definitive win
        win = 0
        for node in env.base_env._state.graph_manager.get_all_nodes():
            if node.node_type.name == "CRITICAL_ASSET":
                try:
                    if env.base_env._state.vulnerability_registry.get_privilege(node.node_id).value == 3: # ROOT
                        win = 1
                except Exception:
                    pass
        
        wins += win
        rewards.append(ep_reward)
        lengths.append(steps)
        
    policy_net.train()
    
    return {
        "win_rate": (wins / episodes) * 100,
        "avg_reward": np.mean(rewards),
        "avg_length": np.mean(lengths)
    }


def train_clean(
    max_episodes: int = 500,
    max_steps: int = 50,
    batch_size: int = 64,
    target_update_freq: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    decay_steps: int = 20000,
    eval_freq: int = 100
):
    print(f"Starting Clean DQN Baseline Training (Episodes: {max_episodes})")
    
    # 1. Setup Determinism
    torch.manual_seed(42)
    np.random.seed(42)
    rng = CentralizedRNG(seed=42)
    
    # 2. Setup Environment
    max_nodes = 32
    base_env = EnvironmentEngine(rng)
    
    state_enc = StateEncoder(max_nodes=max_nodes)
    act_enc = ActionEncoder(max_nodes=max_nodes)
    defender = GreedyDefender("def_1")
    
    env = CyberAttackEnv(
        base_env=base_env,
        state_encoder=state_enc,
        action_encoder=act_enc,
        defender_policy=defender,
        max_steps=max_steps
    )
    
    # 3. Architectures
    state_dim = state_enc.observation_dim
    action_dim = act_enc.action_dim
    hidden_dim = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Freshly initialized Networks
    policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    # Faster decay since we have fewer episodes (e.g., 500 eps = ~25k steps max)
    epsilon_scheduler = EpsilonScheduler(start=1.0, end=0.05, decay_steps=decay_steps)
    
    # Fresh Replay Buffer
    replay_buffer = ReplayBuffer(100000, state_dim, action_dim, device)
    
    global_step = 0
    history = []
    
    for episode in range(1, max_episodes + 1):
        state, info = env.reset()
        mask = info["action_mask"]
        
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            mask_tensor = torch.from_numpy(mask).float()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            # Action Selection
            epsilon = epsilon_scheduler.value(global_step)
            action_idx = policy_net.select_action(state_tensor, mask_tensor, epsilon, device)
            
            # Step
            next_state, reward, term, trunc, n_info = env.step(action_idx)
            done = term or trunc
            next_mask = n_info["action_mask"]
            
            # Store
            replay_buffer.add(state, action_idx, reward, next_state, next_mask, done)
            
            state = next_state
            mask = next_mask
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Optimization
            if len(replay_buffer) >= 1000:
                b_states, b_actions, b_rewards, b_next_states, b_next_masks, b_dones = replay_buffer.sample(batch_size)
                
                # Q(s, a)
                q_values = policy_net(b_states)
                current_q = q_values.gather(1, b_actions.unsqueeze(1)).squeeze()
                
                # Target: r + gamma * max Q'(s', a')
                with torch.no_grad():
                    next_q_values = target_net(b_next_states)
                    masked_next_q = next_q_values.clone()
                    masked_next_q[b_next_masks == 0] = -1e9
                    max_next_q = masked_next_q.max(dim=1)[0]
                    target_q = b_rewards + gamma * max_next_q * (1 - b_dones)
                
                # Loss & Backprop
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()
                
            # Sync Target
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
        # Logging & Evalluation
        if episode % eval_freq == 0 or episode == 1:
            eval_metrics = evaluate(env, policy_net, device, episodes=50)
            print(f"--- Episode {episode} ---")
            print(f"Epsilon: {epsilon:.3f} | Buffer Size: {len(replay_buffer)}")
            print(f"Eval Win Rate: {eval_metrics['win_rate']:.1f}%")
            print(f"Eval Avg Reward: {eval_metrics['avg_reward']:.2f}")
            print(f"Eval Avg Steps: {eval_metrics['avg_length']:.1f}")
            
            history.append({
                "episode": episode,
                "win_rate": eval_metrics["win_rate"],
                "avg_reward": eval_metrics["avg_reward"],
                "avg_steps": eval_metrics["avg_length"]
            })
            
    # Save Final Clean Model
    save_path = "dqn_attacker_clean.pt"
    torch.save(policy_net.state_dict(), save_path)
    
    with open("dqn_attacker_clean_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining Complete. Strict Baseline isolated in `{save_path}`.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DQN Clean Attacker Training")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    
    train_clean(max_episodes=args.episodes)
