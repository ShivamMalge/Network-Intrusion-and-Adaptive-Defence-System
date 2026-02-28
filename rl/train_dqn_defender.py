"""
DQN Training Loop Phase 6B.
Defender vs Fixed GreedyAttacker.

Responsibility boundaries:
- Orchestrates the training process for the DQN Defender.
- Integrates DefenderGymWrapper, Q-Network, Replay Buffer, and Scheduler.
- Implements masked Q-learning for the defender.
- Saves the trained defender model states.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from agents.greedy_attacker import GreedyAttacker
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.defender_gym_wrapper import DefenderEnv
from rl.q_network import QNetwork
from rl.replay_buffer import ReplayBuffer, EpsilonScheduler


def train_defender(
    max_episodes: int = 1000,
    max_steps: int = 50,
    train_start_threshold: int = 1000,
    batch_size: int = 64,
    target_update_freq: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    decay_steps: int = 40000
):
    print(f"Starting Defender DQN Training (Episodes: {max_episodes})")
    
    # 1. Setup Determinism
    torch.manual_seed(42)
    np.random.seed(42)
    rng = CentralizedRNG(seed=42)
    
    # 2. Setup Environment
    max_nodes = 32
    base_env = EnvironmentEngine(rng)
    
    state_enc = StateEncoder(max_nodes=max_nodes)
    act_enc = ActionEncoder(max_nodes=max_nodes)
    attacker = GreedyAttacker("atk_1")
    
    env = DefenderEnv(
        base_env=base_env,
        state_encoder=state_enc,
        action_encoder=act_enc,
        attacker_policy=attacker,
        max_steps=max_steps
    )
    
    # 3. Hyperparameters
    state_dim = state_enc.observation_dim
    action_dim = act_enc.action_dim
    hidden_dim = 256
    
    # Epsilon Scheduler
    epsilon_scheduler = EpsilonScheduler(start=1.0, end=0.05, decay_steps=decay_steps)
    
    # 4. Networks and Buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(100000, state_dim, action_dim, device)
    
    # 5. Training Loop
    global_step = 0
    
    # Rolling Statistics (last 50 episodes)
    last_50_rewards = deque(maxlen=50)
    last_50_lengths = deque(maxlen=50)
    last_50_successes = deque(maxlen=50) # Non-compromised episodes
    
    for episode in range(1, max_episodes + 1):
        state, info = env.reset()
        mask = info["action_mask"]
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            mask_tensor = torch.from_numpy(mask).float()
            
            # Select Action
            epsilon = epsilon_scheduler.value(global_step)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            # QNetwork.select_action handles the masking
            action_idx = policy_net.select_action(state_tensor, mask_tensor, epsilon, device)
            
            # Step Environment
            next_state, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            next_mask = info["action_mask"]
            
            # Save Experience
            replay_buffer.add(state, action_idx, reward, next_state, next_mask, done)
            
            state = next_state
            mask = next_mask
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            # 6. Optimization
            if len(replay_buffer) >= train_start_threshold:
                b_states, b_actions, b_rewards, b_next_states, b_next_masks, b_dones = replay_buffer.sample(batch_size)
                
                # Q(s, a)
                q_values = policy_net(b_states)
                current_q = q_values.gather(1, b_actions.unsqueeze(1)).squeeze()
                
                # Target: r + gamma * max Q'(s', a') with masking
                with torch.no_grad():
                    next_q_values = target_net(b_next_states)
                    masked_next_q = next_q_values.clone()
                    masked_next_q[b_next_masks == 0] = -1e9
                    max_next_q = masked_next_q.max(dim=1)[0]
                    target_q = b_rewards + gamma * max_next_q * (1 - b_dones)
                
                # Loss
                loss = F.mse_loss(current_q, target_q)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()
                
            # 7. Update Target Network
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
        # Record Success condition (Defender perspective)
        # Success = Not CRITICAL_COMPROMISED
        reason = info.get("termination_reason", "TIMEOUT")
        success = 1 if reason != "CRITICAL_COMPROMISED" else 0
        last_50_successes.append(success)
        
        last_50_rewards.append(episode_reward)
        last_50_lengths.append(episode_steps)
        
        if episode % 50 == 0:
            avg_reward = np.mean(last_50_rewards)
            avg_len = np.mean(last_50_lengths)
            success_rate = np.mean(last_50_successes) * 100
            
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1f}% | Avg Len: {avg_len:.1f} | Epsilon: {epsilon:.3f}")

    save_path = "dqn_defender.pt"
    torch.save(policy_net.state_dict(), save_path)
    print(f"Training Complete. Model saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Defender DQN Training")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    
    train_defender(max_episodes=args.episodes)
