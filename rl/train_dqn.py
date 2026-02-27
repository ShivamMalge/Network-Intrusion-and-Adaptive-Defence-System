"""
DQN Training Loop Phase 6A.4.

Responsibility boundaries:
- Orchestrates the training process for the DQN Attacker.
- Integrates Gym Wrapper, Q-Network, Replay Buffer, and Scheduler.
- Implements the standard DQN backpropagation and target network synchronization.
- Saves the trained model states.
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
from agents.greedy_defender import GreedyDefender
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.gym_wrapper import CyberAttackEnv
from rl.q_network import QNetwork
from rl.replay_buffer import ReplayBuffer, EpsilonScheduler
from graph.node import Node, NodeType
from graph.edge import Edge
from vulnerabilities.vulnerability import Vulnerability
from vulnerabilities.privilege_model import PrivilegeLevel


def build_training_topology(env: EnvironmentEngine):
    """
    Sets up a fixed benchmark topology for training.
    DMZ -> INTERNAL -> DATA (Critical)
    """
    gm = env._state.graph_manager
    reg = env._state.vulnerability_registry
    
    # Nodes
    n1 = Node("dmz", NodeType.WORKSTATION)
    n2 = Node("internal", NodeType.SERVER)
    n3 = Node("data", NodeType.CRITICAL_ASSET)
    
    for n in [n1, n2, n3]:
        gm.add_node(n)
        reg.register_node(n.node_id)
        # Add visible vulnerabilities to allow progress
        reg.add_vulnerability(n.node_id, Vulnerability(f"V_{n.node_id}", 8.0, PrivilegeLevel.NONE, False))
        
    # Edges
    gm.add_edge(Edge("dmz", "internal"))
    gm.add_edge(Edge("internal", "data"))
    
    # Attacker starts with dmz knowledge
    env._state.add_scanned_node("dmz")


def train(
    max_episodes: int = 500,
    max_steps: int = 50,
    train_start_threshold: int = 1000,
    batch_size: int = 64,
    target_update_freq: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    decay_steps: int = 10000
):
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
    
    # 3. Hyperparameters (Local override from arguments)
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
    replay_buffer = ReplayBuffer(100000, state_dim, device)
    
    # 5. Training Loop
    global_step = 0
    reward_history = deque(maxlen=50)
    
    print("Starting DQN Training...")
    
    for episode in range(1, max_episodes + 1):
        # Reset and setup topology
        raw_obs, _ = env.reset()
        build_training_topology(env.base_env)
        # Re-reset or manually refresh obs to see nodes
        atk_obs = env.base_env.get_observation_by_id(env.attacker_id)
        state = env.state_encoder.encode(atk_obs, "attacker")
        
        episode_reward = 0
        
        for step in range(env.max_steps):
            # Get Action Mask
            mask = env.action_encoder.generate_action_mask(atk_obs, "attacker")
            mask_tensor = torch.from_numpy(mask).float()
            
            # Select Action
            epsilon = epsilon_scheduler.value(global_step)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            action = policy_net.select_action(state_tensor, mask_tensor, epsilon, device)
            
            # Step Environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Save Experience
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            global_step += 1
            
            # Update last known atk_obs for next mask gen
            atk_obs = env.base_env.get_observation_by_id(env.attacker_id)
            
            # 6. Optimization
            if len(replay_buffer) >= train_start_threshold:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                
                # Q(s, a)
                q_values = policy_net(b_states)
                current_q = q_values.gather(1, b_actions.unsqueeze(1)).squeeze()
                
                # Target: r + gamma * max Q'(s', a')
                with torch.no_grad():
                    next_q_values = target_net(b_next_states)
                    max_next_q = next_q_values.max(dim=1)[0]
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
                
            if done:
                break
                
        reward_history.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = sum(reward_history) / len(reward_history)
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f} | Steps: {global_step}")

    # 8. Save Model
    save_path = "dqn_attacker.pt"
    torch.save(policy_net.state_dict(), save_path)
    print(f"Training Complete. Model saved to {save_path}")


if __name__ == "__main__":
    train()
