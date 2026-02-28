import os
import sys
import torch
import numpy as np

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from rl.q_network import QNetwork
from rl.gym_wrapper import CyberAttackEnv
from rl.defender_gym_wrapper import DefenderEnv
from agents.greedy_defender import GreedyDefender
from agents.greedy_attacker import GreedyAttacker

class DQNAgent:
    def __init__(self, model, action_encoder, role="attacker"):
        self.model = model
        self.action_encoder = action_encoder
        self.role = role
        self.agent_id = "atk_1" if role == "attacker" else "def_1"

    def act(self, observation):
        # This mirrors the logic in the training loops but for a BaseAgent compatible interface
        # In self-play, we need this to pass to the Gym wrappers
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # 1. Encode state
        encoded_state = StateEncoder(max_nodes=32).encode(observation, self.role)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)
        
        # 2. Generate mask
        mask = self.action_encoder.generate_action_mask(observation, self.role)
        mask_tensor = torch.from_numpy(mask).float().to(device)
        
        # 3. Select greedy action
        with torch.no_grad():
            q_values = self.model(state_tensor)
            masked_q = q_values.clone()
            masked_q[0, mask_tensor == 0] = -float('inf')
            action_idx = masked_q.argmax(dim=1).item()
            
        # 4. Decode to Action object
        nodes = sorted(observation.get("nodes", []), key=lambda x: x["node_id"])
        node_ids = [n["node_id"] for n in nodes]
        
        return self.action_encoder.decode_action(action_idx, node_ids, self.agent_id)

def verify():
    print("--- Phase 6C Sanity Check: Verifying Checkpoints ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_nodes = 32
    state_enc = StateEncoder(max_nodes=max_nodes)
    act_enc = ActionEncoder(max_nodes=max_nodes)
    
    state_dim = state_enc.observation_dim
    action_dim = act_enc.action_dim
    hidden_dim = 256
    
    # 1. Load Attacker
    print("Checking dqn_attacker.pt...")
    atk_model = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    try:
        atk_model.load_state_dict(torch.load("dqn_attacker.pt", map_location=device))
        print("SUCCESS: Attacker model loaded.")
    except Exception as e:
        print(f"FAILED: Attacker load error: {e}")
        return

    # 2. Load Defender
    print("Checking dqn_defender.pt...")
    def_model = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    try:
        def_model.load_state_dict(torch.load("dqn_defender.pt", map_location=device))
        print("SUCCESS: Defender model loaded.")
    except Exception as e:
        print(f"FAILED: Defender load error: {e}")
        return

    # 3. Test Determinism (Attacker)
    rng = CentralizedRNG(seed=42)
    base_env = EnvironmentEngine(rng)
    obs_dict = base_env.reset()
    atk_obs = obs_dict["atk_1"]
    
    agent = DQNAgent(atk_model, act_enc, role="attacker")
    
    print("\nVerifying Attacker Determinism...")
    a1 = agent.act(atk_obs)
    a2 = agent.act(atk_obs)
    if a1.action_type == a2.action_type and a1.target_node == a2.target_node:
        print(f"SUCCESS: Attacker is deterministic. Initial Action: {a1.action_type.name} on {a1.target_node}")
    else:
        print("FAILED: Attacker is non-deterministic!")
        return

    # 4. Test Determinism (Defender)
    print("\nVerifying Defender Determinism...")
    def_obs = obs_dict["def_1"]
    agent = DQNAgent(def_model, act_enc, role="defender")
    
    d1 = agent.act(def_obs)
    d2 = agent.act(def_obs)
    if d1.action_type == d2.action_type and d1.target_node == d2.target_node:
        print(f"SUCCESS: Defender is deterministic. Initial Action: {d1.action_type.name} on {d1.target_node}")
    else:
        print("FAILED: Defender is non-deterministic!")
        return

    print("\n--- ALL SANITY CHECKS PASSED ---")

if __name__ == "__main__":
    verify()
