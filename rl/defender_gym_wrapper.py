import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Tuple

from core.environment import EnvironmentEngine
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from agents.base_agent import BaseAgent
from core.actions import DefenderAction


class DefenderEnv(gym.Env):
    """
    Gymnasium-compatible wrapper placing a DQN Defender against a fixed Attacker policy.
    Maintains a strict black-box boundary; exposes encoded states and discrete actions for the defender.
    """
    
    def __init__(
        self,
        base_env: EnvironmentEngine,
        state_encoder: StateEncoder,
        action_encoder: ActionEncoder,
        attacker_policy: BaseAgent,
        max_steps: int = 50,
        defender_id: str = "def_1",
        attacker_id: str = "atk_1"
    ):
        super(DefenderEnv, self).__init__()
        
        self.base_env = base_env
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.attacker_policy = attacker_policy
        self.max_steps = max_steps
        
        self.defender_id = defender_id
        self.attacker_id = attacker_id
        
        self._step_count = 0
        
        # State space is a 1D vector corresponding to the Defender's encoded view
        self.observation_space = spaces.Box(
            low=-2.0, 
            high=2.0, 
            shape=(self.state_encoder.observation_dim,), 
            dtype=np.float32
        )
        
        # Action space is discrete (Defender has PATCH, ISOLATE, RESET_PRIVILEGE + NO_OP)
        self.action_space = spaces.Discrete(self.action_encoder.action_dim)
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment and returns the initial defender state and mask.
        """
        super().reset(seed=seed)
        self._step_count = 0
        
        # Reset base simulation and get states
        initial_obs_dict = self.base_env.reset()
        
        # Extract defender observation
        def_obs = initial_obs_dict[self.defender_id]
        
        # Encode state for the neural network
        encoded_state = self.state_encoder.encode(def_obs, "defender")
        
        # Generate initial action mask for the defender
        action_mask = self.action_encoder.generate_action_mask(def_obs, "defender")
        
        info = {
            "action_mask": action_mask
        }
        
        return encoded_state, info

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advances the environment by one step acting as the Defender.
        """
        self._step_count += 1
        
        # 1. Get Sorted IDs for deterministic action mapping (Defender view)
        def_obs = self.base_env.get_observation_by_id(self.defender_id)
        sorted_nodes = sorted(def_obs.get("nodes", []), key=lambda x: x["node_id"])
        sorted_ids = [n["node_id"] for n in sorted_nodes]
        
        # 2. Decode Defender Action from network
        defender_action = self.action_encoder.decode_action(
            int(action_index), 
            sorted_ids, 
            self.defender_id
        )
        
        # 3. Get Attacker Action from fixed policy
        atk_obs = self.base_env.get_observation_by_id(self.attacker_id)
        attacker_action = self.attacker_policy.act(atk_obs)
        
        # 4. Step the simulator
        obs_dict, rewards, done, info = self.base_env.step(attacker_action, defender_action)
        
        # 5. Encode next state for defender
        next_def_obs = obs_dict[self.defender_id]
        next_state_vector = self.state_encoder.encode(next_def_obs, "defender")
        
        # 6. Defender Reward
        reward = rewards.get(self.defender_id, 0.0)
        
        # 7. Check terminations
        terminated = done
        truncated = self._step_count >= self.max_steps
        
        # 8. Action mask for next state (Defender view)
        action_mask = self.action_encoder.generate_action_mask(next_def_obs, "defender")
        
        # Pass the mask through info dictionary for safe, aligned access
        info["action_mask"] = action_mask
        
        return next_state_vector, float(reward), terminated, truncated, info

    def render(self):
        """Minimal render."""
        return f"Defender Step: {self._step_count}"
