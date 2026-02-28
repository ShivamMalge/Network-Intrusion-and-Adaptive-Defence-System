"""
Gym-Compatible Wrapper Phase 6A.

Responsibility boundaries:
- Conforms to Gymnasium (gym.Env) API.
- Wraps EnvironmentEngine for single-agent (Attacker) training.
- Handles fixed Defender policy internally.
- Integrates State and Action Encoders for tensor-based I/O.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from core.environment import EnvironmentEngine
from agents.base_agent import BaseAgent
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from core.actions import AttackerAction, DefenderAction, ActionType


class CyberAttackEnv(gym.Env):
    """
    Gymnasium environment for cyber attack simulation.
    Designed for Attacker training with a fixed Defender agent.
    """
    
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self, 
        base_env: EnvironmentEngine, 
        state_encoder: StateEncoder, 
        action_encoder: ActionEncoder,
        defender_policy: BaseAgent,
        max_steps: int = 50,
        attacker_id: str = "atk_1",
        defender_id: str = "def_1"
    ):
        super().__init__()
        self.base_env = base_env
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.defender_policy = defender_policy
        self.max_steps = max_steps
        self.attacker_id = attacker_id
        self.defender_id = defender_id
        
        # Current observation from the simulation
        self._last_obs: Dict[str, Any] = {}
        self._step_count = 0

        # Define spaces
        # Observation space: Encoded state vector
        obs_dim = self.state_encoder.observation_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: Discrete indices
        self.action_space = spaces.Discrete(self.action_encoder.action_dim)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment.
        Note: We prioritize base_env's CentralizedRNG but support gym seed if needed.
        """
        super().reset(seed=seed)
        
        # Reset the underlying simulation
        # Note: In our architecture, EnvironmentEngine.reset() re-creates the state
        # but we might want it to use the same RNG sequence.
        self._last_obs = self.base_env.reset()
        self._step_count = 0
        
        # The base_env might need topology setup after reset in many benchmark cases,
        # but here we assume the base_env state is ready or configured by the caller.
        
        # Encode initial attacker observation
        # obs dictionary from reset() contains {"atk_1": ..., "def_1": ...}
        # But we use the helper to be sure we get the right one
        atk_obs = self.base_env.get_observation_by_id(self.attacker_id)
        encoded_state = self.state_encoder.encode(atk_obs, "attacker")
        
        # 3. Generate initial action mask
        action_mask = self.action_encoder.generate_action_mask(atk_obs, "attacker")
        
        info = {
            "action_mask": action_mask
        }
        
        return encoded_state, info
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advances the environment by one step.
        """
        self._step_count += 1
        
        # 1. Get Sorted IDs for deterministic action mapping
        # We must get them from the current attacker observation
        atk_obs = self.base_env.get_observation_by_id(self.attacker_id)
        sorted_nodes = sorted(atk_obs.get("nodes", []), key=lambda x: x["node_id"])
        sorted_ids = [n["node_id"] for n in sorted_nodes]
        
        # 2. Decode Attacker Action
        attacker_action = self.action_encoder.decode_action(
            action_index, 
            sorted_ids, 
            self.attacker_id
        )
        
        # 3. Get Defender Action from fixed policy
        def_obs = self.base_env.get_observation_by_id(self.defender_id)
        defender_action = self.defender_policy.act(def_obs)
        
        # 4. Step the simulator
        obs_dict, rewards, done, info = self.base_env.step(attacker_action, defender_action)
        
        # 5. Encode next state for attacker
        next_atk_obs = obs_dict[self.attacker_id]
        next_state_vector = self.state_encoder.encode(next_atk_obs, "attacker")
        
        # 6. Attacker Reward
        reward = rewards.get(self.attacker_id, 0.0)
        
        # 7. Check terminations
        terminated = done
        truncated = self._step_count >= self.max_steps
        
        # 8. Action mask for next state
        action_mask = self.action_encoder.generate_action_mask(next_atk_obs, "attacker")
        
        # Add mask to info for the agent
        info["action_mask"] = action_mask
        
        return next_state_vector, float(reward), terminated, truncated, info

    def render(self):
        """Minimal render."""
        return f"Step: {self._step_count}, Attacker Knowledge: {len(self.base_env._state.attacker_scanned)} nodes"
