import sys
sys.path.append(".")

from core.environment import EnvironmentEngine
from encoding.state_encoder import StateEncoder
from encoding.action_encoder import ActionEncoder
from agents.greedy_attacker import GreedyAttacker
from rl.defender_gym_wrapper import DefenderEnv
import numpy as np

from utils.rng import CentralizedRNG

def run_verification():
    rng = CentralizedRNG()
    env_engine = EnvironmentEngine(rng)
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    attacker = GreedyAttacker("atk_1")
    
    env = DefenderEnv(
        base_env=env_engine,
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        attacker_policy=attacker,
        max_steps=50
    )
    
    wins = 0
    losses = 0
    total_reward = 0
    
    for ep in range(50):
        state, info = env.reset()
        ep_reward = 0
        
        for step in range(50):
            # Random action from valid mask
            mask = info["action_mask"]
            valid_actions = np.where(mask == 1.0)[0]
            action = np.random.choice(valid_actions)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                reason = info.get("termination_reason", "UNKNOWN")
                if reason == "CRITICAL_COMPROMISED":
                    losses += 1
                else:
                    wins += 1
                break
                
        total_reward += ep_reward
        
    print(f"Random Defender vs Greedy Attacker (50 episodes)")
    print(f"Defense Successes: {wins}")
    print(f"Defense Failures: {losses}")
    print(f"Average Reward: {total_reward / 50:.2f}")

if __name__ == "__main__":
    run_verification()
