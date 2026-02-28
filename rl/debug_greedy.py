import sys
sys.path.append(".")

from core.environment import EnvironmentEngine
from agents.greedy_attacker import GreedyAttacker
from agents.random_defender import RandomDefender
from utils.rng import CentralizedRNG

def debug_greedy():
    rng = CentralizedRNG()
    env = EnvironmentEngine(rng)
    atk = GreedyAttacker("atk_1")
    def_agent = RandomDefender("def_1")

    state = env.reset()
    for step in range(15):
        atk_obs = env.get_observation_by_id("atk_1")
        def_obs = env.get_observation_by_id("def_1")
        
        a_act = atk.act(atk_obs)
        d_act = def_agent.act(def_obs)
        
        print(f"Step {step}:")
        print(f"  Atk Action: {a_act.action_type.name} on {a_act.target_node} metadata: {a_act.metadata}")
        
        obs, rewards, done, info = env.step(a_act, d_act)
        
        print(f"  Atk Known: {[n['node_id'] for n in obs['atk_1'].get('nodes', [])]}")
        print(f"  Atk Compromised: {[n['node_id'] for n in obs['atk_1'].get('nodes', []) if n.get('status') == 'COMPROMISED']}")
        
        if done:
            print("Done early.")
            break

if __name__ == "__main__":
    debug_greedy()
