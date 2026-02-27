"""
Experiment Test Harness Phase 5A.

Responsibility boundaries:
- Orchestrates multi-episode simulation between baseline agents.
- Collects and prints performance metrics.
"""

import sys
import os
from typing import Dict, Any

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from graph.node import Node, NodeType
from graph.edge import Edge
from vulnerabilities.vulnerability import Vulnerability
from vulnerabilities.privilege_model import PrivilegeLevel
from agents.greedy_attacker import GreedyAttacker
from agents.random_defender import RandomDefender


def setup_enterprise_network(env: EnvironmentEngine):
    """
    Creates a small realistic test topology:
    n1 (DMZ) -> n2 (Internal) -> n3 (Critical Data)
    """
    gm = env._state.graph_manager
    reg = env._state.vulnerability_registry
    
    # 1. Add Nodes
    gm.add_node(Node("n1", NodeType.WORKSTATION))
    reg.register_node("n1")
    
    gm.add_node(Node("n2", NodeType.SERVER, metadata={"monitored": True}))
    reg.register_node("n2")
    
    gm.add_node(Node("n3", NodeType.CRITICAL_ASSET))
    reg.register_node("n3")
    
    # 2. Add Links
    gm.add_edge(Edge("n1", "n2"))
    gm.add_edge(Edge("n2", "n3"))
    
    # 3. Add Vulnerabilities (revealed)
    v1 = Vulnerability("CVE-DMZ", 7.0, PrivilegeLevel.NONE, zero_day=False)
    v2 = Vulnerability("CVE-INT", 5.0, PrivilegeLevel.NONE, zero_day=False)
    v3 = Vulnerability("CVE-DATA", 9.0, PrivilegeLevel.NONE, zero_day=False)
    
    reg.add_vulnerability("n1", v1)
    reg.add_vulnerability("n2", v2)
    reg.add_vulnerability("n3", v3)
    
    # 4. Attacker knowledge entry point
    env._state.add_scanned_node("n1")


def run_experiment(num_episodes: int = 100):
    print(f"Starting Baseline Stress Test: 100 Episodes (Greedy vs Random)")
    
    rng = CentralizedRNG(seed=999)
    attacker = GreedyAttacker("atk_greedy")
    defender = RandomDefender("def_random")
    
    attacker_wins = 0
    defender_wins = 0
    total_steps = 0
    
    for ep in range(num_episodes):
        env = EnvironmentEngine(rng)
        env.reset()
        setup_enterprise_network(env)
        
        # Initial observation after setup
        obs = {
            "atk_greedy": env.get_observation_by_id("atk_greedy"),
            "def_random": env.get_observation_by_id("def_random")
        }
        
        done = False
        steps = 0
        while not done and steps < 61: # Slightly over timeout to catch last result
            atk_obs = obs.get("atk_greedy", {})
            def_obs = obs.get("def_random", {})
            
            a_act = attacker.act(atk_obs)
            d_act = defender.act(def_obs)
            
            obs, rewards, done, info = env.step(a_act, d_act)
            steps += 1
            
        total_steps += steps
        
        # Win Condition: Critical Asset ROOT = Attacker Win
        ca_compromised = False
        try:
            ca_priv = env._state.vulnerability_registry.get_privilege("n3")
            if ca_priv == PrivilegeLevel.ROOT:
                ca_compromised = True
        except:
            pass
            
        if ca_compromised:
            attacker_wins += 1
        else:
            defender_wins += 1

    print("\n" + "="*30)
    print("Baseline Experiment Metrics")
    print("="*30)
    print(f"Episodes: {num_episodes}")
    print(f"Attacker Wins: {attacker_wins}")
    print(f"Defender Wins: {defender_wins}")
    print(f"Avg Steps: {total_steps / num_episodes:.2f}")
    print("="*30)


if __name__ == "__main__":
    run_experiment(100)
