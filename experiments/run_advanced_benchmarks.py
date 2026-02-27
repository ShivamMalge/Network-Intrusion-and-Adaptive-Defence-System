"""
Advanced Benchmark Harness for Phase 5B Validation.

Orchestrates 100-episode runs with IDS stochasticity and advanced metrics collection.
"""

import sys
import os
import statistics
from typing import Dict, Any, List

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from graph.node import Node, NodeType
from graph.edge import Edge
from vulnerabilities.vulnerability import Vulnerability
from vulnerabilities.privilege_model import PrivilegeLevel
from agents.greedy_attacker import GreedyAttacker
from agents.greedy_defender import GreedyDefender
from agents.random_defender import RandomDefender


def setup_benchmark_topology(env: EnvironmentEngine):
    """
    n1 (DMZ) -> n2 (INTERNAL, monitored) -> n3 (DATA, Critical)
    """
    gm = env._state.graph_manager
    reg = env._state.vulnerability_registry
    
    # 1. Add Decoys FIRST (to distract Greedy Defender)
    for i in range(1, 4):
        d_id = f"decoy_{i:02d}"
        gm.add_node(Node(d_id, NodeType.WORKSTATION))
        reg.register_node(d_id)
        reg.add_vulnerability(d_id, Vulnerability(f"V-DECOY-{i}", 5.0, PrivilegeLevel.NONE, False))

    # 2. Add real nodes
    gm.add_node(Node("n1", NodeType.WORKSTATION))
    reg.register_node("n1")
    
    gm.add_node(Node("n2", NodeType.SERVER, metadata={"monitored": True}))
    reg.register_node("n2")
    
    gm.add_node(Node("n3", NodeType.CRITICAL_ASSET))
    reg.register_node("n3")
    
    # 3. Add Links
    gm.add_edge(Edge("n1", "n2"))
    gm.add_edge(Edge("n2", "n3"))
    
    # 4. Add Vulnerabilities (revealed)
    reg.add_vulnerability("n1", Vulnerability("V1", 7.0, PrivilegeLevel.NONE, False))
    reg.add_vulnerability("n2", Vulnerability("V2", 5.0, PrivilegeLevel.NONE, False))
    reg.add_vulnerability("n3", Vulnerability("V3", 9.0, PrivilegeLevel.NONE, False))
    
    # 5. Attacker knowledge entry point
    env._state.add_scanned_node("n1")
    
    # 6. Set IDS Parameters (Phase 5B)
    env._state.set_ids_parameters(prob=0.6, prompt_delay=2, fp_rate=0.05)


def run_configuration(name: str, attacker, defender, num_episodes: int = 100):
    print(f"\nRunning Benchmark: {name}")
    print("-" * 30)
    
    rng = CentralizedRNG(seed=42)
    attacker_wins = 0
    defender_wins = 0
    lengths = []
    reasons = {"CRITICAL_COMPROMISED": 0, "TIMEOUT": 0}
    
    for ep in range(num_episodes):
        env = EnvironmentEngine(rng)
        # Environmental defaults
        obs_initial = env.reset()
        setup_benchmark_topology(env)
        
        # Refresh observations after manual topology setup
        obs = {
            "atk_1": env.get_observation_by_id("atk_1"),
            "def_1": env.get_observation_by_id("def_1")
        }
        
        done = False
        steps = 0
        while not done and steps < 50:
            a_act = attacker.act(obs.get("atk_1", {}))
            d_act = defender.act(obs.get("def_1", {}))
            
            obs, rewards, done, info = env.step(a_act, d_act)
            steps += 1
            
        lengths.append(steps)
        
        # Check Winner
        ca_priv = PrivilegeLevel.NONE
        try:
            ca_priv = env._state.vulnerability_registry.get_privilege("n3")
        except:
            pass
            
        if ca_priv == PrivilegeLevel.ROOT:
            attacker_wins += 1
            reasons["CRITICAL_COMPROMISED"] += 1
        else:
            defender_wins += 1
            reasons["TIMEOUT"] += 1

    avg_steps = sum(lengths) / num_episodes
    std_dev = statistics.stdev(lengths) if num_episodes > 1 else 0.0
    
    print(f"Episodes: {num_episodes}")
    print(f"Attacker Wins: {attacker_wins}")
    print(f"Defender Wins: {defender_wins}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print(f"Std Dev Steps: {std_dev:.2f}")
    print(f"Termination Reasons: {reasons}")


if __name__ == "__main__":
    import statistics
    
    # 1. Greedy vs Greedy
    at1 = GreedyAttacker("atk_1")
    df1 = GreedyDefender("def_1")
    run_configuration("Greedy vs Greedy", at1, df1)
    
    # 2. Greedy vs Random
    at2 = GreedyAttacker("atk_1")
    df2 = RandomDefender("def_1")
    run_configuration("Greedy vs Random", at2, df2)
