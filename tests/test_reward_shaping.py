"""
Verification script for Phase 6B: Reward Shaping.
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from core.actions import AttackerAction, DefenderAction, ActionType
from graph.node import Node, NodeType
from graph.edge import Edge
from vulnerabilities.vulnerability import Vulnerability
from vulnerabilities.privilege_model import PrivilegeLevel


def test_reward_shaping():
    print("--- Phase 6B Reward Shaping Test ---")
    
    rng = CentralizedRNG(seed=42)
    env = EnvironmentEngine(rng)
    gm = env._state.graph_manager
    reg = env._state.vulnerability_registry
    
    # Setup simple topology
    n1 = Node("dmz", NodeType.WORKSTATION)
    n2 = Node("critical", NodeType.CRITICAL_ASSET)
    gm.add_node(n1)
    gm.add_node(n2)
    gm.add_edge(Edge("dmz", "critical"))
    
    reg.register_node("dmz")
    reg.register_node("critical")
    reg.add_vulnerability("dmz", Vulnerability("V1", 10.0, PrivilegeLevel.NONE, False))
    reg.add_vulnerability("critical", Vulnerability("V2", 10.0, PrivilegeLevel.ADMIN, False))
    
    # Attacker knows nothing initially
    
    # 1. Test Step Penalty and SCAN discovery
    atk_scan = AttackerAction("atk", ActionType.SCAN, target_node="dmz")
    def_noop = DefenderAction("def", ActionType.DEFENDER_NO_OP)
    
    obs, rewards, done, info = env.step(atk_scan, def_noop)
    
    print(f"Step 1 (SCAN dmz): Reward = {rewards['atk']}")
    # Expected: -0.02 (step) + 0.5 (discovery) = 0.48
    assert abs(rewards['atk'] - 0.48) < 1e-6
    
    # 2. Test repeat SCAN (farming)
    obs, rewards, done, info = env.step(atk_scan, def_noop)
    print(f"Step 2 (Repeat SCAN): Reward = {rewards['atk']}")
    # Expected: -0.02 (step only) = -0.02
    assert abs(rewards['atk'] - (-0.02)) < 1e-6
    
    # 3. Test Privilege Escalation (USER -> ROOT)
    # We manually set DMZ to USER and critical to ADMIN for testing escalation
    reg.escalate_privilege("dmz", PrivilegeLevel.USER)
    reg.escalate_privilege("critical", PrivilegeLevel.ADMIN)
    
    # Move Lateral to critical (discovers it)
    atk_move = AttackerAction("atk", ActionType.MOVE_LATERAL, target_node="critical")
    obs, rewards, done, info = env.step(atk_move, def_noop)
    print(f"Step 3 (MOVE_LATERAL): Reward = {rewards['atk']}")
    # Expected: -0.02 (step) + 1.0 (discovery) = 0.98
    assert abs(rewards['atk'] - 0.98) < 1e-6
    
    # Exploit Critical (ADMIN -> ROOT)
    atk_exploit = AttackerAction("atk", ActionType.EXPLOIT, target_node="critical", metadata={"vuln_id": "V2", "probability": 1.0})
    obs, rewards, done, info = env.step(atk_exploit, def_noop)
    print(f"Step 4 (EXPLOIT critical): Reward = {rewards['atk']}")
    # Expected: -0.02 (step) + 3.0 (ADMIN->ROOT) + 10.0 (CRITICAL_COMPROMISED) = 12.98
    assert abs(rewards['atk'] - 12.98) < 1e-6
    assert done == True
    
    # 5. Test Timeout Penalty
    env.reset()
    # Mock timestamp
    env._state._timestamp = 51
    # Attacker NO_OP should give step penalty
    atk_noop = AttackerAction("atk", ActionType.ATTACKER_NO_OP)
    obs, rewards, done, info = env.step(atk_noop, def_noop)
    print(f"Step 5 (TIMEOUT): Reward = {rewards['atk']}")
    # Expected: -0.02 (step) - 2.0 (timeout) = -2.02
    assert abs(rewards['atk'] - (-2.02)) < 1e-6
    assert done == True
    
    print("\nReward Shaping verification SUCCESS")


if __name__ == "__main__":
    test_reward_shaping()
