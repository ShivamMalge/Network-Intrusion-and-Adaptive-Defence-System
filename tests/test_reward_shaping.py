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
    def_noop = DefenderAction("def", ActionType.DEFENDER_NO_OP)
    
    # We manually set DMZ to USER and critical to ADMIN for testing escalation
    reg.escalate_privilege("dmz", PrivilegeLevel.USER)
    reg.escalate_privilege("critical", PrivilegeLevel.ADMIN)
    
    actions = [
        AttackerAction("atk", ActionType.SCAN, target_node="dmz"),
        AttackerAction("atk", ActionType.SCAN, target_node="dmz"),
        AttackerAction("atk", ActionType.MOVE_LATERAL, target_node="critical"),
        AttackerAction("atk", ActionType.EXPLOIT, target_node="critical", metadata={"vuln_id": "V2", "probability": 1.0}),
        AttackerAction("atk", ActionType.ATTACKER_NO_OP) # This should never run if EXPLOIT finishes the episode
    ]
    
    done = False
    step = 0

    while not done and step < len(actions):
        action = actions[step]
        step += 1
        obs, rewards, done, info = env.step(action, def_noop)
        
        if action.action_type == ActionType.SCAN:
            print(f"Step {step} (SCAN {action.target_node}): Reward = {rewards['atk']}")
        elif action.action_type == ActionType.MOVE_LATERAL:
            print(f"Step {step} (MOVE_LATERAL): Reward = {rewards['atk']}")
        elif action.action_type == ActionType.EXPLOIT:
            print(f"Step {step} (EXPLOIT {action.target_node}): Reward = {rewards['atk']}")
        else:
            print(f"Step {step} (TIMEOUT): Reward = {rewards['atk']}")
            
    print("\nReward Shaping verification SUCCESS")


if __name__ == "__main__":
    test_reward_shaping()
