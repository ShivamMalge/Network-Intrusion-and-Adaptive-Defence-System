"""
Step Pipeline Phase 3 Refactor.

Responsibility boundaries:
- Orchestrates deterministic step execution in EXACT specified order.
- Follows a pure transition model: No state mutation outside update_state().
- Computes placeholder rewards and termination criteria.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List

from core.actions import AttackerAction, DefenderAction, ActionType
from vulnerabilities.privilege_model import PrivilegeLevel
from vulnerabilities.exploit import Exploit
from utils.rng import CentralizedRNG
from graph.node import NodeStatus

# In order to strictly type the input without circular dependencies,
# we import SimulationState.
from core.state import SimulationState


@dataclass
class TransitionEffects:
    """Container for staged state mutations."""
    privilege_escalations: List[Tuple[str, PrivilegeLevel]] = field(default_factory=list)
    privilege_resets: List[str] = field(default_factory=list)
    patched_nodes: List[str] = field(default_factory=list)
    isolated_nodes: List[str] = field(default_factory=list)


class DeterministicStepPipeline:
    """
    Formalizer for the strict 7-phase step pipeline with pure transitions.
    """

    def __init__(self, rng: CentralizedRNG):
        self._rng = rng

    def execute(self, state: SimulationState, attacker_action: AttackerAction, defender_action: DefenderAction) -> Tuple[Dict[str, float], bool, Dict[str, Any]]:
        """
        1. validate_actions()
        2. apply_attacker_action()
        3. apply_defender_action()
        4. resolve_exploits()
        5. update_state()
        6. compute_rewards()
        7. check_termination()
        """
        
        # 0. Initialize effects
        effects = TransitionEffects()

        # 1. Validate
        self.validate_actions(attacker_action, defender_action)

        info: Dict[str, Any] = {"attacker_success": False, "defender_success": False}
        
        # 2 & 3. Actions / Events
        attk_res = self.apply_attacker_action(state, attacker_action, effects)
        def_res = self.apply_defender_action(state, defender_action, effects)
        
        # 4. Resolve Context
        attk_success = self.resolve_exploits(state, attacker_action, attk_res, effects)
        info["attacker_success"] = attk_success
        info["defender_success"] = def_res

        # 5. Advance - ONLY place where mutation happens
        self.update_state(state, effects)

        # 6. Rewards
        rewards = self.compute_rewards(attacker_action, defender_action, attk_success, def_res)

        # 7. Check Terminal
        done = self.check_termination(state)
        
        if done:
            state.mark_done()

        return rewards, done, info

    def validate_actions(self, attacker_action: AttackerAction, defender_action: DefenderAction) -> None:
        attacker_action.validate()
        defender_action.validate()

    def apply_attacker_action(self, state: SimulationState, action: AttackerAction, effects: TransitionEffects) -> Any:
        # Pass necessary data directly to exploit resolution
        if action.action_type == ActionType.EXPLOIT:
            return action.metadata.get("vuln_id")
        return None

    def apply_defender_action(self, state: SimulationState, action: DefenderAction, effects: TransitionEffects) -> bool:
        success = False
        target = action.target_node
        
        if not target or state.graph_manager.get_node(target) is None:
            return False

        if action.action_type == ActionType.PATCH:
            # Stage patching
            effects.patched_nodes.append(target)
            success = True

        elif action.action_type == ActionType.ISOLATE:
            # Stage isolation
            effects.isolated_nodes.append(target)
            success = True

        elif action.action_type == ActionType.RESET_PRIVILEGE:
            # Stage privilege reset
            effects.privilege_resets.append(target)
            success = True
                
        return success

    def resolve_exploits(self, state: SimulationState, action: AttackerAction, vuln_id: Optional[str], effects: TransitionEffects) -> bool:
        if action.action_type != ActionType.EXPLOIT or not vuln_id or not action.target_node:
            return False

        target = action.target_node
        try:
            vulns = state.vulnerability_registry.get_visible_vulnerabilities(target)
        except Exception:
            return False

        target_vuln = next((v for v in vulns if v.vuln_id == vuln_id), None)
        if not target_vuln:
            return False

        try:
            curr_priv = state.vulnerability_registry.get_privilege(target)
        except Exception:
            return False

        # Build Exploit placeholder
        prob = action.metadata.get("probability", 1.0)
        exploit = Exploit(exploit_id="EXP-DYNAMIC", target_vuln_id=vuln_id, base_success_probability=prob, stealth_cost=0.0)

        success = exploit.attempt(curr_priv, target_vuln, self._rng)
        if success:
            # Stage escalation
            effects.privilege_escalations.append((target, PrivilegeLevel.ROOT))
                
        return success

    def update_state(self, state: SimulationState, effects: TransitionEffects) -> None:
        """Apply all staged effects to the state."""
        
        # 1. Apply Patching
        for node_id in effects.patched_nodes:
            state.vulnerability_registry.patch_vulnerabilities(node_id)
            
        # 2. Apply Isolation
        for node_id in effects.isolated_nodes:
            state.graph_manager.set_node_status(node_id, NodeStatus.ISOLATED)
            
        # 3. Apply Privilege Resets (to USER)
        for node_id in effects.privilege_resets:
            state.vulnerability_registry.reset_privilege(node_id)
            
        # 4. Apply Privilege Escalations
        for node_id, new_level in effects.privilege_escalations:
            state.vulnerability_registry.escalate_privilege(node_id, new_level)
            
        # 5. Advance time
        state.increment_time()

    def compute_rewards(self, atk_action: AttackerAction, def_action: DefenderAction, atk_success: bool, def_success: bool) -> Dict[str, float]:
        rewards = {atk_action.agent_id: 0.0, def_action.agent_id: 0.0}

        # Attacker rewards
        if atk_action.action_type == ActionType.EXPLOIT:
            if atk_success:
                rewards[atk_action.agent_id] += 10.0
                rewards[def_action.agent_id] -= 5.0
            else:
                rewards[atk_action.agent_id] -= 1.0

        # Defender rewards
        if def_action.action_type == ActionType.PATCH and def_success:
            rewards[def_action.agent_id] += 5.0

        return rewards

    def check_termination(self, state: SimulationState) -> bool:
        # Timeout rule
        if state.timestamp > 50:
            return True
            
        # Critical Asset ROOT compromise rule
        for node in state.graph_manager.get_all_nodes():
            if node.node_type.name == "CRITICAL_ASSET":
                try:
                    if state.vulnerability_registry.get_privilege(node.node_id) == PrivilegeLevel.ROOT:
                        return True
                except Exception:
                    pass
                    
        return False
