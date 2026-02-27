"""
Step Pipeline Phase 3 Refactor.

Responsibility boundaries:
- Orchestrates deterministic step execution in EXACT specified order.
- Follows a pure transition model: No state mutation outside update_state().
- Computes placeholder rewards and termination criteria.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List, Set

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
    
    # Phase 4 Epistemic updates
    new_scans: List[str] = field(default_factory=list)
    new_compromises: List[str] = field(default_factory=list)
    new_detections: List[str] = field(default_factory=list)
    
    # Phase 5B Stochastic Detection
    queued_detections: List[Tuple[str, int]] = field(default_factory=list)


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

        # 5. Capture context for reward shaping
        target_node = attacker_action.target_node
        pre_scanned = state.attacker_scanned
        pre_priv = PrivilegeLevel.NONE
        if target_node:
            try:
                pre_priv = state.vulnerability_registry.get_privilege(target_node)
            except Exception:
                pass

        # 6. Advance - ONLY place where mutation happens
        self.update_state(state, effects)

        # 7. Check Terminal
        done = self.check_termination(state)
        
        # 8. Rewards
        rewards = self.compute_rewards(
            state, effects, attacker_action, defender_action, 
            attk_success, def_res, done, pre_scanned, pre_priv
        )
        
        if done:
            state.mark_done()

        return rewards, done, info

    def validate_actions(self, attacker_action: AttackerAction, defender_action: DefenderAction) -> None:
        attacker_action.validate()
        defender_action.validate()

    def apply_attacker_action(self, state: SimulationState, action: AttackerAction, effects: TransitionEffects) -> Any:
        # Pass necessary data directly to exploit resolution
        if action.action_type == ActionType.SCAN:
            if action.target_node:
                effects.new_scans.append(action.target_node)
            return None

        if action.action_type == ActionType.EXPLOIT:
            return action.metadata.get("vuln_id")
            
        if action.action_type == ActionType.MOVE_LATERAL:
            if action.target_node:
                # Basic move lateral stages discovery
                effects.new_scans.append(action.target_node)
            return True
            
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
            # Phase 4 Epistemic: Stage compromise
            effects.new_compromises.append(target)
            
            # Phase 5B Stochastic Detection: If exploit succeeds AND node is monitored
            node = state.graph_manager.get_node(target)
            if node and node.metadata.get("monitored", False):
                 # Probabilistic detection roll
                 if self._rng.uniform(0.0, 1.0) < state.detection_probability:
                     # Stage detection with delay
                     trigger = state.timestamp + state.detection_delay
                     effects.queued_detections.append((target, trigger))
                
        return success

    def update_state(self, state: SimulationState, effects: TransitionEffects) -> None:
        """Apply all staged effects to the state."""
        # Semantic Model A: Detection appears at t_exploit + detection_delay
        horizon_time = state.timestamp + 1
        
        # 1. Add new staged detections to state's queue (Phase 5B)
        for node_id, trigger in effects.queued_detections:
            state.add_to_detection_queue(node_id, trigger)

        # 2. Handle False Positives (Phase 5B)
        if self._rng.uniform(0.0, 1.0) < state.false_positive_rate:
            all_nodes = [n.node_id for n in state.graph_manager.get_all_nodes()]
            if all_nodes:
                fp_node = self._rng.choice(all_nodes)
                state.add_to_detection_queue(fp_node, state.timestamp + state.detection_delay)

        # 3. Process triggered detections using end-of-step horizon
        triggered = state.process_detection_queue(horizon_time)
        for node_id in triggered:
            state.add_detected_node(node_id)

        # 4. Apply Epistemic updates (Phase 4)
        for node_id in effects.new_scans:
            state.add_scanned_node(node_id)
            
        for node_id in effects.new_compromises:
            state.add_compromised_node(node_id)
            
        # 5. Apply Patching
        for node_id in effects.patched_nodes:
            state.vulnerability_registry.patch_vulnerabilities(node_id)
            
        # 6. Apply Isolation
        for node_id in effects.isolated_nodes:
            state.graph_manager.set_node_status(node_id, NodeStatus.ISOLATED)
            
        # 7. Apply Privilege Resets (to USER)
        for node_id in effects.privilege_resets:
            state.vulnerability_registry.reset_privilege(node_id)
            
        # 8. Apply Privilege Escalations
        for node_id, new_level in effects.privilege_escalations:
            state.vulnerability_registry.escalate_privilege(node_id, new_level)
            
        # 9. Advance time
        state.increment_time()

    def compute_rewards(
        self, 
        state: SimulationState,
        effects: TransitionEffects,
        atk_action: AttackerAction, 
        def_action: DefenderAction, 
        atk_success: bool, 
        def_success: bool,
        done: bool,
        pre_scanned: Set[str],
        pre_priv: PrivilegeLevel
    ) -> Dict[str, float]:
        rewards = {atk_action.agent_id: 0.0, def_action.agent_id: 0.0}
        atk_id = atk_action.agent_id

        # 1. Per-step penalty (-0.02)
        rewards[atk_id] -= 0.02

        # Context for discovery
        is_new_discovery = False
        if atk_action.target_node and atk_action.target_node not in pre_scanned:
            is_new_discovery = True

        # 2. SCAN Success Reward (+0.5 if expanded knowledge)
        if atk_action.action_type == ActionType.SCAN and is_new_discovery:
            rewards[atk_id] += 0.5

        # 3. Successful Lateral Movement (+1.0 if access to new node)
        if atk_action.action_type == ActionType.MOVE_LATERAL and is_new_discovery:
            rewards[atk_id] += 1.0

        # 4. Privilege Escalation
        if atk_action.action_type == ActionType.EXPLOIT and atk_success:
            escalation = next((new_lvl for node, new_lvl in effects.privilege_escalations if node == atk_action.target_node), None)
            if escalation:
                if pre_priv == PrivilegeLevel.USER and escalation == PrivilegeLevel.ADMIN:
                    rewards[atk_id] += 2.0
                elif pre_priv == PrivilegeLevel.ADMIN and escalation == PrivilegeLevel.ROOT:
                    rewards[atk_id] += 3.0
                elif pre_priv == PrivilegeLevel.USER and escalation == PrivilegeLevel.ROOT:
                    rewards[atk_id] += 5.0 # Combined bonus

        # 5. Terminal Reward (Successful CRITICAL compromise: +10.0)
        # Check termination condition logic
        is_critical_compromised = False
        for node in state.graph_manager.get_all_nodes():
            if node.node_type.name == "CRITICAL_ASSET":
                try:
                    if state.vulnerability_registry.get_privilege(node.node_id) == PrivilegeLevel.ROOT:
                        is_critical_compromised = True
                        break
                except Exception:
                    pass

        if is_critical_compromised:
            rewards[atk_id] += 10.0
            rewards[def_action.agent_id] -= 5.0

        # 6. Timeout Penalty (-2.0)
        # We define timeout here as done=True but no critical compromise
        if done and state.timestamp > 50 and not is_critical_compromised:
            rewards[atk_id] -= 2.0

        # Defender rewards (unchanged baseline)
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
