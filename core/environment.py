"""
Environment Engine Phase 3.

Responsibility boundaries:
- Holds the global game state context.
- Integrates StepPipeline, GraphManager, and Registry.
- Acts as the main entry point for stepping the simulation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

from core.state import BaseState, SimulationState
from core.step_pipeline import DeterministicStepPipeline
from core.actions import AttackerAction, DefenderAction

from graph.graph_manager import GraphManager
from vulnerabilities.vulnerability_registry import VulnerabilityRegistry
from utils.rng import CentralizedRNG
from agents.base_agent import BaseAgent


class BaseEnvironment(ABC):
    """
    Abstract Partially Observable Stochastic Markov Game Environment.
    """

    def __init__(self, rng: CentralizedRNG) -> None:
        """
        Initialize the environment with a centralized RNG to guarantee reproducibility.
        """
        self.rng = rng

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset the simulation to an initial State.
        """
        pass

    @abstractmethod
    def step(self, attacker_action: Any, defender_action: Any) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        Advance the simulation by one time step.
        """
        pass

    @abstractmethod
    def get_observation(self, agent: Optional[BaseAgent]) -> Dict[str, Any]:
        """
        Generate a fog-of-war partial view of the current State for `agent`.
        """
        pass


class EnvironmentEngine(BaseEnvironment):
    """
    Simulation environment executor mapping exact constraints.
    """

    def __init__(self, rng: CentralizedRNG):
        super().__init__(rng)
        self._state: SimulationState = self._build_initial_state()
        self._pipeline = DeterministicStepPipeline(self.rng)

    def _build_initial_state(self) -> SimulationState:
        return SimulationState(GraphManager(), VulnerabilityRegistry())

    def reset(self) -> Dict[str, Any]:
        """
        Reinitialize the simulation and build a fresh State instance.
        """
        self._state = self._build_initial_state()
        return self.get_observation(None)

    def step(self, attacker_action: AttackerAction, defender_action: DefenderAction) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one cycle of the simulation using the Step Pipeline.
        """
        if self._state.episode_done:
            raise RuntimeError("Cannot step a completed episode. Call reset().")

        rewards, done, info = self._pipeline.execute(self._state, attacker_action, defender_action)
        
        observations = {
            attacker_action.agent_id: self.get_observation(None),
            defender_action.agent_id: self.get_observation(None)
        }

        return observations, rewards, done, info

    def get_observation(self, agent: Optional[BaseAgent]) -> Dict[str, Any]:
        """
        Generate a minimal observation describing the environment structure.
        """
        # Minimal placeholder: Return topology metadata and all known vulnerabilities structurally
        nodes = []
        for node_id in self._state.graph_manager._nodes:
            try:
                vulns = self._state.vulnerability_registry.get_visible_vulnerabilities(node_id)
                nodes.append({
                    "node_id": node_id, 
                    "vulns": [v.vuln_id for v in vulns]
                })
            except Exception:
                pass
                
        return {
            "timestep": self._state.timestamp,
            "nodes": nodes
        }


if __name__ == "__main__":
    from graph.node import Node, NodeType
    from vulnerabilities.vulnerability import Vulnerability
    from vulnerabilities.privilege_model import PrivilegeLevel
    from core.actions import ActionType

    print("--- Phase 3 Environment Self-Test ---")
    
    rng = CentralizedRNG(seed=42)
    env = EnvironmentEngine(rng)
    
    # Reset first to create state
    print("\nEnvironment Reset...")
    env.reset()
    
    # Setup test scenario AFTER reset
    gm = env._state.graph_manager
    reg = env._state.vulnerability_registry
    
    ca_node = Node("server_critical", NodeType.CRITICAL_ASSET)
    gm.add_node(ca_node)
    reg.register_node("server_critical")
    
    # Adding visible vulnerability
    vuln = Vulnerability(vuln_id="CVE-BASIC", severity=5.0, required_privilege=PrivilegeLevel.NONE, zero_day=False)
    reg.add_vulnerability("server_critical", vuln)
    
    print(f"Scenario Setup: Added {ca_node.node_id} with {vuln.vuln_id}")

    # Step 1: No-ops
    a_act = AttackerAction("atk_1", ActionType.ATTACKER_NO_OP)
    d_act = DefenderAction("def_1", ActionType.DEFENDER_NO_OP)
    
    _, rewards, done, _ = env.step(a_act, d_act)
    print(f"\nStep 1 (No-ops) | t: {env._state.timestamp} | rewards: {rewards} | done: {done}")

    # Step 2: successful EXPLOIT
    a_act = AttackerAction("atk_1", ActionType.EXPLOIT, target_node="server_critical", metadata={"vuln_id": "CVE-BASIC", "probability": 1.0})
    d_act = DefenderAction("def_1", ActionType.DEFENDER_NO_OP)

    _, rewards, done, _ = env.step(a_act, d_act)
    print(f"\nStep 2 (Exploit) | t: {env._state.timestamp} | rewards: {rewards} | done: {done}")
    
    try:
        curr_priv = reg.get_privilege("server_critical")
        print(f"Privilege of Critical Server: {curr_priv.name}")
    except Exception as e:
        print(f"Error fetching privilege: {e}")
