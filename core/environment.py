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
from observation.observation_builder import ObservationBuilder


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
        self._obs_builder = ObservationBuilder()

    def _build_initial_state(self) -> SimulationState:
        return SimulationState(GraphManager(), VulnerabilityRegistry())

    def reset(self) -> Dict[str, Any]:
        """
        Reinitialize the simulation and build a fresh State instance.
        """
        self._state = self._build_initial_state()
        # Return initial observations for generic roles
        return {
            "atk": self.get_observation_by_id("atk_1"),
            "def": self.get_observation_by_id("def_1")
        }

    def step(self, attacker_action: AttackerAction, defender_action: DefenderAction) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one cycle of the simulation using the Step Pipeline.
        """
        if self._state.episode_done:
            raise RuntimeError("Cannot step a completed episode. Call reset().")

        rewards, done, info = self._pipeline.execute(self._state, attacker_action, defender_action)
        
        observations = {
            attacker_action.agent_id: self.get_observation_by_id(attacker_action.agent_id),
            defender_action.agent_id: self.get_observation_by_id(defender_action.agent_id)
        }

        return observations, rewards, done, info

    def get_observation(self, agent: Optional[BaseAgent]) -> Dict[str, Any]:
        """
        Generate a fog-of-war partial view of the current State for `agent`.
        """
        agent_id = agent.agent_id if agent else "atk_1"
        return self.get_observation_by_id(agent_id)

    def get_observation_by_id(self, agent_id: str) -> Dict[str, Any]:
        """Helper to get observation using agent_id string directly."""
        return self._obs_builder.build_observation(self._state, agent_id)


if __name__ == "__main__":
    from graph.node import Node, NodeType
    from vulnerabilities.vulnerability import Vulnerability
    from vulnerabilities.privilege_model import PrivilegeLevel
    from core.actions import ActionType

    print("--- Phase 4 Partial Observability Self-Test ---")
    
    rng = CentralizedRNG(seed=42)
    env = EnvironmentEngine(rng)
    
    # Reset first to create state
    print("\nEnvironment Reset...")
    initial_obs = env.reset()
    
    # Setup test scenario: 
    # n1 (known) -> n2 (hidden) -> n3 (critical asset, hidden)
    gm = env._state.graph_manager
    reg = env._state.vulnerability_registry
    
    # Register 3 nodes
    gm.add_node(Node("n1", NodeType.WORKSTATION))
    reg.register_node("n1")
    
    # n2 is monitored (honeypot or sensor)
    gm.add_node(Node("n2", NodeType.SERVER, metadata={"monitored": True}))
    reg.register_node("n2")
    
    gm.add_node(Node("n3", NodeType.CRITICAL_ASSET))
    reg.register_node("n3")
    
    # Edges: n1 -> n2 -> n3
    from graph.edge import Edge
    gm.add_edge(Edge("n1", "n2"))
    gm.add_edge(Edge("n2", "n3"))

    # Attacker starts knowing n1
    env._state.add_scanned_node("n1")
    
    print("\nInitial Observation (Attacker knows n1 only):")
    atk_obs = env.get_observation_by_id("atk_1")
    print(f"Nodes known to attacker: {[n['node_id'] for n in atk_obs['nodes']]}")

    # Step 1: Attacker SCANS n2
    print("\nStep 1: Attacker SCANS n2...")
    a_act = AttackerAction("atk_1", ActionType.SCAN, target_node="n2")
    d_act = DefenderAction("def_1", ActionType.DEFENDER_NO_OP)
    obs, rewards, done, info = env.step(a_act, d_act)
    
    print(f"Nodes known to attacker after SCAN: {[n['node_id'] for n in obs['atk_1']['nodes']]}")

    # Step 2: successful EXPLOIT on n2
    # First add a vuln so exploit is possible
    vuln = Vulnerability(vuln_id="CVE-N2", severity=5.0, required_privilege=PrivilegeLevel.NONE, zero_day=False)
    reg.add_vulnerability("n2", vuln)

    print("\nStep 2: Attacker EXPLOITS n2...")
    a_act = AttackerAction("atk_1", ActionType.EXPLOIT, target_node="n2", metadata={"vuln_id": "CVE-N2", "probability": 1.0})
    obs, rewards, done, info = env.step(a_act, d_act)
    
    print(f"Nodes known to attacker after EXPLOIT (should see n3 as neighbor): {[n['node_id'] for n in obs['atk_1']['nodes']]}")
    print(f"Compromised count: {obs['atk_1']['compromised_count']}")

    # Check Defender View
    def_obs = obs["def_1"]
    print(f"\nDefender View (All nodes seen): {[n['node_id'] for n in def_obs['nodes']]}")
    print(f"Detections: {def_obs['detections']}")
