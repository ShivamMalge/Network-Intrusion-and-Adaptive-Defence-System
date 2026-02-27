"""
Observation Builder Phase 4.

Responsibility boundaries:
- Constructs agent-specific partial views for fog-of-war constraints.
- Takes the global environment state and filters it based on agent role.

Mutation constraints:
- Produces entirely new output dicts. No references to live environment state.
"""

from typing import Any, Dict, Set, List
from core.state import BaseState, SimulationState


class ObservationBuilder:
    """
    Constructs partial graph observations based on agent capabilities.
    """

    def build_observation(self, state: SimulationState, agent_id: str) -> Dict[str, Any]:
        """
        Create a customized view of the environment for the specified agent.

        Args:
            state: The globally consistent simulation state.
            agent_id: The identifier of the agent (e.g., starts with 'atk' or 'def').
            
        Returns:
            A sanitized dictionary describing the observed game state.
        """
        if agent_id.startswith("atk"):
            return self._build_attacker_observation(state)
        elif agent_id.startswith("def"):
            return self._build_defender_observation(state)
        else:
            # Fallback/Default minimal view
            return {"timestep": state.timestamp, "error": "Unknown agent role"}

    def _build_attacker_observation(self, state: SimulationState) -> Dict[str, Any]:
        """
        Attacker View:
        - Known nodes: scanned + compromised + neighbors of compromised.
        - Knowledge: node types for compromised nodes.
        - Vulnerabilities: visible ones on known nodes.
        """
        scanned = state.attacker_scanned
        compromised = state.attacker_compromised
        
        known_nodes: Set[str] = scanned.union(compromised)
        
        # Add neighbors of compromised nodes (lateral discovery)
        for node_id in compromised:
            neighbors = state.graph_manager.get_neighbors(node_id)
            known_nodes.update(neighbors)

        nodes_info = []
        for node_id in known_nodes:
            node_data = {"node_id": node_id}
            
            # Attacker knows the type if they've compromised it or it's a known scanned node
            node = state.graph_manager.get_node(node_id)
            if node:
                if node_id in compromised:
                    node_data["node_type"] = node.node_type.name
                    node_data["status"] = "COMPROMISED"
                elif node_id in scanned:
                    node_data["node_type"] = node.node_type.name
                    node_data["status"] = "KNOWN"
                else:
                    node_data["status"] = "DISCOVERED" # Neighbor of compromised

            # Visible vulnerabilities
            try:
                vulns = state.vulnerability_registry.get_visible_vulnerabilities(node_id)
                node_data["vulnerabilities"] = [v.vuln_id for v in vulns]
                
                # If compromised, show privilege
                if node_id in compromised:
                    node_data["privilege"] = state.vulnerability_registry.get_privilege(node_id).name
            except Exception:
                pass
                
            nodes_info.append(node_data)

        # Edges info: Only show edges between known nodes
        edges_info = []
        # This is a bit expensive for large graphs, but standard for simulation
        for src_id in known_nodes:
            for dst_id in state.graph_manager.get_neighbors(src_id):
                if dst_id in known_nodes:
                    edges_info.append({"source": src_id, "target": dst_id})

        return {
            "timestep": state.timestamp,
            "nodes": nodes_info,
            "edges": edges_info,
            "compromised_count": len(compromised)
        }

    def _build_defender_observation(self, state: SimulationState) -> Dict[str, Any]:
        """
        Defender View:
        - Full topology.
        - Node status (ACTIVE, ISOLATED, etc.).
        - Visible vulnerabilities (patching baseline).
        - Detected attacker presence.
        """
        detected = state.defender_detected_nodes
        
        nodes_info = []
        for node in state.graph_manager.get_all_nodes():
            node_id = node.node_id
            node_data = {
                "node_id": node_id,
                "node_type": node.node_type.name,
                "status": node.status.name,
                "detected_threat": node_id in detected
            }
            
            # Visible vulnerabilities
            try:
                vulns = state.vulnerability_registry.get_visible_vulnerabilities(node_id)
                node_data["vulnerabilities"] = [v.vuln_id for v in vulns]
            except Exception:
                pass
                
            nodes_info.append(node_data)

        # Full edges
        edges_info = []
        for node in state.graph_manager.get_all_nodes():
            src_id = node.node_id
            for dst_id in state.graph_manager.get_neighbors(src_id):
                edges_info.append({"source": src_id, "target": dst_id})

        return {
            "timestep": state.timestamp,
            "nodes": nodes_info,
            "edges": edges_info,
            "detections": list(detected)
        }
