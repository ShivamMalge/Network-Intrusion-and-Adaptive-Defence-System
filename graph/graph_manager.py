"""
Graph Manager Phase 1.

Responsibility boundaries:
- Handles topology state queries and updates.
- Maintains Node and Edge indices.

Mutation constraints:
- Node topology operations must be tightly controlled by the Step Pipeline execution phase.
"""

from typing import Dict, List, Optional
from graph.node import Node, NodeType, NodeStatus
from graph.edge import Edge

class DuplicateNodeError(Exception):
    pass

class DuplicateEdgeError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass


class GraphManager:
    """
    Coordinates and maintains network topology for the simulation environment.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        # Stores directed edges: src -> dst -> Edge
        self._adjacency: Dict[str, Dict[str, Edge]] = {}

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise DuplicateNodeError(f"Node '{node.node_id}' already exists.")
        self._nodes[node.node_id] = node
        self._adjacency[node.node_id] = {}

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
            
        # Remove incident edges (where node is target)
        for src_id, targets in self._adjacency.items():
            if node_id in targets:
                del targets[node_id]
                
        # Remove the node and all its outgoing edges
        del self._adjacency[node_id]
        del self._nodes[node_id]

    def add_edge(self, edge: Edge) -> None:
        if edge.source_id not in self._nodes:
            raise NodeNotFoundError(f"Source node '{edge.source_id}' not found.")
        if edge.target_id not in self._nodes:
            raise NodeNotFoundError(f"Target node '{edge.target_id}' not found.")
            
        if edge.target_id in self._adjacency[edge.source_id]:
            raise DuplicateEdgeError(f"Edge from '{edge.source_id}' to '{edge.target_id}' already exists.")
            
        self._adjacency[edge.source_id][edge.target_id] = edge

    def remove_edge(self, source_id: str, target_id: str) -> None:
        if source_id not in self._adjacency or target_id not in self._adjacency[source_id]:
            raise ValueError(f"Edge from '{source_id}' to '{target_id}' does not exist.")
        del self._adjacency[source_id][target_id]

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        if source_id in self._adjacency:
            return self._adjacency[source_id].get(target_id)
        return None

    def get_neighbors(self, node_id: str) -> List[str]:
        if node_id not in self._adjacency:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
        return list(self._adjacency[node_id].keys())

    def number_of_nodes(self) -> int:
        return len(self._nodes)
        
    def get_all_nodes(self) -> List[Node]:
        """Return a copy of the list of all nodes."""
        return list(self._nodes.values())

    def set_node_status(self, node_id: str, status: NodeStatus) -> None:
        """Update the status of a specific node."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
        self._nodes[node_id].status = status
        
    def number_of_edges(self) -> int:
        return sum(len(targets) for targets in self._adjacency.values())


if __name__ == "__main__":
    # Self-test block
    gm = GraphManager()
    
    n1 = Node("n1", NodeType.SERVER)
    n2 = Node("n2", NodeType.WORKSTATION)
    n3 = Node("n3", NodeType.ROUTER)
    
    gm.add_node(n1)
    gm.add_node(n2)
    gm.add_node(n3)
    
    print(f"Nodes added: {gm.number_of_nodes()}")
    
    e1 = Edge("n1", "n2")
    e2 = Edge("n2", "n3")
    e3 = Edge("n3", "n1")
    
    gm.add_edge(e1)
    gm.add_edge(e2)
    gm.add_edge(e3)
    
    print(f"Edges added: {gm.number_of_edges()}")
    
    print(f"Neighbors of n2: {gm.get_neighbors('n2')}")
    
    # Simulate link failure
    gm.remove_edge("n2", "n3")
    print(f"Edges after link failure: {gm.number_of_edges()}")
    
    # Simulate node removal (should remove incident edge n3 -> n1 -> X)
    gm.remove_node("n1")
    print(f"Nodes after removal: {gm.number_of_nodes()}")
    print(f"Edges after removal: {gm.number_of_edges()}")
    
    try:
        gm.add_edge(Edge("n2", "n1"))
    except NodeNotFoundError as e:
        print(f"Caught expected error: {e}")
