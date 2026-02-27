"""
Topology Events Phase 1.

Responsibility boundaries:
- Formalizes network changes like device joins and link failures.
"""

from abc import ABC, abstractmethod
from typing import Any
from graph.graph_manager import GraphManager
from graph.node import Node
from graph.edge import Edge


class TopologyEvent(ABC):
    """Abstract base event for network topology changes."""
    
    @abstractmethod
    def apply(self, graph_manager: GraphManager) -> None:
        """Apply the deterministic topology event to the graph manager."""
        pass


class LinkFailureEvent(TopologyEvent):
    """Represents a network link going down."""
    
    def __init__(self, source_id: str, target_id: str):
        self.source_id = source_id
        self.target_id = target_id
        
    def apply(self, graph_manager: GraphManager) -> None:
        try:
            graph_manager.remove_edge(self.source_id, self.target_id)
        except ValueError:
            pass # Idempotent or edge already gone


class DeviceJoinEvent(TopologyEvent):
    """Represents a new device joining the network."""
    
    def __init__(self, node: Node):
        self.node = node
        
    def apply(self, graph_manager: GraphManager) -> None:
        graph_manager.add_node(self.node)


class DeviceLeaveEvent(TopologyEvent):
    """Represents a device intentionally leaving the network."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
    def apply(self, graph_manager: GraphManager) -> None:
        try:
            graph_manager.remove_node(self.node_id)
        except Exception:
            pass # Idempotent or node already gone
