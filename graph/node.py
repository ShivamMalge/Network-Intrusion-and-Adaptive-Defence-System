"""
Node Entity Phase 1.

Responsibility boundaries:
- Represents an identifiable, immutable point in the graph.
"""

from enum import Enum, auto
from typing import Dict, Any


class NodeType(Enum):
    SERVER = auto()
    WORKSTATION = auto()
    ROUTER = auto()
    HONEYPOT = auto()
    INSIDER_TERMINAL = auto()
    CRITICAL_ASSET = auto()


class NodeStatus(Enum):
    ACTIVE = auto()
    ISOLATED = auto()
    DOWN = auto()


class Node:
    """Network Device Component."""

    def __init__(self, node_id: str, node_type: NodeType, status: NodeStatus = NodeStatus.ACTIVE, metadata: Dict[str, Any] = None):
        self._node_id = node_id
        self._node_type = node_type
        self.status = status
        self.metadata = metadata if metadata is not None else {}

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_type(self) -> NodeType:
        return self._node_type

    def __hash__(self) -> int:
        return hash(self._node_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return False
        return self._node_id == other.node_id

    def __repr__(self) -> str:
        return f"Node(id={self._node_id}, type={self._node_type.name}, status={self.status.name})"
