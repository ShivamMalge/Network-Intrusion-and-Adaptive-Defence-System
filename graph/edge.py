"""
Edge Entity Phase 1.

Responsibility boundaries:
- Represents connectivity between two `Node`s.
- Directed/Undirected structural base.
"""

from typing import Any


class Edge:
    """Connection Link between two Nodes."""

    def __init__(self, source_id: str, target_id: str, bandwidth: float = 1.0, latency: float = 0.0, encrypted: bool = False, monitored: bool = False):
        if source_id == target_id:
            raise ValueError(f"Self-loops are not allowed: {source_id} -> {target_id}")
            
        self._source_id = source_id
        self._target_id = target_id
        self._bandwidth = bandwidth
        self._latency = latency
        self._encrypted = encrypted
        self._monitored = monitored

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def target_id(self) -> str:
        return self._target_id

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def encrypted(self) -> bool:
        return self._encrypted

    @property
    def monitored(self) -> bool:
        return self._monitored

    def __repr__(self) -> str:
        return (f"Edge({self._source_id} -> {self._target_id}, "
                f"bw={self._bandwidth}, lat={self._latency}, "
                f"enc={self._encrypted}, mon={self._monitored})")
