"""
Privilege Model Phase 2.

Responsibility boundaries:
- Formalizes the privilege hierarchy.
- Enforces monotonic escalation rules.
"""

from enum import IntEnum


class PrivilegeLevel(IntEnum):
    """
    Hierarchical privilege levels for a node.
    Cannot decrease or exceed ROOT (3).
    """
    NONE = 0
    USER = 1
    ADMIN = 2
    ROOT = 3

    @classmethod
    def max_level(cls) -> 'PrivilegeLevel':
        """Get the highest possible privilege level."""
        return cls.ROOT
