"""
Configuration Utility Phase 0.

Responsibility boundaries:
- Parses simulation settings, budgets, rewards coefficients.
- Must be passed to initialize systems natively.

Mutation constraints:
- Must freeze after initialization to avoid mid-simulation configuration drift.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable container defining overall runtime scenario setup.
    """
    seed: int = 42
    # Additional configuration properties placeholder
