"""
Centralized Random Number Generator Utility.

This module provides a single, globally instantiable RNG to ensure
strict reproducibility across the entire simulation.

Responsibility boundaries:
- Must be the ONLY source of randomness in the simulation.
- Modules must accept an RNG instance, never instantiate their own `random` or `numpy.random`.

Mutation constraints:
- The internal state of the RNG is mutated only when drawing random numbers.
- The seed can only be set once during initialization.
"""

import random
from typing import Optional, Any


class CentralizedRNG:
    """
    A centralized random number generator to enforce reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the RNG with a specific seed.
        
        Args:
            seed: An integer seed for deterministic execution.
        """
        self._seed = seed
        self._rng_instance = random.Random(seed)

    def get_rng(self) -> random.Random:
        """
        Get the underlying RNG instance.
        
        Returns:
            The RNG instance (Standard library random.Random).
        """
        return self._rng_instance

    def uniform(self, a: float, b: float) -> float:
        """Draw a uniform random float in [a, b]."""
        return self._rng_instance.uniform(a, b)

    def choice(self, seq: Any) -> Any:
        """Randomly choose an element from a non-empty sequence."""
        return self._rng_instance.choice(seq)

    def randint(self, a: int, b: int) -> int:
        """Return a random integer N such that a <= N <= b."""
        return self._rng_instance.randint(a, b)
