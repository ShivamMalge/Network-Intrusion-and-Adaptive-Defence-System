"""
rl/population_manager.py
Manages a rolling pool of historical agent checkpoints for PBT.
"""

import os
import random

class PopulationManager:
    def __init__(self, pool_size=3):
        self.pool_size = pool_size
        self.attacker_pool = []
        self.defender_pool = []

    def load_existing(self, attacker_prefix="dqn_attacker_cycle_", defender_prefix="dqn_defender_cycle_", directory="."):
        """Scan directory for existing cycle checkpoints and populate pools."""
        # 1. Gather all available cycles
        a_cycles = []
        d_cycles = []
        
        # Check baseline files first
        if os.path.exists(os.path.join(directory, "dqn_attacker.pt")):
            self.attacker_pool.append(os.path.join(directory, "dqn_attacker.pt"))
        if os.path.exists(os.path.join(directory, "dqn_defender.pt")):
            self.defender_pool.append(os.path.join(directory, "dqn_defender.pt"))

        # Find cycle checkpoints
        for f in os.listdir(directory):
            if f.startswith(attacker_prefix) and f.endswith(".pt"):
                try:
                    cycle = int(f.replace(attacker_prefix, "").replace(".pt", ""))
                    a_cycles.append((cycle, os.path.join(directory, f)))
                except ValueError:
                    continue
            if f.startswith(defender_prefix) and f.endswith(".pt"):
                try:
                    cycle = int(f.replace(defender_prefix, "").replace(".pt", ""))
                    d_cycles.append((cycle, os.path.join(directory, f)))
                except ValueError:
                    continue
        
        # Sort and take last N
        a_cycles.sort(key=lambda x: x[0])
        d_cycles.sort(key=lambda x: x[0])
        
        for _, path in a_cycles:
            self.add_attacker(path)
        for _, path in d_cycles:
            self.add_defender(path)
            
        print(f"PopulationManager: Loaded {len(self.attacker_pool)} attackers and {len(self.defender_pool)} defenders.")

    def add_attacker(self, path):
        if path not in self.attacker_pool:
            self.attacker_pool.append(path)
        if len(self.attacker_pool) > self.pool_size:
            self.attacker_pool.pop(0)

    def add_defender(self, path):
        if path not in self.defender_pool:
            self.defender_pool.append(path)
        if len(self.defender_pool) > self.pool_size:
            self.defender_pool.pop(0)

    def get_random_attacker(self):
        if not self.attacker_pool:
            return None
        return random.choice(self.attacker_pool)

    def get_random_defender(self):
        if not self.defender_pool:
            return None
        return random.choice(self.defender_pool)

    def get_latest_attacker(self):
        return self.attacker_pool[-1] if self.attacker_pool else None

    def get_latest_defender(self):
        return self.defender_pool[-1] if self.defender_pool else None
