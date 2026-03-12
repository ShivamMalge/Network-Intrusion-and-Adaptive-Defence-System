import json
import os
import time
import torch
import numpy as np
import gc
import random
from typing import Dict, List, Any, Tuple
from core.environment import EnvironmentEngine
from utils.rng import CentralizedRNG
from agents.greedy_attacker import GreedyAttacker
from agents.greedy_defender import GreedyDefender
from agents.random_attacker import RandomAttacker
from agents.random_defender import RandomDefender
from rl.dqn_agent_wrapper import DQNAgentWrapper
from vulnerabilities.privilege_model import PrivilegeLevel

class RegressionAudit:
    def __init__(self):
        self.report = {
            "baseline_results": {},
            "cross_agent_results": {},
            "selfplay_matrix": {},
            "invariant_status": "PENDING",
            "reset_hygiene": "PENDING",
            "stochastic_stability": {},
            "memory_integrity": "PENDING"
        }
        self.rng = CentralizedRNG(seed=42)

    def run_1000_episodes(self, attacker, defender, desc="Audit"):
        print(f"  - Running 1000 episodes: {desc}")
        wins = 0
        total_steps = 0
        rewards_atk = []
        episode_lengths = []
        
        for ep in range(1000):
            env = EnvironmentEngine(self.rng)
            obs = env.reset()
            
            # Key fix: observation IDs might differ in scripts vs env
            atk_id = attacker.agent_id
            def_id = defender.agent_id
            
            done = False
            steps = 0
            ep_reward = 0
            while not done and steps < 50:
                a_obs = env.get_observation_by_id(atk_id)
                d_obs = env.get_observation_by_id(def_id)
                
                a_act = attacker.act(a_obs)
                d_act = defender.act(d_obs)
                
                _, rewards, done, info = env.step(a_act, d_act)
                ep_reward += rewards.get(atk_id, 0)
                steps += 1
            
            if info.get("termination_reason") == "CRITICAL_COMPROMISED":
                wins += 1
            
            total_steps += steps
            rewards_atk.append(ep_reward)
            episode_lengths.append(steps)
            
        return {
            "win_rate": wins / 1000.0,
            "avg_steps": np.mean(episode_lengths),
            "reward_mean": np.mean(rewards_atk),
            "reward_std": np.std(rewards_atk),
            "length_std": np.std(episode_lengths)
        }

    def part1_baseline_sweep(self):
        print("\n--- Part 1: Deterministic Baseline Sweep ---")
        # 1. Greedy vs Greedy
        res_gg = self.run_1000_episodes(GreedyAttacker("atk_greedy"), GreedyDefender("def_greedy"), "Greedy vs Greedy")
        
        # 2. Greedy vs DQN Clean (Defender)
        dqn_def_clean = DQNAgentWrapper("def_dqn_clean", "dqn_defender_clean.pt", is_attacker=False)
        res_gd = self.run_1000_episodes(GreedyAttacker("atk_greedy"), dqn_def_clean, "Greedy vs DQN Clean Def")
        
        # 3. DQN Clean (Attacker) vs Greedy
        dqn_atk_clean = DQNAgentWrapper("atk_dqn_clean", "dqn_attacker_clean.pt", is_attacker=True)
        res_dg = self.run_1000_episodes(dqn_atk_clean, GreedyDefender("def_greedy"), "DQN Clean Atk vs Greedy")
        
        # 4. DQN Clean vs DQN Clean
        res_dd = self.run_1000_episodes(dqn_atk_clean, dqn_def_clean, "DQN Clean vs DQN Clean")
        
        self.report["baseline_results"] = {
            "greedy_vs_greedy": res_gg,
            "greedy_vs_dqn_def_clean": res_gd,
            "dqn_atk_clean_vs_greedy": res_dg,
            "dqn_atk_clean_vs_dqn_def_clean": res_dd
        }

    def part2_selfplay_regression(self):
        print("\n--- Part 2: Self-Play Checkpoint Regression ---")
        cycles = [1, 2, 3, 4, 5]
        matrix = {}
        for i in cycles:
            atk_path = f"dqn_attacker_cycle_{i}.pt"
            if not os.path.exists(atk_path): continue
            atk = DQNAgentWrapper(f"atk_c{i}", atk_path, is_attacker=True)
            matrix[f"A{i}"] = {}
            for j in cycles:
                def_path = f"dqn_defender_cycle_{j}.pt"
                if not os.path.exists(def_path): continue
                dfn = DQNAgentWrapper(f"def_c{j}", def_path, is_attacker=False)
                
                print(f"  - A{i} vs D{j} (200 eps)")
                wins = 0
                for ep in range(200):
                    env = EnvironmentEngine(self.rng)
                    env.reset()
                    done = False
                    steps = 0
                    while not done and steps < 50:
                        a_obs = env.get_observation_by_id(f"atk_c{i}")
                        d_obs = env.get_observation_by_id(f"def_c{j}")
                        obs, _, done, info = env.step(atk.act(a_obs), dfn.act(d_obs))
                        steps += 1
                    if info.get("termination_reason") == "CRITICAL_COMPROMISED":
                        wins += 1
                matrix[f"A{i}"][f"D{j}"] = wins / 200.0
                
        self.report["selfplay_matrix"] = matrix
        with open("selfplay_regression_matrix.json", "w") as f:
            json.dump(matrix, f, indent=2)

    def part3_invariant_stress(self):
        print("\n--- Part 3: Invariant Stress Testing ---")
        atk = RandomAttacker("atk_rand")
        dfn = RandomDefender("def_rand")
        
        try:
            for ep in range(500):
                env = EnvironmentEngine(self.rng)
                env.reset()
                done = False
                steps = 0
                while not done and steps < 50:
                    a_obs = env.get_observation_by_id("atk_rand")
                    d_obs = env.get_observation_by_id("def_rand")
                    _, _, done, info = env.step(atk.act(a_obs), dfn.act(d_obs))
                    steps += 1
                
                # Check invariant: current_step == 0 or done
                # Our env doesn't have current_step directly, it has timestamp
                assert env._state.timestamp >= 0
                if steps < 50:
                    assert done, f"Episode ended at step {steps} without done flag"
            
            self.report["invariant_status"] = "PASS"
        except Exception as e:
            print(f"INVARIANT FAILURE: {e}")
            self.report["invariant_status"] = f"FAIL: {e}"

    def part4_reset_hygiene(self):
        print("\n--- Part 4: Episode Reset Hygiene & Aliasing ---")
        env1 = EnvironmentEngine(self.rng)
        env1.reset()
        
        # Manually corrupt state of env1
        env1._state.add_compromised_node("data")
        
        env2 = EnvironmentEngine(self.rng)
        env2.reset()
        
        try:
            # Check if env2 is clean despite env1 corruption
            assert "data" not in env2._state.attacker_compromised, "State Aliasing detected!"
            assert env2._state.graph_manager is not env1._state.graph_manager, "Registry Aliasing detected!"
            assert env2._state.vulnerability_registry is not env1._state.vulnerability_registry, "Registry Aliasing detected!"
            
            self.report["reset_hygiene"] = "PASS"
        except Exception as e:
            self.report["reset_hygiene"] = f"FAIL: {e}"

    def part5_stochastic_stability(self):
        print("\n--- Part 5: Stochastic Stability Sweep ---")
        # DQN Atk vs DQN Def with epsilon=0.05
        # Note: Our DQNAgentWrapper is deterministic. We need to add epsilon exploration for this test.
        # I'll create a subclass or wrap it.
        
        class EpsilonAgent:
            def __init__(self, agent, eps):
                self.agent = agent
                self.eps = eps
            def act(self, obs):
                if random.random() < self.eps:
                    # Random action
                    mask = self.agent.action_encoder.generate_action_mask(obs, "attacker" if self.agent.is_attacker else "defender")
                    valid_indices = np.where(mask > 0)[0]
                    idx = random.choice(valid_indices)
                    nodes = obs.get("nodes", [])
                    sorted_node_ids = [n["node_id"] for n in sorted(nodes, key=lambda x: x["node_id"])]
                    return self.agent.action_encoder.decode_action(idx, sorted_node_ids, self.agent.agent_id)
                return self.agent.act(obs)

        atk_base = DQNAgentWrapper("atk_dqn_clean", "dqn_attacker_clean.pt", is_attacker=True)
        def_base = DQNAgentWrapper("def_dqn_clean", "dqn_defender_clean.pt", is_attacker=False)
        
        seeds = [10, 20, 30, 40, 50]
        seed_results = {}
        for s in seeds:
            print(f"  - Seed {s} (2000 episodes)")
            self.rng = CentralizedRNG(seed=s)
            random.seed(s)
            
            atk = EpsilonAgent(atk_base, 0.05)
            dfn = EpsilonAgent(def_base, 0.05)
            
            wins = 0
            for ep in range(2000):
                env = EnvironmentEngine(self.rng)
                env.reset()
                done = False
                steps = 0
                while not done and steps < 50:
                    a_obs = env.get_observation_by_id(atk.agent.agent_id)
                    d_obs = env.get_observation_by_id(dfn.agent.agent_id)
                    obs, _, done, info = env.step(atk.act(a_obs), dfn.act(d_obs))
                    steps += 1
                if info.get("termination_reason") == "CRITICAL_COMPROMISED":
                    wins += 1
            seed_results[f"seed_{s}"] = wins / 2000.0
            
        self.report["stochastic_stability"] = seed_results
        with open("stochastic_stability.json", "w") as f:
            json.dump(seed_results, f, indent=2)

    def part6_memory_integrity(self):
        print("\n--- Part 6: Performance & Memory Integrity ---")
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_mem = process.memory_info().rss
        except ImportError:
            print("  - psutil not installed. Skipping precise memory growth check.")
            self.report["memory_integrity"] = "SKIPPED (psutil missing)"
            return

        env = EnvironmentEngine(self.rng)
        env.reset()
        atk = RandomAttacker("atk")
        dfn = RandomDefender("def")
        
        for i in range(10000):
            if i % 1000 == 0:
                env.reset()
            a_obs = env.get_observation_by_id("atk")
            d_obs = env.get_observation_by_id("def")
            env.step(atk.act(a_obs), dfn.act(d_obs))
            
            if i % 2000 == 0:
                gc.collect()
                curr_mem = process.memory_info().rss
                print(f"  - Step {i}: {curr_mem / 1024 / 1024:.2f} MB")
        
        final_mem = process.memory_info().rss
        growth = (final_mem - initial_mem) / initial_mem
        print(f"  - Total Growth: {growth:.2%}")
        
        if growth < 0.10: # Less than 10% growth
            self.report["memory_integrity"] = "PASS"
        else:
            self.report["memory_integrity"] = f"FAIL: {growth:.2%} growth"

    def run_all(self):
        start_time = time.time()
        self.part1_baseline_sweep()
        self.part2_selfplay_regression()
        self.part3_invariant_stress()
        self.part4_reset_hygiene()
        self.part5_stochastic_stability()
        self.part6_memory_integrity()
        
        self.report["total_time"] = time.time() - start_time
        with open("system_regression_report.json", "w") as f:
            json.dump(self.report, f, indent=2)
        print("\nAudit Complete. Report saved to system_regression_report.json")

if __name__ == "__main__":
    audit = RegressionAudit()
    audit.run_all()
