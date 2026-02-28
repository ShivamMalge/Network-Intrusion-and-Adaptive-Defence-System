"""
Phase 6C Step 2: Elo Rating System for Adversarial Self-Play.
Computes competitive strength ratings from evaluation_matrix.json.
"""

import json
import os
import math

class EloRating:
    def __init__(self, initial_rating=1500, k=32):
        self.initial_rating = initial_rating
        self.k = k
        self.ratings = {}

    def get_rating(self, agent_id):
        return self.ratings.get(agent_id, self.initial_rating)

    def expected_score(self, rating_a, rating_b):
        """Probability that A beats B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_rating(self, rating, expected, actual):
        """New rating after a match."""
        return rating + self.k * (actual - expected)

def compute_elo_from_matrix(matrix_path, episodes_per_match=100):
    if not os.path.exists(matrix_path):
        print(f"Error: {matrix_path} not found.")
        return None

    with open(matrix_path, "r") as f:
        matrix = json.load(f)

    elo = EloRating(initial_rating=1500, k=32)
    
    # Initialize all agents
    attackers = list(matrix.keys())
    defenders = set()
    for atk in attackers:
        for dfnd in matrix[atk].keys():
            defenders.add(dfnd)
    defenders = sorted(list(defenders))

    # We treat each matchup as a set of N independent matches
    # to simulate the evolution of ratings if these agents played a tournament.
    # However, to avoid order-dependency issues in a static matrix, 
    # we can also use an aggregated update step.
    # We will iterate multiple times (epochs) to stabilize the ratings 
    # since this isn't a chronological stream of matches.
    
    current_ratings = {agent: 1500.0 for agent in attackers + defenders}
    
    # Static iteration to converge on ratings that explain the win rates
    for epoch in range(10):
        new_ratings = current_ratings.copy()
        for atk_id in attackers:
            for def_id in defenders:
                if def_id not in matrix[atk_id]:
                    continue
                
                win_rate = matrix[atk_id][def_id]["win_rate"] / 100.0
                
                r_atk = current_ratings[atk_id]
                r_def = current_ratings[def_id]
                
                # Expected score for attacker
                e_atk = elo.expected_score(r_atk, r_def)
                # Actual score is win_rate
                
                # Update attacker
                new_ratings[atk_id] += (elo.k / 10) * (win_rate - e_atk)
                # Update defender (zero-sum)
                new_ratings[def_id] += (elo.k / 10) * ((1.0 - win_rate) - (1.0 - e_atk))
        
        current_ratings = new_ratings

    return {
        "attackers": {atk: round(current_ratings[atk]) for atk in attackers},
        "defenders": {dfnd: round(current_ratings[dfnd]) for dfnd in defenders}
    }

def main():
    matrix_path = "evaluation_matrix.json"
    results = compute_elo_from_matrix(matrix_path)
    
    if results:
        output_path = "elo_ratings.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print("\n" + "="*40)
        print("ELO RATINGS SUMMARY")
        print("="*40)
        
        print("\nATTACKERS:")
        # Sort by rating descending
        sorted_atk = sorted(results["attackers"].items(), key=lambda x: x[1], reverse=True)
        for name, rating in sorted_atk:
            print(f"  {name:7}: {rating}")
            
        print("\nDEFENDERS:")
        sorted_def = sorted(results["defenders"].items(), key=lambda x: x[1], reverse=True)
        for name, rating in sorted_def:
            print(f"  {name:7}: {rating}")
            
        print("="*40)
        
        # Interpretation logic
        d3_elo = results["defenders"].get("D3", 1500)
        d0_elo = results["defenders"].get("D0", 1500)
        a3_elo = results["attackers"].get("A3", 1500)
        a0_elo = results["attackers"].get("A0", 1500)
        
        print("\nINTERPRETATION:")
        if d3_elo > d0_elo:
            print(f"- Genuine Monotonic Defense Improvement: D3 (+{d3_elo - d0_elo}) > D0")
        else:
            print("- Possible Defense Stagnation or Cycling detected.")
            
        if a1_elo := results["attackers"].get("A1", 1500):
             if a3_elo > a1_elo:
                 print(f"- Robust Attacker Progression: A3 (+{a3_elo - a1_elo}) > A1")
             else:
                 print("- Attacker Reached Local Peak at Cycle 1; Cycle 3 focuses on Robustness.")

if __name__ == "__main__":
    main()
