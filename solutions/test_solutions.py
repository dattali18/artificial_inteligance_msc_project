import time
import random
import pandas as pd

from map_loader.map_loader import load_map

from solutions.aco import ACOSolver
from solutions.q_learning import QLearningSolver
from solutions.a_star import AStarSolver

def run_evaluation(G, num_scenarios=100):
    all_nodes = list(G.nodes)
    results = []

    print(f"Starting evaluation of {num_scenarios} scenarios...")

    for i in range(num_scenarios):
        start = random.choice(all_nodes)
        end = random.choice(all_nodes)

        # Make sure start and end aren't the same
        while start == end:
            end = random.choice(all_nodes)

        # 1. A* Search
        solver_astar = AStarSolver(G)
        t0 = time.time()
        path_a, cost_a, exp_a = solver_astar.solve(start, end)
        t_a = time.time() - t0

        # 2. Ant Colony Optimization
        solver_aco = ACOSolver(G, n_ants=5, n_iterations=10)  # Lowered params for speed
        t0 = time.time()
        path_aco, cost_aco, exp_aco = solver_aco.solve(start, end)
        t_aco = time.time() - t0

        # 3. Q-Learning
        solver_ql = QLearningSolver(G, episodes=50)  # Lowered params for speed
        t0 = time.time()
        path_ql, cost_ql, exp_ql = solver_ql.solve(start, end)
        t_ql = time.time() - t0

        results.append({
            'Scenario': i + 1,
            'A*_Cost': cost_a, 'A*_Time': t_a, 'A*_Nodes': exp_a,
            'ACO_Cost': cost_aco, 'ACO_Time': t_aco, 'ACO_Nodes': exp_aco,
            'QL_Cost': cost_ql, 'QL_Time': t_ql, 'QL_Nodes': exp_ql
        })

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_scenarios} scenarios...")

    # Create comparison table
    df = pd.DataFrame(results)

    # Calculate summary metrics for the final report
    summary = pd.DataFrame({
        'Algorithm': ['A*', 'ACO', 'Q-Learning'],
        'Avg Cost': [df['A*_Cost'].mean(), df['ACO_Cost'].mean(), df['QL_Cost'].mean()],
        'Avg Time (s)': [df['A*_Time'].mean(), df['ACO_Time'].mean(), df['QL_Time'].mean()],
        'Avg Nodes Expanded': [df['A*_Nodes'].mean(), df['ACO_Nodes'].mean(), df['QL_Nodes'].mean()]
    })

    print("\n--- Final Comparison Summary ---")
    print(summary.to_string(index=False))

    # Optionally save to CSV for your Word document
    df.to_csv("algorithm_comparison_results.csv", index=False)
    return df, summary


if __name__ == "__main__":
    # Load your map here
    map_name = "../maps/ELTA_Square_Ashdod_1000m.graphml"
    G = load_map(map_name)

    # Example usage:
    df, summary = run_evaluation(G, num_scenarios=100)

    # save df

    print(summary)
    print(df.head())