import time
import random
import pandas as pd

from map_loader.map_loader import load_map

from solutions.aco import ACOSolver
from solutions.q_learning import QLearningSolver
from solutions.a_star import AStarSolver

def get_reachable_pair(G, min_steps=5, max_steps=25):
    """
    Performs a random walk to find a start and goal node that are
    guaranteed to be connected and relatively close to each other.
    """
    all_nodes = list(G.nodes)
    start = random.choice(all_nodes)
    current = start

    # Walk a random number of edges
    steps = random.randint(min_steps, max_steps)
    for _ in range(steps):
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break  # Hit a dead end, stop walking
        current = random.choice(neighbors)

    # Ensure we didn't just stay in place
    if start == current:
        return get_reachable_pair(G, min_steps, max_steps)

    return start, current


def run_evaluation_converged_only(G, num_scenarios=100):
    results = []
    attempts = 0

    print(f"Starting evaluation: seeking {num_scenarios} converged scenarios...")

    while len(results) < num_scenarios:
        attempts += 1

        # Use the random walk to pick points with a higher chance of convergence
        start, end = get_reachable_pair(G, min_steps=10, max_steps=30)

        # 1. A* Search
        solver_astar = AStarSolver(G)
        t0 = time.time()
        path_a, cost_a, exp_a = solver_astar.solve(start, end)
        t_a = time.time() - t0

        if cost_a == float('inf'):
            continue  # Skip if completely disconnected

        # 2. Ant Colony Optimization
        solver_aco = ACOSolver(G, n_ants=5, n_iterations=15)
        t0 = time.time()
        path_aco, cost_aco, exp_aco = solver_aco.solve(start, end)
        t_aco = time.time() - t0

        # 3. Q-Learning
        solver_ql = QLearningSolver(G, episodes=100)
        t0 = time.time()
        path_ql, cost_ql, exp_ql = solver_ql.solve(start, end)
        t_ql = time.time() - t0

        # Only record the scenario if BOTH algorithms found a path
        if cost_aco < float('inf') and cost_ql < float('inf'):
            results.append({
                'Scenario': len(results) + 1,
                'A*_Cost': cost_a, 'A*_Time': t_a, 'A*_Nodes': exp_a,
                'ACO_Cost': cost_aco, 'ACO_Time': t_aco, 'ACO_Nodes': exp_aco,
                'QL_Cost': cost_ql, 'QL_Time': t_ql, 'QL_Nodes': exp_ql
            })

            if len(results) % 5 == 0:
                print(
                    f"Found {len(results)}/{num_scenarios} converged scenarios (Total attempts so far: {attempts})...")

    print(f"\nEvaluation complete. Took {attempts} total attempts to find {num_scenarios} converged paths.")

    # Create comparison table
    df = pd.DataFrame(results)

    # Calculate summary metrics
    summary = pd.DataFrame({
        'Algorithm': ['A*', 'ACO', 'Q-Learning'],
        'Avg Cost': [df['A*_Cost'].mean(), df['ACO_Cost'].mean(), df['QL_Cost'].mean()],
        'Avg Time (s)': [df['A*_Time'].mean(), df['ACO_Time'].mean(), df['QL_Time'].mean()],
        'Avg Nodes Expanded': [df['A*_Nodes'].mean(), df['ACO_Nodes'].mean(), df['QL_Nodes'].mean()]
    })

    print("\n--- Final Comparison Summary (Converged Runs Only) ---")
    print(summary.to_string(index=False))

    # Save to CSV for easy import into Word/Excel
    df.to_csv("algorithm_comparison_converged.csv", index=False)
    return df, summary


if __name__ == "__main__":
    G = load_map("../maps/ELTA_Square_Ashdod_1000m.graphml")
    df, summary = run_evaluation_converged_only(G, num_scenarios=100)
