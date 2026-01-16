import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import random
from matplotlib.collections import LineCollection

# Ensure these match your actual file names
from database import map_loader
from a_star import get_node_by_street
from tsp_aco import ACOSolver, compute_distance_matrix


def visualize_aco_tsp(map_name, targets_count=5):
    # 1. Load Map
    print("Loading Map...")
    # Adjust dist=1000 if you want a larger/smaller area
    G = map_loader.load_map(map_name)

    # 2. Select Targets
    start_node = list(G.nodes)[0]
    targets = [start_node]

    # Pick random distinct nodes for the rest
    all_nodes = list(G.nodes)
    while len(targets) < targets_count:
        cand = random.choice(all_nodes)
        if cand not in targets:
            targets.append(cand)

    print(f"Selected {len(targets)} targets.")

    # 3. Pre-compute Geometry (Distance Matrix & Paths)
    dist_matrix, paths_cache = compute_distance_matrix(G, targets)

    # 4. Setup Visualization
    print("Initializing Plot...")
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color='#e0e0e0')

    # Plot Targets
    t_lons = [G.nodes[t]['x'] for t in targets]
    t_lats = [G.nodes[t]['y'] for t in targets]
    ax.scatter(t_lons, t_lats, c='red', s=50, zorder=10, label='Targets')
    ax.scatter(t_lons[0], t_lats[0], c='gold', s=100, zorder=11, label='Start')

    # --- PREPARE LINES FOR ALL PAIRS ---
    lines = []
    pair_indices = []

    for i in range(targets_count):
        for j in range(i + 1, targets_count):
            path_nodes = paths_cache.get((i, j))
            if path_nodes:
                coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path_nodes]
                lines.append(coords)
                pair_indices.append((i, j))

    # Create the Collection (initially invisible)
    lc = LineCollection(lines, colors='blue', linewidths=2, alpha=0.0, zorder=5)
    ax.add_collection(lc)

    # Best Tour Collection (Green)
    best_tour_lc = LineCollection([], colors='green', linewidths=4, alpha=0.9, zorder=6)
    ax.add_collection(best_tour_lc)

    plt.title("Initializing ACO...")
    plt.legend()

    # 5. Run ACO Generator
    aco = ACOSolver(dist_matrix, targets, n_ants=10, n_iterations=30, evaporation=0.1)
    generator = aco.solve_step_by_step()

    print("Starting Animation...")

    try:
        for i, best_tour, best_cost, pheromones in generator:

            # --- UPDATE PHEROMONE LINES ---
            max_pher = np.max(pheromones)
            min_pher = np.min(pheromones)

            new_alphas = []
            for (u, v) in pair_indices:
                # FIX: Use single direction value (matrix is symmetric)
                val = pheromones[u][v]

                # Normalize to 0.0 - 1.0 range
                if max_pher > min_pher:
                    norm = (val - min_pher) / (max_pher - min_pher)
                else:
                    norm = 0.0

                # Scale to visibility (e.g., 0.05 to 1.0)
                alpha = 0.05 + (norm * 0.95)

                # SAFETY CLIP: Ensure strictly within [0, 1]
                alpha = np.clip(alpha, 0.0, 1.0)

                new_alphas.append(alpha)

            lc.set_alpha(new_alphas)

            # --- UPDATE BEST TOUR HIGHLIGHT ---
            if best_tour:
                best_segments = []
                for k in range(len(best_tour) - 1):
                    u, v = best_tour[k], best_tour[k + 1]
                    path_nodes = paths_cache.get((u, v))
                    if path_nodes:
                        coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path_nodes]
                        best_segments.append(coords)

                best_tour_lc.set_segments(best_segments)

            ax.set_title(f"ACO Iteration: {i + 1} | Best Cost: {best_cost:.0f}m")
            plt.pause(0.1)

        print("Optimization Complete.")
        plt.show()

    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    # Ensure this points to your file
    visualize_aco_tsp("../database/Ramat_Sharet_Jerusalem_Israel_1000m.graphml")