import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import random
from matplotlib.collections import LineCollection

import map_loader
from a_star import get_node_by_street
from aco_tsp import ACOSolver, compute_distance_matrix


def visualize_aco_tsp(map_name, targets_count=5):
    # 1. Load Map
    print("Loading Map...")
    G = map_loader.load_map(map_name, dist=1000)

    # 2. Select Targets
    # Let's fix the start node and pick random others
    start_node = list(G.nodes)[0]
    # Alternatively use specific streets:
    # start_node = get_node_by_street(G, 'Herzl')

    targets = [start_node]

    # Pick random distinct nodes for the rest
    all_nodes = list(G.nodes)
    while len(targets) < targets_count:
        cand = random.choice(all_nodes)
        if cand not in targets:
            targets.append(cand)

    print(f"Selected {len(targets)} targets.")

    # 3. Pre-compute Geometry (Distance Matrix & Paths)
    # We need the actual lat/lon paths for the visualization
    dist_matrix, paths_cache = compute_distance_matrix(G, targets)

    # 4. Setup Visualization
    print("Initializing Plot...")
    # Plot base graph in light grey
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color='#e0e0e0')

    # Plot Targets (Red Dots)
    t_lons = [G.nodes[t]['x'] for t in targets]
    t_lats = [G.nodes[t]['y'] for t in targets]
    ax.scatter(t_lons, t_lats, c='red', s=50, zorder=10, label='Targets')
    ax.scatter(t_lons[0], t_lats[0], c='gold', s=100, zorder=11, label='Start')  # Highlight start

    # --- PREPARE LINES FOR ALL PAIRS ---
    # We create a LineCollection for every possible connection (i -> j)
    # This allows us to update their opacity instantly without redrawing everything.
    lines = []
    pair_indices = []  # To map line index back to (i, j) matrix index

    for i in range(targets_count):
        for j in range(i + 1, targets_count):
            # Get the street path coordinates
            path_nodes = paths_cache.get((i, j))
            if path_nodes:
                # Convert list of node IDs to list of (x, y) tuples
                coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path_nodes]
                lines.append(coords)
                pair_indices.append((i, j))

    # Create the Collection (initially invisible/low alpha)
    # Blue lines for pheromones
    lc = LineCollection(lines, colors='blue', linewidths=2, alpha=0.0, zorder=5)
    ax.add_collection(lc)

    # Also create a separate collection for the BEST TOUR (Highlighted in Green)
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
            # Map pheromone value to Opacity (0.0 to 1.0)
            # We normalize based on min/max pheromone in current matrix for contrast
            max_pher = np.max(pheromones)
            min_pher = np.min(pheromones)

            # Calculate alphas for all lines
            new_alphas = []
            for (u, v) in pair_indices:
                # Get pheromone strength for this edge (undirected sum)
                p_strength = pheromones[u][v] + pheromones[v][u]

                # Normalize to 0.0 - 1.0 range
                if max_pher > min_pher:
                    norm = (p_strength - min_pher) / (max_pher - min_pher)
                else:
                    norm = 0.1

                # Clip to ensure visibility range (e.g., 0.05 to 0.8)
                alpha = 0.05 + (norm * 0.75)
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

            # Update Title
            ax.set_title(f"ACO Iteration: {i + 1} | Best Cost: {best_cost:.0f}m")

            # Refresh plot
            plt.pause(0.1)

        print("Optimization Complete.")
        plt.show()

    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    visualize_aco_tsp("Ramat_Sharet_Jerusalem_Israel")