import random
import networkx as nx
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt

# Import your existing tools
import map_loader
from a_star import AStarSolver, get_node_by_street


class ACOSolver:
    def __init__(self, dist_matrix, places_indices, n_ants=10, n_iterations=100,
                 alpha=1.0, beta=2.0, evaporation=0.5, Q=100):
        """
        :param dist_matrix: 2D array of distances between all targets
        :param places_indices: List of original map node IDs [Start, Target1, Target2...]
        :param alpha: Importance of Pheromone (History)
        :param beta: Importance of Distance (Heuristic)
        :param evaporation: How fast pheromones disappear (0.0 to 1.0)
        """
        self.dist_matrix = dist_matrix
        self.places = places_indices
        self.n_points = len(places_indices)

        # ACO Parameters
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone weight
        self.beta = beta  # Visibility/Distance weight
        self.evaporation = evaporation
        self.Q = Q

        # Pheromone Matrix (initialized with small value)
        self.pheromones = np.ones((self.n_points, self.n_points)) * 0.1

    def solve(self):
        best_tour = None
        best_dist = float('inf')

        for iteration in range(self.n_iterations):
            all_tours = []

            # 1. Generate Tours for each Ant
            for ant in range(self.n_ants):
                tour = self._generate_ant_tour()
                dist = self._calculate_tour_cost(tour)
                all_tours.append((tour, dist))

                # Check if this is the global best
                if dist < best_dist:
                    best_dist = dist
                    best_tour = tour
                    print(f"New best found at iter {iteration}: {best_dist:.2f} meters")

            # 2. Update Pheromones
            self._update_pheromones(all_tours)

        return best_tour, best_dist

    def _generate_ant_tour(self):
        # Always start at index 0 (The defined Start Node)
        curr_idx = 0
        visited = {0}
        tour = [0]

        # Construct the path point by point
        for _ in range(self.n_points - 1):
            probabilities = self._calculate_probabilities(curr_idx, visited)

            # Choose next city based on probabilities
            next_city = np.random.choice(range(self.n_points), p=probabilities)

            tour.append(next_city)
            visited.add(next_city)
            curr_idx = next_city

        # Return to start
        tour.append(0)
        return tour

    def _calculate_probabilities(self, curr, visited):
        pheromones = np.power(self.pheromones[curr], self.alpha)

        # Heuristic: 1 / distance (avoid division by zero)
        # We assume distance > 0. If 0, use small epsilon.
        with np.errstate(divide='ignore'):
            visibility = np.power(1.0 / (self.dist_matrix[curr] + 1e-10), self.beta)

        # Zero out visited cities
        mask = np.ones(self.n_points)
        for v in visited:
            mask[v] = 0

        probs = pheromones * visibility * mask

        # Normalize
        total = np.sum(probs)
        if total == 0:
            # Fallback: choose uniformly among unvisited
            probs = mask
            total = np.sum(probs)

        return probs / total

    def _calculate_tour_cost(self, tour):
        dist = 0
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            dist += self.dist_matrix[u][v]
        return dist

    def _update_pheromones(self, all_tours):
        # Evaporation
        self.pheromones *= (1.0 - self.evaporation)

        # Deposit
        for tour, dist in all_tours:
            contribution = self.Q / dist
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i + 1]
                self.pheromones[u][v] += contribution
                self.pheromones[v][u] += contribution  # Symmetric


# --- PRE-PROCESSING HELPER ---
def compute_distance_matrix(graph, points_list):
    """
    Uses A* to calculate distances between all pairs of points.
    Returns: Numpy 2D Matrix, Dictionary of Paths {(u,v): [path_nodes]}
    """
    n = len(points_list)
    matrix = np.zeros((n, n))
    paths_cache = {}  # To reconstruct the map route later

    solver = AStarSolver(graph)

    print(f"Computing pairwise distances for {n} locations...")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # If we already computed i->j or j->i, reuse it (assuming undirected/walkable both ways)
            if (j, i) in paths_cache:
                matrix[i][j] = matrix[j][i]
                paths_cache[(i, j)] = paths_cache[(j, i)][::-1]  # Reverse path
                continue

            # Run A*
            path, cost, _ = solver.solve(points_list[i], points_list[j])
            if path:
                matrix[i][j] = cost
                paths_cache[(i, j)] = path
            else:
                matrix[i][j] = float('inf')  # unreachable

    return matrix, paths_cache


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Map
    map_name = "Ramat_Sharet_Jerusalem_Israel"
    G = map_loader.load_map(map_name, dist=1000)

    # 2. Define Targets (Start + 4 other places)
    # Using street names or random nodes
    start_node = get_node_by_street(G, 'מרץ דוד')  # Start/End

    # Pick 4 random nodes to be our "Deliveries"
    all_nodes = list(G.nodes)
    targets = [start_node]
    # Ensure random nodes are reachable (simple check: pick from largest component)
    # For now, just pick random ones
    for _ in range(4):
        targets.append(random.choice(all_nodes))

    print(f"\nOptimization Goal: Visit {len(targets)} locations and return to start.")

    # 3. Compute Distance Matrix (The Heavy Lifting)
    dist_matrix, paths_cache = compute_distance_matrix(G, targets)

    # 4. Run ACO
    print("\n--- Starting Ant Colony Optimization ---")
    aco = ACOSolver(dist_matrix, targets, n_ants=5, n_iterations=20)
    best_tour_indices, best_cost = aco.solve()

    print(f"\nFinal Best Tour Cost: {best_cost:.2f} meters")
    print(f"Best Order of Indices: {best_tour_indices}")

    # 5. Reconstruct the Full Street Path
    full_path_nodes = []

    for k in range(len(best_tour_indices) - 1):
        idx_a = best_tour_indices[k]
        idx_b = best_tour_indices[k + 1]

        # Get the A* path between these two stops
        segment = paths_cache[(idx_a, idx_b)]

        # Avoid duplicating the connection node
        if full_path_nodes:
            full_path_nodes.extend(segment[1:])
        else:
            full_path_nodes.extend(segment)

    # 6. Visualize
    print("Plotting full TSP route...")

    # We can plot the 'targets' as red dots to see them clearly
    fig, ax = ox.plot_graph_route(G, full_path_nodes, node_size=0, edge_linewidth=2, show=False, close=False)

    # Overlay the target points
    target_lats = [G.nodes[t]['y'] for t in targets]
    target_lons = [G.nodes[t]['x'] for t in targets]
    ax.scatter(target_lons, target_lats, c='red', s=50, zorder=5, label='Targets')
    ax.scatter(target_lons[0], target_lats[0], c='yellow', s=100, zorder=6, label='Start')

    plt.legend()
    plt.show()