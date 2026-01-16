import random
import networkx as nx
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from database import load_map
from a_star import AStarSolver, get_node_by_street


class ACOSolver:
    def __init__(self, dist_matrix, places_indices, n_ants=10, n_iterations=100,
                 alpha=1.0, beta=2.0, evaporation=0.5, Q=100):
        self.dist_matrix = dist_matrix
        self.places = places_indices
        self.n_points = len(places_indices)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        # Initialize pheromones
        self.pheromones = np.ones((self.n_points, self.n_points)) * 0.1

    def solve_step_by_step(self):
        """
        Generator that yields the state of the algorithm at each iteration.
        Yields: (iteration_number, best_tour_so_far, best_cost_so_far, pheromone_matrix)
        """
        best_tour = None
        best_dist = float('inf')

        for iteration in range(self.n_iterations):
            all_tours = []

            # 1. Ants build solutions
            for ant in range(self.n_ants):
                tour = self._generate_ant_tour()
                dist = self._calculate_tour_cost(tour)
                all_tours.append((tour, dist))

                if dist < best_dist:
                    best_dist = dist
                    best_tour = tour

            # 2. Update Pheromones
            self._update_pheromones(all_tours)

            # 3. Yield current state to the visualizer
            # We copy the pheromones so the visualizer doesn't mess with the math
            yield iteration, best_tour, best_dist, self.pheromones.copy()

        return best_tour, best_dist

    def _generate_ant_tour(self):
        curr_idx = 0  # Start at the first node
        visited = {0}
        tour = [0]

        for _ in range(self.n_points - 1):
            probabilities = self._calculate_probabilities(curr_idx, visited)
            next_city = np.random.choice(range(self.n_points), p=probabilities)
            tour.append(next_city)
            visited.add(next_city)
            curr_idx = next_city

        tour.append(0)  # Return to start
        return tour

    def _calculate_probabilities(self, curr, visited):
        pheromones = np.power(self.pheromones[curr], self.alpha)
        with np.errstate(divide='ignore'):
            visibility = np.power(1.0 / (self.dist_matrix[curr] + 1e-10), self.beta)

        mask = np.ones(self.n_points)
        for v in visited:
            mask[v] = 0

        probs = pheromones * visibility * mask
        total = np.sum(probs)
        if total == 0:
            return mask / np.sum(mask)  # Uniform random fallback
        return probs / total

    def _calculate_tour_cost(self, tour):
        dist = 0
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            dist += self.dist_matrix[u][v]
        return dist

    def _update_pheromones(self, all_tours):
        self.pheromones *= (1.0 - self.evaporation)
        for tour, dist in all_tours:
            contribution = self.Q / dist
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i + 1]
                self.pheromones[u][v] += contribution
                self.pheromones[v][u] += contribution


def compute_distance_matrix(graph, points_list):
    """Helper to pre-calculate paths between all pairs"""
    n = len(points_list)
    matrix = np.zeros((n, n))
    paths_cache = {}

    solver = AStarSolver(graph)
    print(f"Pre-computing paths for {n} points...")

    for i in range(n):
        for j in range(n):
            if i == j: continue
            if (j, i) in paths_cache:
                matrix[i][j] = matrix[j][i]
                paths_cache[(i, j)] = paths_cache[(j, i)][::-1]
                continue

            path, cost, _ = solver.solve(points_list[i], points_list[j])
            if path:
                matrix[i][j] = cost
                paths_cache[(i, j)] = path
            else:
                matrix[i][j] = float('inf')

    return matrix, paths_cache