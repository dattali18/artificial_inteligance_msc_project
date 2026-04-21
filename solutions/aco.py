import random
import osmnx as ox


class ACOSolver:
    def __init__(self, graph, n_ants=10, n_iterations=15, decay=0.1, alpha=1, beta=3):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance (distance to goal)
        self.pheromones = {}

        for u, v, k in self.graph.edges(keys=True):
            self.pheromones[(u, v)] = 1.0

    def _heuristic_distance(self, node_u, node_v):
        point_u = self.graph.nodes[node_u]
        point_v = self.graph.nodes[node_v]
        return ox.distance.great_circle(point_u['y'], point_u['x'], point_v['y'], point_v['x'])

    def solve(self, start_node, goal_node):
        best_path = None
        best_cost = float('inf')
        nodes_expanded = 0

        for iteration in range(self.n_iterations):
            paths = []
            for ant in range(self.n_ants):
                path, cost, expanded = self._construct_path(start_node, goal_node)
                nodes_expanded += expanded
                if path:
                    paths.append((path, cost))
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path

            self._update_pheromones(paths)

        return best_path, best_cost, nodes_expanded

    def _construct_path(self, start, goal):
        current = start
        path = [current]
        cost = 0
        expanded = 0
        visited = set([start])

        while current != goal:
            neighbors = list(self.graph.neighbors(current))
            unvisited_neighbors = [n for n in neighbors if n not in visited]

            if not unvisited_neighbors:
                return None, float('inf'), expanded  # Dead end trap

            expanded += 1
            probabilities = []
            denominator = 0.0

            for neighbor in unvisited_neighbors:
                pheromone = self.pheromones.get((current, neighbor), 1.0)

                # Heuristic: Inverse of distance to the goal.
                # Closer to goal = much higher probability of being chosen
                dist_to_goal = self._heuristic_distance(neighbor, goal)
                heuristic_val = 1.0 / (dist_to_goal + 1e-5)

                prob = (pheromone ** self.alpha) * (heuristic_val ** self.beta)
                probabilities.append(prob)
                denominator += prob

            if denominator == 0:
                next_node = random.choice(unvisited_neighbors)
            else:
                probabilities = [p / denominator for p in probabilities]
                next_node = random.choices(unvisited_neighbors, weights=probabilities, k=1)[0]

            edge_data = self.graph.get_edge_data(current, next_node)[0]
            cost += edge_data.get('length', 1.0)
            current = next_node
            path.append(current)
            visited.add(current)

        return path, cost, expanded

    def _update_pheromones(self, paths):
        for edge in self.pheromones:
            self.pheromones[edge] *= (1.0 - self.decay)

        for path, cost in paths:
            pheromone_deposit = 10000.0 / cost if cost > 0 else 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if (u, v) in self.pheromones:
                    self.pheromones[(u, v)] += pheromone_deposit