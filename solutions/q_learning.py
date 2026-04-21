import random
import osmnx as ox


class QLearningSolver:
    def __init__(self, graph, episodes=100, alpha=0.2, gamma=0.9, epsilon=0.9, epsilon_decay=0.95):
        self.graph = graph
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def _get_distance(self, node_u, node_v):
        point_u = self.graph.nodes[node_u]
        point_v = self.graph.nodes[node_v]
        return ox.distance.great_circle(point_u['y'], point_u['x'], point_v['y'], point_v['x'])

    def solve(self, start_node, goal_node):
        nodes_expanded = 0

        # Training Phase
        for episode in range(self.episodes):
            current = start_node
            visited = set([current])

            for step in range(500):  # Max steps per episode
                neighbors = list(self.graph.neighbors(current))
                if not neighbors:
                    break

                nodes_expanded += 1

                # Epsilon-greedy
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(neighbors)
                else:
                    q_values = [self._get_q(current, n) for n in neighbors]
                    max_q = max(q_values)
                    best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
                    action = random.choice(best_actions)

                # Reward Shaping Strategy (Dense Rewards)
                dist_current_to_goal = self._get_distance(current, goal_node)
                dist_next_to_goal = self._get_distance(action, goal_node)
                edge_data = self.graph.get_edge_data(current, action)[0]
                edge_len = edge_data.get('length', 1.0)

                if action == goal_node:
                    reward = 10000.0
                elif action in visited:
                    reward = -edge_len * 5  # High penalty for loops
                else:
                    # Positive reward if we moved closer, negative if we moved away
                    progress = dist_current_to_goal - dist_next_to_goal
                    reward = progress - (edge_len * 0.1)

                # Q-Value Update
                next_neighbors = list(self.graph.neighbors(action))
                max_next_q = max([self._get_q(action, n) for n in next_neighbors]) if next_neighbors else 0.0

                current_q = self._get_q(current, action)
                new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
                self.q_table[(current, action)] = new_q

                current = action
                visited.add(current)

                if current == goal_node:
                    break

            self.epsilon *= self.epsilon_decay

        # Extraction Phase (Greedy Path)
        path = [start_node]
        current = start_node
        cost = 0
        visited_extract = set([current])

        while current != goal_node:
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                return None, float('inf'), nodes_expanded

            q_values = [self._get_q(current, n) for n in neighbors]
            max_q = max(q_values)

            # If Q-values are 0, it means it never explored successfully here.
            # Fallback to geographic distance to prevent getting stuck
            if max_q == 0.0:
                best_actions = neighbors
                best_actions.sort(key=lambda n: self._get_distance(n, goal_node))
                action = best_actions[0]
            else:
                best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q and n not in visited_extract]
                if not best_actions:
                    return None, float('inf'), nodes_expanded
                action = best_actions[0]

            edge_data = self.graph.get_edge_data(current, action)[0]
            cost += edge_data.get('length', 1.0)

            current = action
            path.append(current)
            visited_extract.add(current)

            if len(path) > 1000:
                return None, float('inf'), nodes_expanded

        return path, cost, nodes_expanded