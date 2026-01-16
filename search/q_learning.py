import numpy as np
import random
import osmnx as ox
from database import map_loader
from a_star import get_node_by_street


class QLearningAgent:
    def __init__(self, graph, start_node, goal_node,
                 alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        :param alpha: Learning Rate (How much we accept new info)
        :param gamma: Discount Factor (How much we care about future rewards)
        :param epsilon: Exploration Rate (1.0 = 100% random moves initially)
        """
        self.graph = graph
        self.start = start_node
        self.goal = goal_node

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # The Q-Table: A dictionary mapping state (node) -> {action (neighbor): q_value}
        # We use a dict of dicts because the number of neighbors varies per node.
        self.q_table = {}

    def get_q_value(self, state, action):
        """Helper to get Q-value, defaulting to 0.0 if not yet visited"""
        if state not in self.q_table:
            self.q_table[state] = {}
        return self.q_table[state].get(action, 0.0)

    def choose_action(self, state):
        """Epsilon-Greedy Strategy"""
        neighbors = list(self.graph.neighbors(state))
        if not neighbors:
            return None  # Dead end

        # EXPLORE: Random move
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(neighbors)

        # EXPLOIT: Choose best known move
        # Find neighbor with highest Q-value
        best_acc = -float('inf')
        best_action = neighbors[0]

        for n in neighbors:
            val = self.get_q_value(state, n)
            if val > best_acc:
                best_acc = val
                best_action = n

        return best_action

    def train(self, episodes=1000):
        print(f"Training RL Agent for {episodes} episodes...")

        for ep in range(episodes):
            state = self.start
            done = False
            steps = 0

            # Limit steps per episode to prevent infinite loops during early training
            max_steps = 200

            while not done and steps < max_steps:
                # 1. Choose Action
                action = self.choose_action(state)
                if action is None:
                    break  # Dead end

                # 2. Perform Action (Observe Reward & New State)
                next_state = action

                # REWARD FUNCTION
                # ----------------
                # Goal Reached: Big positive reward
                # Step Taken: Small negative reward (based on distance) to encourage efficiency
                # ----------------
                if next_state == self.goal:
                    reward = 1000
                    done = True
                else:
                    # Get distance of this edge
                    edge_data = self.graph.get_edge_data(state, next_state)[0]
                    dist = edge_data['length']
                    # Negative reward = cost (minimize distance)
                    reward = -dist

                    # 3. Update Q-Value (The "Learning" Part)
                # Bellman Equation: Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]

                current_q = self.get_q_value(state, action)

                # Find max Q for next state
                next_neighbors = list(self.graph.neighbors(next_state))
                if next_neighbors:
                    max_next_q = max([self.get_q_value(next_state, n) for n in next_neighbors])
                else:
                    max_next_q = 0  # Terminal state or dead end

                new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

                # Save it
                if state not in self.q_table: self.q_table[state] = {}
                self.q_table[state][action] = new_q

                state = next_state
                steps += 1

            # Decay Epsilon (Reduce randomness over time)
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            if (ep + 1) % 100 == 0:
                print(f"Episode {ep + 1}/{episodes} complete. Epsilon: {self.epsilon:.2f}")

    def get_best_path(self):
        """Uses the learned Q-Table to trace the best path without exploration."""
        path = [self.start]
        current = self.start

        # Safety limit to prevent infinite loops if policy is bad
        for _ in range(100):
            if current == self.goal:
                break

            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break

            # Greedily choose best Q-value
            best_n = max(neighbors, key=lambda n: self.get_q_value(current, n))

            path.append(best_n)
            current = best_n

        return path


# --- TEST BLOCK ---
if __name__ == "__main__":
    import map_loader
    import osmnx as ox
    import matplotlib.pyplot as plt

    # 1. Load a SMALLER map for RL (so it learns faster)
    # 500m radius is good for testing
    map_name = "Ramat_Sharet_Jerusalem_Israel_RL_Test"
    G = map_loader.load_map("Ramat Sharet, Jerusalem", dist=400, filename=None)

    # 2. Define Start/Goal
    nodes = list(G.nodes)
    start = nodes[0]
    end = nodes[50]  # Pick a node not too far away

    print(f"Start: {start}, Goal: {end}")

    # 3. Initialize and Train
    agent = QLearningAgent(G, start, end)

    # Needs enough episodes to explore the map!
    agent.train(episodes=2000)

    # 4. Extract Path
    learned_path = agent.get_best_path()
    print(f"Learned Path Length: {len(learned_path)} nodes")

    if learned_path[-1] == end:
        print("SUCCESS: Agent reached the goal!")
        ox.plot_graph_route(G, learned_path, node_size=0)
    else:
        print("FAILURE: Agent did not reach the goal (needs more training or simpler map).")
        # Plot where it got stuck
        ox.plot_graph_route(G, learned_path, node_size=0, route_color='red')