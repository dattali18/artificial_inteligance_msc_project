import heapq
import osmnx as ox


class AStarSolver:
    def __init__(self, graph):
        self.graph = graph

    def heuristic(self, node_u, node_v):
        point_u = self.graph.nodes[node_u]
        point_v = self.graph.nodes[node_v]
        # Using great circle distance as the admissible heuristic
        return ox.distance.great_circle(point_u['y'], point_u['x'], point_v['y'], point_v['x'])

    def solve(self, start_node, goal_node):
        open_set = []
        heapq.heappush(open_set, (0, start_node))

        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes}
        g_score[start_node] = 0

        f_score = {node: float('inf') for node in self.graph.nodes}
        f_score[start_node] = self.heuristic(start_node, goal_node)

        closed_set = set()
        nodes_expanded = 0

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            # Goal reached
            if current == goal_node:
                path = self._reconstruct_path(came_from, current)
                cost = g_score[goal_node]
                return path, cost, nodes_expanded

            # Explore neighbors
            for neighbor in self.graph.neighbors(current):
                # OSMnx uses MultiDiGraphs; [0] grabs the primary edge data
                edge_data = self.graph.get_edge_data(current, neighbor)[0]
                edge_len = edge_data.get('length', 1.0)

                tentative_g = g_score[current] + edge_len

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # Return failure if the open set is exhausted without finding the goal
        return None, float('inf'), nodes_expanded

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]