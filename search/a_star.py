import heapq
import math
import networkx as nx
import osmnx as ox
from database import load_map


class AStarSolver:
    def __init__(self, graph):
        self.graph = graph

    def heuristic(self, node_u, node_v):
        point_u = self.graph.nodes[node_u]
        point_v = self.graph.nodes[node_v]
        return ox.distance.great_circle(point_u['y'], point_u['x'], point_v['y'], point_v['x'])

    def solve(self, start_node, goal_node):
        """Standard solve method that returns the final result immediately."""
        # We reuse the generator but consume it entirely to get the last result
        generator = self.solve_step_by_step(start_node, goal_node)
        last_state = None
        for state in generator:
            last_state = state

        # Unpack the last state to return standard format
        if last_state and last_state['path']:
            return last_state['path'], last_state['cost'], last_state['expanded']
        return None, float('inf'), 0

    def solve_step_by_step(self, start_node, goal_node):
        """
        Generator function for Visualization.
        Yields a dictionary of the current state at each step.
        """
        open_set = []
        heapq.heappush(open_set, (0, start_node))

        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes}
        g_score[start_node] = 0

        f_score = {node: float('inf') for node in self.graph.nodes}
        f_score[start_node] = self.heuristic(start_node, goal_node)

        closed_set = set()  # For visualization only
        nodes_expanded = 0

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            # --- YIELD CURRENT STATE FOR VISUALIZATION ---
            # We yield: current node, open_set list, closed_set, etc.
            yield {
                'current_node': current,
                'open_set': [item[1] for item in open_set],  # Just node IDs
                'closed_set': list(closed_set),
                'path': None,  # Path is not ready yet
                'finished': False,
                'expanded': nodes_expanded,
                'cost': 0
            }

            if current == goal_node:
                final_path = self.reconstruct_path(came_from, current)
                yield {
                    'current_node': current,
                    'open_set': [],
                    'closed_set': list(closed_set),
                    'path': final_path,
                    'finished': True,
                    'expanded': nodes_expanded,
                    'cost': g_score[goal_node]
                }
                return

            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                edge_len = edge_data[0]['length']
                tentative_g = g_score[current] + edge_len

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]


# Copy the helper function here too so it's accessible
def get_node_by_street(G, street_name):
    search_term = street_name.lower().strip()
    for u, v, data in G.edges(data=True):
        if 'name' in data:
            names = data['name'] if isinstance(data['name'], list) else [data['name']]
            for n in names:
                if search_term in n.lower():
                    return u
    return None

# --- Visualization & Testing Block ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. Load the map (Using your local file if it exists)
    map_name = "../database/Ramat_Sharet_Jerusalem_Israel_1000m.graphml"
    print(f"Loading map: {map_name}...")
    G = load_map(map_name)

    # 2. Pick random Start and End points
    # (We convert the graph nodes view to a list to pick by index)
    all_nodes = list(G.nodes)
    start = get_node_by_street(G, 'מרץ דוד')
    end = get_node_by_street(G, 'הפסגה')

    print(f"Start Node: {start}")
    print(f"Goal Node: {end}")

    # 3. Run A*
    solver = AStarSolver(G)
    print("Running A* Search...")
    path, cost, expanded = solver.solve(start, end)

    if path:
        print(f"Path Found!")
        print(f"Total Distance: {cost:.2f} meters")
        print(f"Nodes Visited: {expanded}")

        # 4. Visualize
        print("Plotting result...")
        # plot_graph_route highlights the path nodes/edges in red (default)
        fig, ax = ox.plot_graph_route(G, path, node_size=0, edge_linewidth=0.5)
    else:
        print("No path found.")