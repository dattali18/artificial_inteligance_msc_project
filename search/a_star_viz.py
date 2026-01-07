import matplotlib.pyplot as plt
import osmnx as ox
import database as map_loader
from a_star import AStarSolver, get_node_by_street


def visualize_search(map_name, start_street, end_street):
    # 1. Load Map
    print("Loading map...")
    G = map_loader.load_map(map_name)

    # 2. Find Nodes
    start_node = get_node_by_street(G, start_street)
    end_node = get_node_by_street(G, end_street)

    if not start_node or not end_node:
        print("Error: Could not find one of the streets.")
        return

    # 3. Setup Plot
    # We plot the base graph ONCE.
    print("Initializing visualization...")
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color='#999999', edge_linewidth=0.5)

    # Create empty scatter plots that we will update in the loop
    # Red dots = Closed Set (Visited)
    # Green dots = Open Set (Frontier/Neighbors to visit)
    # Blue dot = Current Head
    closed_scatter = ax.scatter([], [], c='red', s=10, label='Visited', zorder=2)
    open_scatter = ax.scatter([], [], c='green', s=10, label='Frontier', zorder=3)
    head_scatter = ax.scatter([], [], c='blue', s=30, label='Current', zorder=4)

    plt.legend(loc='upper left')
    plt.title("A* Search Progress")

    # 4. Run the Generator Loop
    solver = AStarSolver(G)
    generator = solver.solve_step_by_step(start_node, end_node)

    step_count = 0

    print("Starting animation... (Close window to stop)")

    try:
        for state in generator:
            step_count += 1

            # OPTIMIZATION: Only redraw every N steps to make it faster
            # Change this % 5 to % 1 if you want to see literally every single step
            if step_count % 5 == 0 or state['finished']:

                # Get coordinates for the sets
                closed_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in state['closed_set']]
                open_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in state['open_set']]
                curr_coords = [(G.nodes[state['current_node']]['x'], G.nodes[state['current_node']]['y'])]

                # Update the scatter plot data
                # Matplotlib requires (x, y) arrays transposed
                if closed_coords:
                    closed_scatter.set_offsets(closed_coords)
                if open_coords:
                    open_scatter.set_offsets(open_coords)
                head_scatter.set_offsets(curr_coords)

                # Update Title
                ax.set_title(f"A* Search: {step_count} Steps Expanded")

                # Pause to refresh the GUI
                plt.pause(1)

            if state['finished']:
                print("Goal Reached!")
                # Plot the final route line on top
                if state['path']:
                    ox.plot_graph_route(G, state['path'], ax=ax, route_color='blue', route_linewidth=4, node_size=0)
                plt.show()  # Keep window open at the end
                break

    except KeyboardInterrupt:
        print("Visualization stopped by user.")


if __name__ == "__main__":
    MAP_NAME = "../database/Ramat_Sharet_Jerusalem_Israel_1000m.graphml"
    START = "מרץ דוד"
    END = "הפסגה"

    visualize_search(MAP_NAME, START, END)