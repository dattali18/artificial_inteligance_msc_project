import osmnx as ox
import networkx as nx

# 1. SETUP: Load the map again (or reuse 'G' from previous step)
place_name = "Rehavia, Jerusalem, Israel"
G = ox.graph_from_address(place_name, dist=500, network_type='walk')

# --- COMPONENT 1: STATES (Nodes) ---
# In OSM, every intersection or point is a "Node" with a unique ID.
# For A*, we need coordinates (x, y) to calculate the Heuristic (Euclidean distance).

node_id = list(G.nodes)[0]  # Get the first node ID
node_data = G.nodes[node_id]

print(f"--- STATE REPRESENTATION (Node {node_id}) ---")
print(f"Latitude (y):  {node_data['y']}")
print(f"Longitude (x): {node_data['x']}")
# NOTE: In your A* heuristic, you will use these x/y values to estimate distance to the goal.


# --- COMPONENT 2: SUCCESSOR FUNCTION (Neighbors) ---
# In Search/RL, you need to know: "From current node U, where can I go?"
# NetworkX makes this easy with G.neighbors(u).

print(f"\n--- SUCCESSOR FUNCTION (Neighbors of {node_id}) ---")
neighbors = list(G.neighbors(node_id))
print(f"From Node {node_id}, you can move to: {neighbors}")

# --- COMPONENT 3: COST FUNCTION (Edges) ---
# The cost of moving from U to V is stored in the edge data.
# Note: It's a MultiDiGraph, so we access edges via G[u][v][key]. Usually key=0.

if neighbors:
    neighbor_id = neighbors[0]
    # Access the edge attributes between the node and its first neighbor
    edge_data = G.get_edge_data(node_id, neighbor_id)[0]

    print(f"\n--- COST FUNCTION (Edge from {node_id} to {neighbor_id}) ---")
    print(f"Full Data: {edge_data}")
    print(f"Cost (Length): {edge_data['length']} meters")

    # Sometimes edges have extra info like 'grade' (steepness) or 'surface' (paved/unpaved)
    # You could use these later for your 'Safe/Accessible' routing constraints!
    if 'name' in edge_data:
        print(f"Street Name: {edge_data['name']}")