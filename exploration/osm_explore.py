import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# 1. SETUP: Define location and radius
# Instead of relying on a boundary, we define a center point and a distance.
place_name = "Ramat Sharet, Jerusalem, Israel"
dist_meters = 1000  # 1 km radius around the center

print(f"Downloading map data for: {place_name} (Radius: {dist_meters}m)...")

# 2. DOWNLOAD: Get the graph from OSM using Address + Distance
# 'dist' defines how many meters from the center point to download.
# 'network_type'='walk' ensures we get sidewalks/paths suitable for your project.
G = ox.graph_from_address(place_name, dist=dist_meters, network_type='walk')

print("Map downloaded successfully!")

# 3. VISUALIZE: Plot the raw map
# node_size=0 makes it look cleaner (hides the dots for intersections)
fig, ax = ox.plot_graph(G, node_size=0, edge_color='w', edge_linewidth=0.5)

# 4. INSPECT: Understand the Data Structure
# The graph G consists of Nodes (points with ID, lat, lon) and Edges (paths between nodes).

# Pick a random node
first_node_id = list(G.nodes)[0]
node_data = G.nodes[first_node_id]
print(f"\n--- Node Inspection (ID: {first_node_id}) ---")
print(f"Data: {node_data}")

# Pick a random edge
# Edges are defined by (u, v, key). 'data' holds the attributes.
first_edge = list(G.edges(data=True))[0]
u, v, data = first_edge
print(f"\n--- Edge Inspection (From {u} to {v}) ---")
print(f"Data: {data}")
# Look for 'length' in the output—this is your cost function!

# 5. TEST: Basic Routing (Sanity Check)
# Calculate shortest path between the first and last node in the downloaded set
start_node = list(G.nodes)[0]
end_node = list(G.nodes)[-1]

# Check if a path actually exists (graph might be disconnected)
try:
    route = ox.shortest_path(G, start_node, end_node, weight='length')
    if route:
        print(f"\nRoute found! It passes through {len(route)} nodes.")
        fig, ax = ox.plot_graph_route(G, route, node_size=0)
    else:
        print("\nNo path found (nodes might be in disconnected components).")
except Exception as e:
    print(f"\nRouting failed: {e}")