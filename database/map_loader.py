import osmnx as ox
import os


def download_map(place_name="Rehavia, Jerusalem, Israel", dist=1000, filename=None):
    """
    Acquires the map graph.
    1. Tries to load from a local file first.
    2. If file doesn't exist, downloads from OSM and saves it.

    Args:
        place_name (str): Address/Name for the center point.
        dist (int): Radius in meters.
        filename (str, optional): Custom filename. If None, generates one from the place name.

    Returns:
        G: The NetworkX MultiDiGraph object.
    """

    print(f"[DOWNLOAD] Local file not found. Downloading {place_name} (r={dist}m)...")
    try:
        G = ox.graph_from_address(place_name, dist=dist, network_type='walk')

        # Save to disk
        ox.save_graphml(G, filepath=filename)
        print(f"[SAVE] Map saved successfully to: {filename}")

    except Exception as e:
        print(f"[ERROR] Could not download map: {e}")
        return None

    return G

def load_map(filename : str):
    #  Check if file exists (The Cache Hit)
    if os.path.exists(filename):
        print(f"[CACHE] Loading map from local file: {filename}")
        # GraphML is the standard format for NetworkX graphs
        G = ox.load_graphml(filename)
    else:
        print(f"[ERROR] File not found: {filename}")
        return None

    return G

# --- Test Block ---
# This runs only if you execute this file directly (not when importing it)
if __name__ == "__main__":
    # Example usage:
    download_map("Ramat Shared Jerusalem Israel", 1000)
    graph = load_map("Ramat_Sharet_Jerusalem_Israel_1000m.graphml")

    if graph:
        print(f"Successfully loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")