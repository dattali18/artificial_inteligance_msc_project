import osmnx as ox
import os


def download_map(place_name="Rehavia, Jerusalem, Israel", dist=1000, filename=None):
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


if __name__ == "__main__":
    # Example usage:
    download_map("ELTA Square Ashdod", 1000, filename="../maps/ELTA_Square_Ashdod_1000m.graphml")
    # graph = load_map("Ramat_Sharet_Jerusalem_Israel_1000m.graphml")
    #
    # if graph:
    #     print(f"Successfully loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")