from database import load_map
import osmnx as ox


def get_all_street_names(G):
    """
    Extracts all unique street names from the graph edges.
    """
    street_names = set()

    # Iterate over all edges.
    # u = start node, v = end node, data = edge attributes
    for u, v, data in G.edges(data=True):

        # Check if the edge has a 'name' attribute
        if 'name' in data:
            name = data['name']

            # Sometimes 'name' is a list (e.g., if two streets merge)
            if isinstance(name, list):
                for n in name:
                    street_names.add(n)
            else:
                street_names.add(name)

    # Remove empty names or purely numeric refs if any exist
    return sorted(list(street_names))


if __name__ == "__main__":
    # 1. Load the map using your existing loader
    # Make sure this matches the filename you already have in your database folder
    map_name = "../database/Ramat_Sharet_Jerusalem_Israel_1000m.graphml"
    G = load_map(map_name)

    if G:
        # 2. Get names
        streets = get_all_street_names(G)

        # 3. Print Summary
        print(f"\n--- Street Analysis for {map_name} ---")
        print(f"Total unique street names found: {len(streets)}")

        print("\n--- Street List ---")
        for i, street in enumerate(streets):
            print(f"{i + 1}. {street}")

        # 4. Advanced: Find the longest street (most segments)
        # This counts how many graph edges belong to each street name
        from collections import Counter

        all_segments = []
        for u, v, data in G.edges(data=True):
            if 'name' in data:
                val = data['name']
                if isinstance(val, list):
                    all_segments.extend(val)
                else:
                    all_segments.append(val)

        common = Counter(all_segments).most_common(5)
        print("\n--- Top 5 Biggest Streets (by number of segments) ---")
        for name, count in common:
            print(f"{name}: {count} segments")