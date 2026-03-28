"""
flat_route.py — Find the flattest cycling route between two points in Tel Aviv.

Minimizes total elevation gain (uphill cost) rather than distance.
Uses OpenStreetMap bike network + OpenTopoData elevation API.
"""

import time
import math
import requests
import numpy as np
import networkx as nx
import osmnx as ox

# ---------------------------------------------------------------------------
# 1. Download Tel Aviv bike network
# ---------------------------------------------------------------------------

def load_graph(place: str = "Tel Aviv, Israel") -> nx.MultiDiGraph:
    print(f"Downloading bike network for {place}...")
    G = ox.graph_from_place(place, network_type="bike")
    G = ox.project_graph(G)  # project to UTM for accurate edge lengths
    G = ox.project_graph(G, to_crs="EPSG:4326")  # back to lat/lon for elevation API
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


# ---------------------------------------------------------------------------
# 2. Fetch elevations via OpenTopoData (srtm30m)
# ---------------------------------------------------------------------------

ELEVATION_API = "https://api.opentopodata.org/v1/srtm30m"
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 1.2  # seconds


def fetch_elevations(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Fetch elevation for every node and store as node attribute 'elevation'."""
    nodes = list(G.nodes(data=True))
    total = len(nodes)
    print(f"Fetching elevations for {total} nodes (batches of {BATCH_SIZE})...")

    elevations: dict[int, float] = {}

    for batch_start in range(0, total, BATCH_SIZE):
        batch = nodes[batch_start: batch_start + BATCH_SIZE]
        locations = "|".join(f"{data['y']},{data['x']}" for _, data in batch)

        try:
            resp = requests.get(
                ELEVATION_API,
                params={"locations": locations},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

            for (node_id, _), result in zip(batch, results):
                elev = result.get("elevation")
                elevations[node_id] = float(elev) if elev is not None else 0.0

        except Exception as e:
            print(f"  Warning: elevation batch {batch_start}–{batch_start+len(batch)} failed ({e}). Using 0.")
            for node_id, _ in batch:
                elevations.setdefault(node_id, 0.0)

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = math.ceil(total / BATCH_SIZE)
        print(f"  Batch {batch_num}/{total_batches} done", end="\r")

        if batch_start + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print()  # newline after \r progress

    nx.set_node_attributes(G, elevations, "elevation")
    return G


# ---------------------------------------------------------------------------
# 3. Annotate edges with elevation_gain
# ---------------------------------------------------------------------------

def annotate_elevation_gain(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add elevation_gain attribute to each edge: uphill cost only."""
    elev = nx.get_node_attributes(G, "elevation")
    for u, v, key, data in G.edges(keys=True, data=True):
        gain = max(0.0, elev.get(v, 0.0) - elev.get(u, 0.0))
        G[u][v][key]["elevation_gain"] = gain
    return G


# ---------------------------------------------------------------------------
# 4. Routing
# ---------------------------------------------------------------------------

def get_flat_route(
    G: nx.MultiDiGraph,
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
) -> list[tuple[float, float]]:
    """
    Find the route that minimizes total elevation gain.

    Args:
        G: OSMnx graph with elevation_gain edge attributes
        origin_coords: (lat, lon)
        dest_coords:   (lat, lon)

    Returns:
        Ordered list of (lat, lon) pairs along the route.
    """
    orig_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
    dest_node = ox.distance.nearest_nodes(G, X=dest_coords[1], Y=dest_coords[0])

    path = nx.shortest_path(G, orig_node, dest_node, weight="elevation_gain")
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]


def get_shortest_route(
    G: nx.MultiDiGraph,
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
) -> list[tuple[float, float]]:
    """Find the shortest-distance route (standard routing)."""
    orig_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
    dest_node = ox.distance.nearest_nodes(G, X=dest_coords[1], Y=dest_coords[0])

    path = nx.shortest_path(G, orig_node, dest_node, weight="length")
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]


# ---------------------------------------------------------------------------
# 5. Stats helper
# ---------------------------------------------------------------------------

def compute_route_stats(
    G: nx.MultiDiGraph,
    coords: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    Given a coordinate list, return (distance_km, elevation_gain_m).
    Snaps each coord back to nearest node to look up edge attributes.
    """
    nodes = [
        ox.distance.nearest_nodes(G, X=lon, Y=lat)
        for lat, lon in coords
    ]

    total_distance = 0.0
    total_gain = 0.0

    for u, v in zip(nodes[:-1], nodes[1:]):
        # pick the edge with lowest length among parallel edges
        edge_data = min(G[u][v].values(), key=lambda d: d.get("length", 0))
        total_distance += edge_data.get("length", 0.0)
        total_gain += edge_data.get("elevation_gain", 0.0)

    return total_distance / 1000.0, total_gain


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    ORIGIN = (32.0853, 34.7818)       # Tel Aviv city center
    DESTINATION = (32.0679, 34.7751)  # Jaffa

    # Build graph
    G = load_graph()
    G = fetch_elevations(G)
    G = annotate_elevation_gain(G)

    # Flat route
    print("\nComputing flat route (min elevation gain)...")
    flat_coords = get_flat_route(G, ORIGIN, DESTINATION)
    flat_dist, flat_gain = compute_route_stats(G, flat_coords)

    # Shortest route
    print("Computing shortest route (min distance)...")
    short_coords = get_shortest_route(G, ORIGIN, DESTINATION)
    short_dist, short_gain = compute_route_stats(G, short_coords)

    # Output
    print("\n" + "=" * 50)
    print("FLAT ROUTE (minimizes elevation gain)")
    print("=" * 50)
    print(f"  Distance:       {flat_dist:.2f} km")
    print(f"  Elevation gain: {flat_gain:.1f} m")
    print(f"  Waypoints:      {len(flat_coords)}")
    print("  Coordinates:")
    for lat, lon in flat_coords:
        print(f"    ({lat:.6f}, {lon:.6f})")

    print("\n" + "=" * 50)
    print("SHORTEST ROUTE (minimizes distance)")
    print("=" * 50)
    print(f"  Distance:       {short_dist:.2f} km")
    print(f"  Elevation gain: {short_gain:.1f} m")
    print(f"  Waypoints:      {len(short_coords)}")

    print("\n" + "=" * 50)
    print("TRADEOFF SUMMARY")
    print("=" * 50)
    dist_overhead = flat_dist - short_dist
    gain_saved = short_gain - flat_gain
    print(f"  Flat route is {dist_overhead:+.2f} km longer")
    print(f"  Flat route saves {gain_saved:.1f} m of climbing")


if __name__ == "__main__":
    main()
