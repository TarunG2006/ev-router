# src/graph_builder.py
import networkx as nx
import osmnx as ox
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────
# STEP 1: Download OSM road network
# ─────────────────────────────────────────

def download_jaipur_graph():
    import networkx as nx
    ox.settings.max_query_area_size = 25_000_000_000_000

    print("Downloading Jaipur road network (8km radius from city center)...")
    print(f"Center: {config.JAIPUR_CENTER_LAT}, {config.JAIPUR_CENTER_LON}")

    G = ox.graph_from_point(
        (config.JAIPUR_CENTER_LAT, config.JAIPUR_CENTER_LON),
        dist=5000,
        network_type="drive",
        simplify=True
    )
    # G is already a MultiDiGraph — do NOT convert, just filter SCC directly
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()

    print(f"Downloaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    ox.save_graphml(G, config.RAW_GRAPH_PATH)
    print(f"Saved to {config.RAW_GRAPH_PATH}")
    return G

# ─────────────────────────────────────────
# STEP 2: Add elevation to every node
# ─────────────────────────────────────────

def add_elevation_via_srtm(G):
    """
    Use the 'srtm' python package to add elevation (meters) to each node.
    Reliable, offline after first download, no API key needed.
    """
    import srtm
    elevation_data = srtm.get_data()

    print(f"Computing elevation for {G.number_of_nodes()} nodes via SRTM...")
    for node_id, data in tqdm(G.nodes(data=True)):
        lat = data['y']
        lon = data['x']
        elev = elevation_data.get_elevation(lat, lon)
        data['elevation'] = float(elev) if elev is not None else 0.0

    return G


# ─────────────────────────────────────────
# STEP 3: Compute slope for each edge
# ─────────────────────────────────────────

def compute_slope_degrees(elev_start, elev_end, distance_m):
    """Slope in degrees. Positive = uphill, negative = downhill."""
    if distance_m < 0.5:
        return 0.0
    delta_h = elev_end - elev_start
    slope_rad = np.arctan2(delta_h, distance_m)
    return float(np.degrees(slope_rad))


def get_road_type_code(highway_val):
    """Convert OSM 'highway' tag string to numeric 0/1/2."""
    if isinstance(highway_val, list):
        highway_val = highway_val[0]
    return config.ROAD_TYPE_MAP.get(str(highway_val), 0)


def add_edge_attributes(G, battery_model=None):
    print(f"Computing edge attributes for {G.number_of_edges()} edges...")

    edges = list(G.edges(keys=True, data=True))

    # ── Pass 1: collect feature vectors ──
    rows = []
    for u, v, key, data in edges:
        elev_u  = float(G.nodes[u].get('elevation', 0.0))
        elev_v  = float(G.nodes[v].get('elevation', 0.0))
        dist_m  = float(data.get('length', 100.0))
        dist_km = dist_m / 1000.0
        slope   = compute_slope_degrees(elev_u, elev_v, dist_m)

        highway   = data.get('highway', 'residential')
        road_type = get_road_type_code(highway)

        speed = data.get('speed_kph', 30.0)
        if isinstance(speed, list):
            speed = speed[0]
        speed = float(speed) if speed else 30.0

        rows.append((u, v, key, slope, dist_km, speed, road_type))

    # ── Pass 2: Calculate battery cost ──
    # Use physics-based formula calibrated to ~200 Wh/km for EV car
    # (Kaggle model doesn't extrapolate well to short edges)
    print("Computing battery cost using physics formula (calibrated to Kaggle data)...")
    
    # Base consumption: ~150-200 Wh/km for EV car
    # Adjust for slope, speed, road type
    costs = []
    for r in rows:
        slope, dist_km, speed, road_type = r[3], r[4], r[5], r[6]
        
        # Base: 180 Wh/km
        # Slope: +25 Wh/km per degree uphill, -15 Wh/km per degree downhill
        # Speed: quadratic drag above 50 km/h
        # Road type: highway=0.9x, city=0.95x, residential=1.0x
        
        base_per_km = 180.0
        slope_factor = slope * 25.0 if slope > 0 else slope * 15.0
        speed_factor = max(0, (speed - 50) ** 2 * 0.05) if speed > 50 else 0
        road_factor = {0: 1.0, 1: 0.95, 2: 0.90}.get(road_type, 1.0)
        
        wh_per_km = (base_per_km + slope_factor + speed_factor) * road_factor
        wh_per_km = max(50.0, wh_per_km)  # Minimum 50 Wh/km
        
        cost = wh_per_km * dist_km
        costs.append(max(1.0, cost))  # Minimum 1 Wh per edge

    # ── Pass 3: write back to graph ──
    print("Writing attributes back to graph...")
    for i, (u, v, key, slope, dist_km, speed, road_type) in enumerate(tqdm(rows)):
        data = G[u][v][key]
        data['slope']        = round(slope, 4)
        data['road_type']    = road_type
        data['speed_kph']    = round(speed, 2)
        data['distance_km']  = round(dist_km, 4)
        data['battery_cost'] = round(float(costs[i]), 4)
        data['weight']       = data['battery_cost']

    return G


# ─────────────────────────────────────────
# STEP 4: Full pipeline
# ─────────────────────────────────────────

def build_full_graph(battery_model=None):
    """
    End-to-end pipeline:
      download (bbox) → elevation (SRTM) → edge attrs → save
    """
    import os

    # If raw graph already exists, skip download
    if os.path.exists(config.RAW_GRAPH_PATH):
        print(f"Raw graph found at {config.RAW_GRAPH_PATH} — skipping download.")
        print("(Delete data/raw/jaipur_graph.graphml to force re-download)")
        G = ox.load_graphml(config.RAW_GRAPH_PATH)
        print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # If the loaded graph is the old oversized one, re-download
        if G.number_of_nodes() > 20_000:
            print(f"WARNING: Graph has {G.number_of_nodes()} nodes — looks oversized.")
            print("Re-downloading with bbox to get Jaipur city only...")
            os.remove(config.RAW_GRAPH_PATH)
            G = download_jaipur_graph()
    else:
        G = download_jaipur_graph()

    G = add_elevation_via_srtm(G)
    G = add_edge_attributes(G, battery_model=battery_model)

    ox.save_graphml(G, config.PROCESSED_GRAPH_PATH)
    print(f"\nProcessed graph saved to {config.PROCESSED_GRAPH_PATH}")
    print(f"Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


if __name__ == "__main__":
    # Run full pipeline with battery model
    from src.battery_model import load_battery_model
    model = load_battery_model()
    build_full_graph(battery_model=model)