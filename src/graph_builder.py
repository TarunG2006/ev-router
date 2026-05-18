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

    print("Downloading Jaipur road network (5km radius from city center)...")
    print(f"Center: {config.JAIPUR_CENTER_LAT}, {config.JAIPUR_CENTER_LON}")

    G = ox.graph_from_point(
        (config.JAIPUR_CENTER_LAT, config.JAIPUR_CENTER_LON),
        dist=5000,
        network_type="drive",
        simplify=True
    )
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


def _physics_cost(slope, dist_km, speed, road_type):
    """
    Fallback physics formula when no ML model is available.
    Calibrated to ~180 Wh/km base for an EV car.
    """
    base_per_km  = 180.0
    slope_factor = slope * 25.0 if slope > 0 else slope * 15.0
    speed_factor = max(0, (speed - 50) ** 2 * 0.05) if speed > 50 else 0
    road_factor  = {0: 1.0, 1: 0.95, 2: 0.90}.get(road_type, 1.0)
    wh_per_km    = (base_per_km + slope_factor + speed_factor) * road_factor
    wh_per_km    = max(50.0, wh_per_km)
    return max(1.0, wh_per_km * dist_km)


def add_edge_attributes(G, battery_model=None):
    """
    Compute and attach edge attributes:
      slope, road_type, speed_kph, distance_km, battery_cost, weight

    If battery_model is provided, battery_cost is predicted by the
    trained RandomForest (predict_edge_cost). Otherwise falls back
    to the calibrated physics formula.
    """
    # Import here to avoid circular import when battery_model=None
    if battery_model is not None:
        from src.battery_model import predict_edge_cost
        print("Computing battery cost using trained RandomForest model...")
    else:
        print("No model provided — using physics formula fallback...")

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

    # ── Pass 2: calculate battery cost ──
    # FIX: use the trained RF model when available.
    # Previously battery_model was passed in but never called — the function
    # always used the hardcoded physics formula. Now RF is the primary path.
    if battery_model is not None:
        features = np.array([[r[3], r[5], r[6]] for r in rows])  # slope, speed, road_type
        wh_per_km_arr = battery_model.predict(features)
        costs = [max(1.0, float(wh_per_km_arr[i]) * rows[i][4]) for i in range(len(rows))]
    else:
        costs = [_physics_cost(r[3], r[4], r[5], r[6]) for r in rows]

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

    source = "RandomForest" if battery_model is not None else "physics formula"
    print(f"Edge costs computed using: {source}")
    return G


# ─────────────────────────────────────────
# STEP 4: Full pipeline
# ─────────────────────────────────────────

def build_full_graph(battery_model=None):
    """
    End-to-end pipeline:
      download → elevation (SRTM) → edge attrs (RF or physics) → save

    Pass battery_model to use the trained RandomForest for edge costs.
    Omit to use the physics fallback.
    """
    if os.path.exists(config.RAW_GRAPH_PATH):
        print(f"Raw graph found at {config.RAW_GRAPH_PATH} — skipping download.")
        G = ox.load_graphml(config.RAW_GRAPH_PATH)
        print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_nodes() > 20_000:
            print(f"WARNING: Graph has {G.number_of_nodes()} nodes — looks oversized.")
            print("Re-downloading with 5km radius...")
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
    from src.battery_model import load_battery_model
    model = load_battery_model()
    build_full_graph(battery_model=model)