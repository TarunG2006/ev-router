# visualizer/dashboard.py

import folium
import osmnx as ox
import pandas as pd
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.battery_model import load_battery_model
from src.lpa_star import LPAStar, make_haversine_heuristic


def draw_delivery_map(G, path, rerouted_path=None, closed_edges=None,
                      output="visualizer/delivery_map.html"):
    m = folium.Map(
        location=[config.JAIPUR_CENTER_LAT, config.JAIPUR_CENTER_LON],
        zoom_start=13,
        tiles="CartoDB positron"
    )

    # Original planned route (blue)
    if path:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
        folium.PolyLine(coords, color='steelblue', weight=5, opacity=0.7,
                        tooltip="Original Route").add_to(m)
        folium.Marker(coords[0], popup="📦 START",
                      icon=folium.Icon(color='blue')).add_to(m)
        folium.Marker(coords[-1], popup="🏠 DELIVERY",
                      icon=folium.Icon(color='red')).add_to(m)

    # Rerouted path (green dashed)
    if rerouted_path:
        coords2 = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in rerouted_path]
        folium.PolyLine(coords2, color='green', weight=5, opacity=0.9,
                        dash_array='10',
                        tooltip="LPA* Rerouted Path").add_to(m)

    # Closed edges (red)
    if closed_edges:
        for u, v in closed_edges:
            uc = (G.nodes[u]['y'], G.nodes[u]['x'])
            vc = (G.nodes[v]['y'], G.nodes[v]['x'])
            folium.PolyLine([uc, vc], color='red', weight=8, opacity=1.0,
                            tooltip="🚫 Road Closed").add_to(m)

    m.save(output)
    print(f"Map saved: {output}")


def demo_visualization():
    print("Loading graph...")
    G = ox.load_graphml(config.PROCESSED_GRAPH_PATH)
    nodes = list(G.nodes())

    random.seed(42)
    start = random.choice(nodes)
    end   = random.choice(nodes)

    # Initial route
    h = make_haversine_heuristic(G, end)
    lpa = LPAStar(G, start, end, h)
    lpa.compute_shortest_path()
    original_path = lpa.extract_path()

    if not original_path or len(original_path) < 5:
        print("Path too short, picking new nodes")
        return

    # Pick an edge to close mid-route
    mid = len(original_path) // 2
    cu, cv = original_path[mid], original_path[mid+1]
    print(f"Closing edge: {cu} -> {cv}")

    lpa.close_road(cu, cv)
    lpa.compute_shortest_path()
    rerouted = lpa.extract_path()

    draw_delivery_map(G, original_path, rerouted_path=rerouted,
                      closed_edges=[(cu, cv)])
    print("Open visualizer/delivery_map.html in your browser to see the map.")


if __name__ == "__main__":
    demo_visualization()