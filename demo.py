#!/usr/bin/env python3
"""
Interactive CLI demo for EV Delivery Router
Shows LPA* rerouting with road closures on a live map
"""

import sys
import os
import random
import webbrowser
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import folium
import osmnx as ox
import networkx as nx
import config
from src.battery_model import load_battery_model, predict_edge_cost
from src.lpa_star import LPAStar, make_haversine_heuristic


def load_graph():
    """Load the processed Jaipur graph."""
    print("Loading Jaipur road network...")
    G = ox.load_graphml(config.PROCESSED_GRAPH_PATH)
    # Convert string attributes to float
    for u, v, k, d in G.edges(keys=True, data=True):
        d['battery_cost'] = float(d.get('battery_cost', 100))
        d['length'] = float(d.get('length', 100))
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def find_nearest_node(G, lat, lon):
    """Find the graph node nearest to given coordinates."""
    return ox.nearest_nodes(G, lon, lat)


def calculate_route_stats(G, path, model=None):
    """Calculate total distance and battery consumption for a path."""
    if not path or len(path) < 2:
        return 0, 0
    
    total_dist = 0
    total_battery = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)
            # Get first edge if multigraph
            if isinstance(edge_data, dict) and 0 in edge_data:
                edge_data = edge_data[0]
            total_dist += float(edge_data.get('length', 0))
            total_battery += float(edge_data.get('battery_cost', 0))
    
    return total_dist / 1000, total_battery  # km, Wh


def draw_interactive_map(G, original_path, rerouted_path=None, closed_edges=None,
                         start_name="Start", end_name="Destination"):
    """Draw an interactive map with routes and closures."""
    
    # Center map on route
    if original_path:
        lats = [G.nodes[n]['y'] for n in original_path]
        lons = [G.nodes[n]['x'] for n in original_path]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        center_lat = config.JAIPUR_CENTER_LAT
        center_lon = config.JAIPUR_CENTER_LON
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron"
    )
    
    # Add tile layer options
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)
    folium.LayerControl().add_to(m)
    
    # Original route (blue)
    if original_path and len(original_path) > 1:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in original_path]
        folium.PolyLine(
            coords, 
            color='#3388ff', 
            weight=6, 
            opacity=0.8,
            tooltip="Original Route (before closure)"
        ).add_to(m)
        
        # Start marker
        folium.Marker(
            coords[0], 
            popup=f"<b>START</b><br>{start_name}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # End marker
        folium.Marker(
            coords[-1], 
            popup=f"<b>DESTINATION</b><br>{end_name}",
            icon=folium.Icon(color='red', icon='flag')
        ).add_to(m)
    
    # Closed edges (red X markers)
    if closed_edges:
        for u, v in closed_edges:
            uc = (G.nodes[u]['y'], G.nodes[u]['x'])
            vc = (G.nodes[v]['y'], G.nodes[v]['x'])
            
            # Red line for closed road
            folium.PolyLine(
                [uc, vc], 
                color='red', 
                weight=10, 
                opacity=1.0,
                tooltip="ROAD CLOSED"
            ).add_to(m)
            
            # X marker at midpoint
            mid_lat = (uc[0] + vc[0]) / 2
            mid_lon = (uc[1] + vc[1]) / 2
            folium.Marker(
                [mid_lat, mid_lon],
                popup="<b>ROAD CLOSED</b>",
                icon=folium.Icon(color='red', icon='remove')
            ).add_to(m)
    
    # Rerouted path (orange dashed)
    if rerouted_path and len(rerouted_path) > 1:
        coords2 = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in rerouted_path]
        folium.PolyLine(
            coords2, 
            color='#ff7800', 
            weight=6, 
            opacity=0.9,
            dash_array='10, 10',
            tooltip="LPA* Rerouted Path (after closure)"
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 15px; border-radius: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3); font-family: Arial;">
        <b>EV Delivery Router Demo</b><br><br>
        <i style="background: #3388ff; width: 30px; height: 4px; display: inline-block;"></i> Original Route<br>
        <i style="background: #ff7800; width: 30px; height: 4px; display: inline-block; border-style: dashed;"></i> LPA* Rerouted<br>
        <i style="background: red; width: 30px; height: 4px; display: inline-block;"></i> Road Closed<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def get_landmark_coords():
    """Return dict of Jaipur landmarks with coordinates."""
    return {
        'hawa mahal': (26.9239, 75.8267),
        'city palace': (26.9258, 75.8237),
        'jaipur junction': (26.9196, 75.7879),
        'amber fort': (26.9855, 75.8513),
        'nahargarh fort': (26.9372, 75.8156),
        'jantar mantar': (26.9246, 75.8244),
        'albert hall': (26.9117, 75.8186),
        'birla mandir': (26.8923, 75.8149),
        'world trade park': (26.8948, 75.8093),
        'jaipur airport': (26.8242, 75.8122),
        'mansarovar': (26.8804, 75.7583),
        'malviya nagar': (26.8548, 75.8051),
        'vaishali nagar': (26.9124, 75.7373),
        'raja park': (26.9089, 75.7978),
        'c scheme': (26.9054, 75.7914),
    }


def find_location(G, query, landmarks):
    """Find a location by name or coordinates."""
    query = query.strip().lower()
    
    # Check landmarks
    if query in landmarks:
        lat, lon = landmarks[query]
        return find_nearest_node(G, lat, lon), query.title()
    
    # Check if coordinates (lat, lon)
    if ',' in query:
        try:
            parts = query.split(',')
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return find_nearest_node(G, lat, lon), f"({lat:.4f}, {lon:.4f})"
        except:
            pass
    
    # Random node
    if query in ['random', 'r', '']:
        node = random.choice(list(G.nodes()))
        lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
        return node, f"Random ({lat:.4f}, {lon:.4f})"
    
    return None, None


def interactive_demo():
    """Run interactive CLI demo."""
    print("=" * 60)
    print("   EV DELIVERY ROUTER - Interactive Demo")
    print("   LPA* Real-Time Rerouting with Road Closures")
    print("=" * 60)
    print()
    
    G = load_graph()
    model = load_battery_model()
    landmarks = get_landmark_coords()
    
    print("\nAvailable landmarks:")
    for name in sorted(landmarks.keys()):
        print(f"  - {name.title()}")
    print("\nOr enter coordinates as: lat, lon")
    print("Or press Enter for random location")
    print()
    
    # Get start location
    while True:
        start_input = input("Enter START location: ").strip()
        if not start_input:
            start_input = "random"
        start_node, start_name = find_location(G, start_input, landmarks)
        if start_node:
            print(f"  -> Start: {start_name}")
            break
        print("  Location not found. Try a landmark name or coordinates.")
    
    # Get end location
    while True:
        end_input = input("Enter DESTINATION: ").strip()
        if not end_input:
            end_input = "random"
        end_node, end_name = find_location(G, end_input, landmarks)
        if end_node:
            print(f"  -> Destination: {end_name}")
            break
        print("  Location not found. Try a landmark name or coordinates.")
    
    if start_node == end_node:
        print("Start and destination are the same. Please try again.")
        return
    
    # Calculate initial route
    print("\nCalculating initial route with LPA*...")
    h = make_haversine_heuristic(G, end_node)
    lpa = LPAStar(G, start_node, end_node, h)
    lpa.compute_shortest_path()
    original_path = lpa.extract_path()
    
    if not original_path:
        print("No route found between these locations.")
        return
    
    dist_km, battery_wh = calculate_route_stats(G, original_path, model)
    print(f"\nInitial Route:")
    print(f"  Distance:    {dist_km:.2f} km")
    print(f"  Battery:     {battery_wh:.0f} Wh ({battery_wh/config.BATTERY_CAPACITY_WH*100:.1f}%)")
    print(f"  Path nodes:  {len(original_path)}")
    
    # Ask about road closures
    closed_edges = []
    rerouted_path = None
    
    print("\n" + "-" * 40)
    closure_input = input("Simulate road closure? (y/n): ").strip().lower()
    
    if closure_input in ['y', 'yes']:
        if len(original_path) < 4:
            print("Route too short for mid-route closure demo.")
        else:
            # Close a road in the middle of the route
            mid = len(original_path) // 2
            cu, cv = original_path[mid], original_path[mid + 1]
            closed_edges.append((cu, cv))
            
            cu_lat, cu_lon = G.nodes[cu]['y'], G.nodes[cu]['x']
            cv_lat, cv_lon = G.nodes[cv]['y'], G.nodes[cv]['x']
            
            print(f"\nRoad closed at: ({cu_lat:.4f}, {cu_lon:.4f}) -> ({cv_lat:.4f}, {cv_lon:.4f})")
            print("Rerouting with LPA* (incremental update)...")
            
            # LPA* incremental reroute
            import time
            t0 = time.perf_counter()
            lpa.close_road(cu, cv)
            lpa.compute_shortest_path()
            rerouted_path = lpa.extract_path()
            replan_time = (time.perf_counter() - t0) * 1000
            
            if rerouted_path:
                new_dist, new_battery = calculate_route_stats(G, rerouted_path, model)
                print(f"\nRerouted Path:")
                print(f"  Distance:    {new_dist:.2f} km (+{new_dist - dist_km:.2f} km)")
                print(f"  Battery:     {new_battery:.0f} Wh ({new_battery/config.BATTERY_CAPACITY_WH*100:.1f}%)")
                print(f"  Replan time: {replan_time:.1f} ms")
                print(f"  Path nodes:  {len(rerouted_path)}")
            else:
                print("No alternate route found!")
    
    # Generate map
    print("\n" + "-" * 40)
    print("Generating interactive map...")
    
    m = draw_interactive_map(
        G, 
        original_path, 
        rerouted_path=rerouted_path,
        closed_edges=closed_edges,
        start_name=start_name,
        end_name=end_name
    )
    
    output_file = "visualizer/demo_route.html"
    m.save(output_file)
    print(f"Map saved: {output_file}")
    
    # Open in browser
    open_browser = input("\nOpen map in browser? (y/n): ").strip().lower()
    if open_browser in ['y', 'yes', '']:
        abs_path = os.path.abspath(output_file)
        webbrowser.open(f"file://{abs_path}")
        print("Map opened in browser!")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    interactive_demo()
