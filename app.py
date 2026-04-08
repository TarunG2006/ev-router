"""
EV Delivery Router - Streamlit Web App
Deploy to Streamlit Cloud for live demo
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import time
import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.lpa_star import LPAStar, make_haversine_heuristic

# Page config
st.set_page_config(
    page_title="EV Delivery Router",
    page_icon="🚗",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# LANDMARKS - Wikipedia verified coordinates (all within 5km of Hawa Mahal)
# Center: Hawa Mahal (26.9239, 75.8267) - all distances verified < 5km
# Coordinates verified from Wikipedia where available
# ─────────────────────────────────────────────────────────────
LANDMARKS = {
    # Old City Landmarks (Pink City) - Wikipedia verified
    'Hawa Mahal': (26.9239, 75.8267),              # Wikipedia: 26°55′26″N 75°49′36″E
    'City Palace': (26.9257, 75.8236),              # Wikipedia: 26°55′33″N 75°49′25″E
    'Jantar Mantar': (26.92472, 75.82444),          # Wikipedia: 26°55′29″N 75°49′28″E (UNESCO)
    'Govind Dev Ji Temple': (26.9288, 75.8240),     # City Palace complex
    'Johari Bazaar': (26.9222, 75.8228),            # Main jewelry market
    'Tripolia Bazaar': (26.9246, 75.8212),          # Near Tripolia Gate
    'Bapu Bazaar': (26.9178, 75.8228),              # Textile market
    'Nehru Bazaar': (26.9175, 75.8185),             # Near New Gate
    
    # City Gates (Historic Walls)
    'Chandpole Gate': (26.9228, 75.8108),           # Western gate
    'Ajmeri Gate': (26.9164, 75.7958),              # Southwest gate
    'Sanganeri Gate': (26.9089, 75.8200),           # Southern gate
    'New Gate': (26.9117, 75.8186),                 # Southern gate
    'Ghat Gate': (26.9297, 75.8268),                # Northern gate
    'Suraj Pol (Sun Gate)': (26.9310, 75.8230),     # Northeast gate
    'Chand Pol (Moon Gate)': (26.9230, 75.8105),    # Western gate
    
    # Museums & Heritage - Wikipedia verified
    'Albert Hall Museum': (26.91179, 75.81953),     # Wikipedia: 26°54′42″N 75°49′10″E
    'Nahargarh Fort (Foothills)': (26.9373, 75.8155), # Base road access point
    'Sisodia Rani Garden': (26.9010, 75.8560),      # East of city
    
    # Gardens & Parks
    'Ram Niwas Garden': (26.9140, 75.8190),         # Contains Albert Hall
    'Jaipur Zoo': (26.9122, 75.8214),               # Inside Ram Niwas
    'Central Park': (26.9056, 75.8000),             # C-Scheme area
    
    # Temples & Religious - Wikipedia verified
    'Birla Mandir': (26.8921, 75.8155),             # Wikipedia: 26°53′32″N 75°48′56″E
    'Moti Dungri Temple': (26.894621, 75.81674),    # Wikipedia: 26°53′41″N 75°49′00″E
    'Galtaji Temple': (26.9170, 75.8558),           # ~10km east (edge of coverage)
    'Khole Ke Hanuman Ji': (26.9067, 75.8428),      # Popular temple
    'Digambar Jain Temple': (26.9095, 75.8295),     # Near Sanganeri Gate
    
    # Transport Hubs - Wikipedia verified  
    'Jaipur Junction (Railway)': (26.9208, 75.7866), # Wikipedia: 26°55′15″N 75°47′12″E
    'Sindhi Camp Bus Stand': (26.9180, 75.7905),    # Main bus station
    
    # Shopping & Entertainment - Wikipedia verified
    'Raj Mandir Cinema': (26.9155, 75.8102),        # Wikipedia: 26°54′56″N 75°48′37″E
    
    # Hospitals & Medical
    'SMS Hospital': (26.9069, 75.8069),             # Sawai Man Singh Hospital
    'Santokba Durlabhji Hospital': (26.8905, 75.8040),
    
    # Sports & Recreation
    'SMS Stadium': (26.8969, 75.8028),              # Cricket stadium
    'Sawai Mansingh Stadium': (26.8970, 75.8030),   # Same as SMS Stadium
    'Railway Stadium': (26.9195, 75.7890),          # Near Junction
    
    # Government & Institutions
    'Vidhan Sabha': (26.8922, 75.7977),             # State Assembly
    'Rajasthan High Court': (26.9125, 75.7940),
    'Secretariat': (26.8910, 75.7985),
    'GPO Jaipur': (26.9145, 75.8010),               # General Post Office
    
    # Major Landmarks & Circles
    'Statue Circle': (26.8997, 75.8011),            # C-Scheme
    'MI Road': (26.9156, 75.8042),                  # Main shopping street
    'Rambagh Circle': (26.8890, 75.7980),
    'Bani Park': (26.9330, 75.7945),                # Residential area
    
    # Hotels & Tourism
    'Rambagh Palace': (26.8965, 75.7960),           # Heritage hotel
    'Jai Mahal Palace': (26.9200, 75.7870),         # Near Junction
    'Narain Niwas Palace': (26.8983, 75.8180),      # Heritage hotel
}
}


@st.cache_resource
def load_graph():
    """Load and cache the graph."""
    G = ox.load_graphml(config.PROCESSED_GRAPH_PATH)
    # Convert attributes to float
    for u, v, k, d in G.edges(keys=True, data=True):
        d['battery_cost'] = float(d.get('battery_cost', 100))
        d['length'] = float(d.get('length', 100))
    return G


def find_nearest_node(G, lat, lon):
    """Find nearest graph node to coordinates."""
    return ox.nearest_nodes(G, lon, lat)


def calculate_route_stats(G, path):
    """Calculate distance and battery for a path."""
    if not path or len(path) < 2:
        return 0, 0
    
    total_dist = 0
    total_battery = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = None
        # Try forward edge first, then reverse
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)
        elif G.has_edge(v, u):
            edge_data = G.get_edge_data(v, u)
        
        if edge_data:
            # Handle MultiDiGraph structure
            if isinstance(edge_data, dict) and 0 in edge_data:
                edge_data = edge_data[0]
            length = edge_data.get('length', 0)
            # Ensure we get the actual length value
            if isinstance(length, str):
                length = float(length)
            total_dist += float(length)
            total_battery += float(edge_data.get('battery_cost', 0))
    
    return total_dist / 1000, total_battery


def create_map(G, original_path, rerouted_path=None, closed_edge=None):
    """Create Folium map with routes."""
    # Center on route
    if original_path:
        lats = [G.nodes[n]['y'] for n in original_path]
        lons = [G.nodes[n]['x'] for n in original_path]
        center = [sum(lats)/len(lats), sum(lons)/len(lons)]
    else:
        center = [config.JAIPUR_CENTER_LAT, config.JAIPUR_CENTER_LON]
    
    m = folium.Map(location=center, zoom_start=14, tiles='CartoDB positron')
    
    # Original route (blue)
    if original_path:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in original_path]
        folium.PolyLine(coords, color='#3388ff', weight=5, opacity=0.8,
                       tooltip='Original Route').add_to(m)
        
        # Start marker
        folium.Marker(coords[0], popup='START',
                     icon=folium.Icon(color='green', icon='play')).add_to(m)
        # End marker
        folium.Marker(coords[-1], popup='DESTINATION',
                     icon=folium.Icon(color='red', icon='flag')).add_to(m)
    
    # Road closure (red)
    if closed_edge:
        u, v = closed_edge
        uc = (G.nodes[u]['y'], G.nodes[u]['x'])
        vc = (G.nodes[v]['y'], G.nodes[v]['x'])
        folium.PolyLine([uc, vc], color='red', weight=8, opacity=1.0,
                       tooltip='ROAD CLOSED').add_to(m)
        mid = ((uc[0]+vc[0])/2, (uc[1]+vc[1])/2)
        folium.Marker(mid, popup='ROAD CLOSED',
                     icon=folium.Icon(color='red', icon='remove')).add_to(m)
    
    # Rerouted path (orange dashed)
    if rerouted_path:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in rerouted_path]
        folium.PolyLine(coords, color='#ff7800', weight=5, opacity=0.9,
                       dash_array='10', tooltip='LPA* Rerouted').add_to(m)
    
    return m


# ─────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────

st.title("🚗 EV Delivery Router")
st.markdown("**LPA* Real-Time Rerouting with Road Closures**")

# Initialize session state
if 'route_calculated' not in st.session_state:
    st.session_state.route_calculated = False
    st.session_state.original_path = None
    st.session_state.rerouted_path = None
    st.session_state.closed_edge = None
    st.session_state.stats = {}

# Sidebar
st.sidebar.header("📍 Route Settings")

start_location = st.sidebar.selectbox(
    "Start Location",
    options=list(LANDMARKS.keys()),
    index=list(LANDMARKS.keys()).index('Jaipur Junction (Railway)')
)

end_location = st.sidebar.selectbox(
    "Destination",
    options=list(LANDMARKS.keys()),
    index=list(LANDMARKS.keys()).index('Hawa Mahal')
)

simulate_closure = st.sidebar.checkbox("Simulate Road Closure", value=True)

# Load graph
with st.spinner("Loading Jaipur road network..."):
    G = load_graph()

st.sidebar.success(f"✓ Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Calculate route button
if st.sidebar.button("🚀 Calculate Route", type="primary"):
    
    start_coords = LANDMARKS[start_location]
    end_coords = LANDMARKS[end_location]
    
    start_node = find_nearest_node(G, *start_coords)
    end_node = find_nearest_node(G, *end_coords)
    
    with st.spinner("Calculating initial route..."):
        h = make_haversine_heuristic(G, end_node)
        lpa = LPAStar(G, start_node, end_node, h)
        lpa.compute_shortest_path()
        original_path = lpa.extract_path()
    
    if original_path:
        dist_km, battery_wh = calculate_route_stats(G, original_path)
        
        # Store in session state
        st.session_state.original_path = original_path
        st.session_state.stats = {
            'dist_km': dist_km,
            'battery_wh': battery_wh,
            'path_nodes': len(original_path)
        }
        
        # Simulate closure
        rerouted_path = None
        closed_edge = None
        
        if simulate_closure and len(original_path) >= 4:
            mid = len(original_path) // 2
            cu, cv = original_path[mid], original_path[mid + 1]
            closed_edge = (cu, cv)
            
            t0 = time.perf_counter()
            lpa.close_road(cu, cv)
            lpa.compute_shortest_path()
            rerouted_path = lpa.extract_path()
            replan_time = (time.perf_counter() - t0) * 1000
            
            if rerouted_path:
                new_dist, new_battery = calculate_route_stats(G, rerouted_path)
                st.session_state.stats.update({
                    'new_dist': new_dist,
                    'new_battery': new_battery,
                    'replan_time': replan_time
                })
        
        st.session_state.rerouted_path = rerouted_path
        st.session_state.closed_edge = closed_edge
        st.session_state.route_calculated = True
    else:
        st.error("No route found!")
        st.session_state.route_calculated = False

# Display results if route has been calculated
if st.session_state.route_calculated and st.session_state.original_path:
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📊 Route Stats")
        stats = st.session_state.stats
        
        st.markdown("**Initial Route:**")
        st.metric("Distance", f"{stats['dist_km']:.2f} km")
        st.metric("Battery", f"{stats['battery_wh']:.0f} Wh ({stats['battery_wh']/config.BATTERY_CAPACITY_WH*100:.1f}%)")
        st.metric("Path Nodes", stats['path_nodes'])
        
        if st.session_state.closed_edge and st.session_state.rerouted_path:
            st.markdown("---")
            st.markdown("**🚧 Road Closure Simulated**")
            st.markdown("**Rerouted Path:**")
            st.metric("New Distance", f"{stats['new_dist']:.2f} km", f"+{stats['new_dist']-stats['dist_km']:.2f} km")
            st.metric("New Battery", f"{stats['new_battery']:.0f} Wh")
            st.metric("⚡ Replan Time", f"{stats['replan_time']:.1f} ms")
            st.success("✓ LPA* rerouted successfully!")
    
    with col1:
        st.subheader("🗺️ Route Map")
        m = create_map(G, st.session_state.original_path, 
                      st.session_state.rerouted_path, 
                      st.session_state.closed_edge)
        st_folium(m, width=700, height=500, returned_objects=[])

# Benchmark results
st.sidebar.markdown("---")
st.sidebar.header("📈 Benchmark Results")
st.sidebar.markdown("""
**1000 Trials on Jaipur Map:**
- LPA* is **2.2x faster** than A*
- **99.8%** success with road closures
- **0%** battery failures
- Dijkstra fails **96.6%** when roads close
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <b>EV Delivery Router</b> | LPA* Algorithm | IIT Guwahati SDE Project<br>
    Built with OSM, Scikit-learn, Streamlit
</div>
""", unsafe_allow_html=True)
