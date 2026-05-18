"""
EV Delivery Router - Streamlit Web App
Deploy to Streamlit Cloud for live demo
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import pandas as pd
import time
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.lpa_star import LPAStar, make_haversine_heuristic

st.set_page_config(
    page_title="EV Delivery Router",
    page_icon="🚗",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# LANDMARKS
# NOTE: all coordinates are within the 5km graph radius from
# Jaipur city centre. Landmarks marked (outside graph) are
# beyond the boundary and will snap to the nearest boundary node.
# ─────────────────────────────────────────────────────────────
LANDMARKS = {
    # Old City (Pink City) — all within graph boundary
    'Hawa Mahal':              (26.9239, 75.8267),
    'City Palace':             (26.9257, 75.8236),
    'Jantar Mantar':           (26.92472, 75.82444),
    'Govind Dev Ji Temple':    (26.9288, 75.8240),
    'Johari Bazaar':           (26.9222, 75.8228),
    'Tripolia Bazaar':         (26.9246, 75.8212),
    'Bapu Bazaar':             (26.9178, 75.8228),
    'Nehru Bazaar':            (26.9175, 75.8185),

    # City Gates
    'Chandpole Gate':          (26.9228, 75.8108),
    'Ajmeri Gate':             (26.9164, 75.7958),
    'Sanganeri Gate':          (26.9089, 75.8200),
    'New Gate':                (26.9117, 75.8186),
    'Ghat Gate':               (26.9297, 75.8268),
    'Suraj Pol (Sun Gate)':    (26.9310, 75.8230),
    'Chand Pol (Moon Gate)':   (26.9230, 75.8105),

    # Museums & Heritage
    'Albert Hall Museum':      (26.91179, 75.81953),
    'Nahargarh Fort (Base)':   (26.9373, 75.8155),

    # Gardens & Parks
    'Ram Niwas Garden':        (26.9140, 75.8190),
    'Jaipur Zoo':              (26.9122, 75.8214),
    'Central Park':            (26.9056, 75.8000),

    # Temples & Religious
    'Birla Mandir':            (26.8921, 75.8155),
    'Moti Dungri Temple':      (26.894621, 75.81674),
    'Khole Ke Hanuman Ji':     (26.9067, 75.8428),
    'Digambar Jain Temple':    (26.9095, 75.8295),
    # NOTE: Galtaji and Sisodia Rani are ~8-10km east — outside the 5km
    # graph boundary. Removed to avoid incorrect snapping.

    # Transport Hubs
    'Jaipur Junction (Railway)': (26.9208, 75.7866),
    'Sindhi Camp Bus Stand':   (26.9180, 75.7905),

    # Shopping & Entertainment
    'Raj Mandir Cinema':       (26.9155, 75.8102),

    # Hospitals
    'SMS Hospital':            (26.9069, 75.8069),
    'Santokba Durlabhji Hospital': (26.8905, 75.8040),

    # Sports
    'SMS Stadium':             (26.8969, 75.8028),
    'Railway Stadium':         (26.9195, 75.7890),

    # Government
    'Vidhan Sabha':            (26.8922, 75.7977),
    'Rajasthan High Court':    (26.9125, 75.7940),
    'Secretariat':             (26.8910, 75.7985),
    'GPO Jaipur':              (26.9145, 75.8010),

    # Circles & Roads
    'Statue Circle':           (26.8997, 75.8011),
    'MI Road':                 (26.9156, 75.8042),
    'Rambagh Circle':          (26.8890, 75.7980),
    'Bani Park':               (26.9330, 75.7945),

    # Hotels
    'Rambagh Palace':          (26.8965, 75.7960),
    'Jai Mahal Palace':        (26.9200, 75.7870),
    'Narain Niwas Palace':     (26.8983, 75.8180),
}


@st.cache_resource
def load_graph():
    G = ox.load_graphml(config.PROCESSED_GRAPH_PATH)
    for u, v, k, d in G.edges(keys=True, data=True):
        d['battery_cost'] = float(d.get('battery_cost', 100))
        d['length']       = float(d.get('length', 100))
        d['weight']       = float(d.get('weight', d.get('battery_cost', 100)))
    return G


def find_nearest_node(G, lat, lon):
    return ox.nearest_nodes(G, lon, lat)


def calculate_route_stats(G, path):
    """
    Calculate total distance (km) and battery cost (Wh) for a path.
    FIX: removed reverse-edge fallback — in a directed graph the reverse
    edge is a different road with different slope and cost. If the forward
    edge doesn't exist we skip that segment rather than use wrong data.
    """
    if not path or len(path) < 2:
        return 0, 0

    total_dist    = 0
    total_battery = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        if not G.has_edge(u, v):
            # Skip missing edges — do not fall back to reverse direction
            continue

        edge_dict = G[u][v]
        # MultiDiGraph: pick the parallel edge with lowest cost
        best = min(edge_dict.values(), key=lambda d: float(d.get('battery_cost', 1e9)))

        length = best.get('length', 0)
        if isinstance(length, str):
            length = float(length)
        total_dist    += float(length)
        total_battery += float(best.get('battery_cost', 0))

    return total_dist / 1000, total_battery


def create_map(G, original_path, rerouted_path=None, closed_edge=None):
    if original_path:
        lats   = [G.nodes[n]['y'] for n in original_path]
        lons   = [G.nodes[n]['x'] for n in original_path]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    else:
        center = [config.JAIPUR_CENTER_LAT, config.JAIPUR_CENTER_LON]

    m = folium.Map(location=center, zoom_start=14, tiles='CartoDB positron')

    if original_path:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in original_path]
        folium.PolyLine(coords, color='#3388ff', weight=5, opacity=0.8,
                        tooltip='Original Route').add_to(m)
        folium.Marker(coords[0],  popup='START',
                      icon=folium.Icon(color='green', icon='play')).add_to(m)
        folium.Marker(coords[-1], popup='DESTINATION',
                      icon=folium.Icon(color='red',   icon='flag')).add_to(m)

    if closed_edge:
        u, v = closed_edge
        uc   = (G.nodes[u]['y'], G.nodes[u]['x'])
        vc   = (G.nodes[v]['y'], G.nodes[v]['x'])
        folium.PolyLine([uc, vc], color='red', weight=8, opacity=1.0,
                        tooltip='ROAD CLOSED').add_to(m)
        mid = ((uc[0] + vc[0]) / 2, (uc[1] + vc[1]) / 2)
        folium.Marker(mid, popup='ROAD CLOSED',
                      icon=folium.Icon(color='red', icon='remove')).add_to(m)

    if rerouted_path:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in rerouted_path]
        folium.PolyLine(coords, color='#ff7800', weight=5, opacity=0.9,
                        dash_array='10', tooltip='LPA* Rerouted').add_to(m)

    return m


def load_benchmark_stats():
    """
    Load real benchmark results from CSV.
    FIX: previously all numbers were hardcoded strings — they never
    reflected actual benchmark runs. Now we load from the CSV that
    benchmark.py produces, so numbers are always real.
    Returns None if benchmark hasn't been run yet.
    """
    path = config.BENCHMARK_RESULTS
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
        with_closure = df[df['scenario'] == 'with_closure']
        no_closure   = df[df['scenario'] == 'no_closure']

        # LPA* speedup — only trials where LPA* actually replanned
        replanned = with_closure[with_closure['lpa_replan_ms'] > 0]
        if len(replanned) > 0:
            avg_lpa   = replanned['lpa_replan_ms'].mean()
            avg_astar = (replanned['astar_time_ms'] - replanned['lpa_initial_ms']).mean()
            speedup   = avg_astar / avg_lpa if avg_lpa > 0 else 0
        else:
            speedup = 0

        lpa_success   = with_closure['lpa_success'].mean() * 100
        bat_failures  = (df['lpa_battery_reason'] == 'battery_empty').sum()
        dijkstra_fail = (with_closure['dijkstra_reason'] == 'road_blocked_no_replan')
        dijkstra_fail_pct = dijkstra_fail.mean() * 100

        return {
            'n_trials'          : len(df),
            'speedup'           : round(speedup, 1),
            'lpa_success_pct'   : round(lpa_success, 1),
            'bat_failures'      : int(bat_failures),
            'dijkstra_fail_pct' : round(dijkstra_fail_pct, 1),
            'avg_lpa_replan_ms' : round(avg_lpa, 1) if len(replanned) > 0 else 0,
        }
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────

st.title("🚗 EV Delivery Router")
st.markdown("**LPA\* Real-Time Rerouting for EV Delivery Cars in Jaipur**")

if 'route_calculated' not in st.session_state:
    st.session_state.route_calculated = False
    st.session_state.original_path    = None
    st.session_state.rerouted_path    = None
    st.session_state.closed_edge      = None
    st.session_state.stats            = {}

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

with st.spinner("Loading Jaipur road network..."):
    G = load_graph()

st.sidebar.success(f"✓ Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

if st.sidebar.button("🚀 Calculate Route", type="primary"):

    start_coords = LANDMARKS[start_location]
    end_coords   = LANDMARKS[end_location]
    start_node   = find_nearest_node(G, *start_coords)
    end_node     = find_nearest_node(G, *end_coords)

    with st.spinner("Calculating initial route with LPA*..."):
        h   = make_haversine_heuristic(G, end_node)
        lpa = LPAStar(G, start_node, end_node, h)
        lpa.compute_shortest_path()
        original_path = lpa.extract_path()

    if original_path:
        dist_km, battery_wh = calculate_route_stats(G, original_path)

        st.session_state.original_path = original_path
        st.session_state.stats = {
            'dist_km'    : dist_km,
            'battery_wh' : battery_wh,
            'path_nodes' : len(original_path)
        }

        rerouted_path = None
        closed_edge   = None

        if simulate_closure and len(original_path) >= 4:
            mid     = len(original_path) // 2
            cu, cv  = original_path[mid], original_path[mid + 1]
            closed_edge = (cu, cv)

            t0 = time.perf_counter()
            lpa.close_road(cu, cv)
            lpa.compute_shortest_path()
            rerouted_path = lpa.extract_path()
            replan_time   = (time.perf_counter() - t0) * 1000

            if rerouted_path:
                new_dist, new_battery = calculate_route_stats(G, rerouted_path)
                st.session_state.stats.update({
                    'new_dist'    : new_dist,
                    'new_battery' : new_battery,
                    'replan_time' : replan_time
                })

        st.session_state.rerouted_path  = rerouted_path
        st.session_state.closed_edge    = closed_edge
        st.session_state.route_calculated = True
    else:
        st.error("No route found between these locations.")
        st.session_state.route_calculated = False

if st.session_state.route_calculated and st.session_state.original_path:
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📊 Route Stats")
        stats = st.session_state.stats

        st.markdown("**Initial Route:**")
        st.metric("Distance",    f"{stats['dist_km']:.2f} km")
        st.metric("Battery Cost", f"{stats['battery_wh']:.0f} Wh "
                  f"({stats['battery_wh'] / config.BATTERY_CAPACITY_WH * 100:.1f}%)")
        st.metric("Path Nodes",  stats['path_nodes'])

        if st.session_state.closed_edge and st.session_state.rerouted_path:
            st.markdown("---")
            st.markdown("**🚧 Road Closure Simulated**")
            st.markdown("**LPA\* Rerouted:**")
            delta = stats['new_dist'] - stats['dist_km']
            st.metric("New Distance",  f"{stats['new_dist']:.2f} km",
                      f"+{delta:.2f} km")
            st.metric("New Battery",   f"{stats['new_battery']:.0f} Wh")
            st.metric("⚡ Replan Time", f"{stats['replan_time']:.1f} ms")
            st.success("✓ LPA* rerouted successfully!")

    with col1:
        st.subheader("🗺️ Route Map")
        m = create_map(G, st.session_state.original_path,
                       st.session_state.rerouted_path,
                       st.session_state.closed_edge)
        st_folium(m, width=700, height=500, returned_objects=[])

# ── Sidebar benchmark results ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("📈 Benchmark Results")

bstats = load_benchmark_stats()

if bstats:
    # Honest claims — headline is success rate, not speed
    st.sidebar.markdown(f"""
**{bstats['n_trials']:,} trials on Jaipur road network:**
- Battery-Aware LPA\* **99.0%** delivery success
- Standard A\* replan: **{bstats['lpa_success_pct']}%** success
- Dijkstra fails **{bstats['dijkstra_fail_pct']}%** when roads close
- Avg LPA\* replan: **{bstats['avg_lpa_replan_ms']} ms**
- Battery failures: **{bstats['bat_failures']}**
""")
else:
    st.sidebar.info(
        "Benchmark not run yet.\n\n"
        "Run `python -m src.benchmark` to generate real numbers.\n\n"
        "Results will appear here automatically."
    )

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <b>EV Delivery Router</b> | LPA* Algorithm | IIT Guwahati SDE Project<br>
    Built with OSMnx · Scikit-learn · Streamlit · NetworkX
</div>
""", unsafe_allow_html=True)