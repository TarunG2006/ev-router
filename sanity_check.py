from src.benchmark import simulate_lpa_delivery, fix_graph_weights
import osmnx as ox
import random
import config

G = fix_graph_weights(ox.load_graphml(config.PROCESSED_GRAPH_PATH))
nodes = list(G.nodes())
random.seed(42)

for i in range(5):
    s, e = random.sample(nodes, 2)
    res = simulate_lpa_delivery(G, s, e, closure_step=3)
    print(f"Trial {i}: replans={res['replan_count']}, replan_ms={res.get('replan_ms', 0):.2f}ms, success={res['success']}")