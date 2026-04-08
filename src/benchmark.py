# src/benchmark.py
"""
Benchmark comparing Dijkstra, Full A*, LPA*, and Battery-Aware LPA* replanning.
FIXED VERSION: Uses while loops so step=0 reset actually works after replan.
Measures: replan speed, battery failures, delivery success rate.
"""

import time
import random
import math
import copy
import networkx as nx
import pandas as pd
from tqdm import tqdm
import osmnx as ox
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lpa_star import LPAStar, make_haversine_heuristic
from src.lpa_star_battery import LPAStarBattery, make_battery_heuristic
from src.battery_model import load_battery_model
import config


def fix_graph_weights(G):
    """Convert all edge attributes from strings to correct types after GraphML load."""
    for u, v, k, data in G.edges(keys=True, data=True):
        for attr in ['weight', 'battery_cost', 'length', 'slope', 'speed_kph']:
            if attr in data:
                try:
                    data[attr] = float(data[attr])
                except (ValueError, TypeError):
                    data[attr] = 15.0
        if 'road_type' in data:
            try:
                data['road_type'] = int(float(data['road_type']))
            except (ValueError, TypeError):
                data['road_type'] = 0
    return G


def simulate_delivery_with_dynamic_closure(G, start, end, method, closure_step=None):
    if method == 'dijkstra':
        return simulate_dijkstra_delivery(G, start, end, closure_step)
    elif method == 'astar_replan':
        return simulate_astar_delivery(G, start, end, closure_step)
    elif method == 'lpa_incremental':
        return simulate_lpa_delivery(G, start, end, closure_step)
    else:
        raise ValueError(f"Unknown method: {method}")


def simulate_dijkstra_delivery(G, start, end, closure_step):
    """Static Dijkstra - cannot handle dynamic closures."""
    t0 = time.perf_counter()

    try:
        path = nx.dijkstra_path(G, start, end, weight='weight')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {
            "success": False, "reason": "no_initial_path",
            "time_ms": (time.perf_counter() - t0) * 1000,
            "battery_left": 0, "replan_count": 0, "expanded": 0
        }

    plan_time = (time.perf_counter() - t0) * 1000
    battery = config.BATTERY_CAPACITY_WH

    # FIX: while loop so closure_step triggers correctly
    step = 0
    while step < len(path) - 1:
        u, v = path[step], path[step + 1]

        if closure_step is not None and step == closure_step:
            # Dijkstra cannot replan
            return {
                "success": False, "reason": "road_blocked_no_replan",
                "time_ms": plan_time, "battery_left": battery,
                "replan_count": 0, "expanded": 0
            }

        if G.has_edge(u, v):
            edge_data = G[u][v][min(G[u][v].keys())]
            cost = float(edge_data.get('battery_cost', 15.0))
            battery -= cost

            if battery < config.BATTERY_LOW_WH:
                return {
                    "success": False, "reason": "battery_empty",
                    "time_ms": plan_time, "battery_left": battery,
                    "replan_count": 0, "expanded": 0
                }

        step += 1

    return {
        "success": True, "reason": "delivered",
        "time_ms": plan_time, "battery_left": battery,
        "replan_count": 0, "expanded": 0
    }


def simulate_astar_delivery(G, start, end, closure_step):
    """Full A* replan from scratch when road closes."""
    G = copy.deepcopy(G)
    total_time = 0
    total_expanded = 0
    replan_count = 0

    # Initial planning
    h = make_haversine_heuristic(G, end)
    t0 = time.perf_counter()
    lpa = LPAStar(G, start, end, h)
    cost, expanded = lpa.compute_shortest_path()
    path = lpa.extract_path()
    total_time += (time.perf_counter() - t0) * 1000
    total_expanded += expanded

    if not path:
        return {
            "success": False, "reason": "no_initial_path",
            "time_ms": total_time, "battery_left": 0,
            "replan_count": 0, "expanded": total_expanded
        }

    battery = config.BATTERY_CAPACITY_WH
    current_pos = start

    # FIX: while loop — step += 1 only on normal move, reset to 0 after replan
    step = 0
    while step < len(path) - 1:
        u, v = path[step], path[step + 1]

        if closure_step is not None and step == closure_step:
            # Close the road
            if G.has_edge(u, v):
                for k in G[u][v]:
                    G[u][v][k]['weight'] = math.inf

            # Full A* replan from current position
            h_new = make_haversine_heuristic(G, end)
            t0 = time.perf_counter()
            lpa_new = LPAStar(G, u, end, h_new)
            cost, exp = lpa_new.compute_shortest_path()
            new_path = lpa_new.extract_path()
            total_time += (time.perf_counter() - t0) * 1000
            total_expanded += exp
            replan_count += 1

            if not new_path:
                return {
                    "success": False, "reason": "no_path_after_closure",
                    "time_ms": total_time, "battery_left": battery,
                    "replan_count": replan_count, "expanded": total_expanded
                }

            path = new_path
            step = 0
            closure_step = None  # Prevent re-triggering same closure
            continue

        if G.has_edge(u, v):
            edge_data = G[u][v][min(G[u][v].keys())]
            cost = float(edge_data.get('battery_cost', 15.0))
            battery -= cost

            if battery < config.BATTERY_LOW_WH:
                return {
                    "success": False, "reason": "battery_empty",
                    "time_ms": total_time, "battery_left": battery,
                    "replan_count": replan_count, "expanded": total_expanded
                }

        current_pos = v
        step += 1  # FIX: manual increment

    return {
        "success": True, "reason": "delivered",
        "time_ms": total_time, "battery_left": battery,
        "replan_count": replan_count, "expanded": total_expanded
    }


def simulate_lpa_delivery(G, start, end, closure_step):
    """LPA* incremental replan - only updates inconsistent nodes."""
    G = copy.deepcopy(G)
    total_time = 0
    total_expanded = 0
    replan_count = 0
    replan_time_only = 0

    # Initial planning
    h = make_haversine_heuristic(G, end)
    t0 = time.perf_counter()
    lpa = LPAStar(G, start, end, h)
    cost, expanded = lpa.compute_shortest_path()
    path = lpa.extract_path()
    initial_time = (time.perf_counter() - t0) * 1000
    total_time += initial_time
    total_expanded += expanded

    if not path:
        return {
            "success": False, "reason": "no_initial_path",
            "time_ms": total_time, "initial_ms": initial_time, "replan_ms": 0,
            "battery_left": 0, "replan_count": 0, "expanded": total_expanded
        }

    battery = config.BATTERY_CAPACITY_WH
    current_pos = start

    # FIX: while loop — this is the key fix, step=0 now actually resets traversal
    step = 0
    while step < len(path) - 1:
        u, v = path[step], path[step + 1]

        if closure_step is not None and step == closure_step:
            # Incremental LPA* update — THIS is the core algorithmic difference vs A*
            t0 = time.perf_counter()
            lpa.close_road(u, v)               # Mark edge closed → update rhs values
            cost, exp = lpa.compute_shortest_path()  # Only re-expands inconsistent nodes
            new_path = lpa.extract_path()
            replan_ms = (time.perf_counter() - t0) * 1000

            total_time += replan_ms
            replan_time_only += replan_ms
            total_expanded += exp
            replan_count += 1

            if not new_path:
                return {
                    "success": False, "reason": "no_path_after_closure",
                    "time_ms": total_time, "initial_ms": initial_time,
                    "replan_ms": replan_time_only, "battery_left": battery,
                    "replan_count": replan_count, "expanded": total_expanded
                }

            path = new_path
            # Resume from current node position in new path
            step = path.index(u) if u in path else 0
            closure_step = None  # Prevent re-triggering same closure
            continue

        if G.has_edge(u, v):
            edge_data = G[u][v][min(G[u][v].keys())]
            cost = float(edge_data.get('battery_cost', 15.0))
            battery -= cost

            if battery < config.BATTERY_LOW_WH:
                return {
                    "success": False, "reason": "battery_empty",
                    "time_ms": total_time, "initial_ms": initial_time,
                    "replan_ms": replan_time_only, "battery_left": battery,
                    "replan_count": replan_count, "expanded": total_expanded
                }

        current_pos = v
        step += 1  # FIX: manual increment

    return {
        "success": True, "reason": "delivered",
        "time_ms": total_time, "initial_ms": initial_time,
        "replan_ms": replan_time_only, "battery_left": battery,
        "replan_count": replan_count, "expanded": total_expanded
    }


def simulate_lpa_battery_delivery(G, start, end, closure_step):
    """
    LPA* with battery-aware state space (node, battery_bucket).
    Guarantees battery-feasible routes.
    """
    G = copy.deepcopy(G)
    total_time = 0
    total_expanded = 0
    replan_count = 0
    replan_time_only = 0

    h = make_battery_heuristic(G, end)
    t0 = time.perf_counter()
    lpa = LPAStarBattery(G, start, end, config.BATTERY_CAPACITY_WH, h)
    cost, expanded = lpa.compute_shortest_path()
    path_with_battery = lpa.extract_path()
    initial_time = (time.perf_counter() - t0) * 1000
    total_time += initial_time
    total_expanded += expanded

    if not path_with_battery:
        return {
            "success": False, "reason": "no_initial_path",
            "time_ms": total_time, "initial_ms": initial_time, "replan_ms": 0,
            "battery_left": 0, "replan_count": 0, "expanded": total_expanded,
            "battery_feasible": False
        }

    path = [p[0] for p in path_with_battery]
    battery = config.BATTERY_CAPACITY_WH

    # FIX: while loop
    step = 0
    while step < len(path) - 1:
        u, v = path[step], path[step + 1]

        if closure_step is not None and step == closure_step:
            t0 = time.perf_counter()
            lpa.close_road(u, v)
            cost, exp = lpa.compute_shortest_path()
            new_path_battery = lpa.extract_path()
            replan_ms = (time.perf_counter() - t0) * 1000

            total_time += replan_ms
            replan_time_only += replan_ms
            total_expanded += exp
            replan_count += 1

            if not new_path_battery:
                return {
                    "success": False, "reason": "no_path_after_closure",
                    "time_ms": total_time, "initial_ms": initial_time,
                    "replan_ms": replan_time_only, "battery_left": battery,
                    "replan_count": replan_count, "expanded": total_expanded,
                    "battery_feasible": True
                }

            path = [p[0] for p in new_path_battery]
            step = path.index(u) if u in path else 0
            closure_step = None  # Prevent re-triggering
            continue

        if G.has_edge(u, v):
            edge_data = G[u][v][min(G[u][v].keys())]
            cost = float(edge_data.get('battery_cost', 15.0))
            battery -= cost

            if battery < config.BATTERY_LOW_WH:
                return {
                    "success": False, "reason": "battery_empty",
                    "time_ms": total_time, "initial_ms": initial_time,
                    "replan_ms": replan_time_only, "battery_left": battery,
                    "replan_count": replan_count, "expanded": total_expanded,
                    "battery_feasible": False  # Should never happen with battery-aware
                }

        step += 1  # FIX: manual increment

    return {
        "success": True, "reason": "delivered",
        "time_ms": total_time, "initial_ms": initial_time,
        "replan_ms": replan_time_only, "battery_left": battery,
        "replan_count": replan_count, "expanded": total_expanded,
        "battery_feasible": True
    }


def run_full_benchmark(n=500):
    print("=" * 70)
    print("EV DELIVERY ROUTER BENCHMARK (FIXED - while loops)")
    print("=" * 70)

    print("\nLoading processed graph...")
    G = ox.load_graphml(config.PROCESSED_GRAPH_PATH)
    print("Converting graph weights...")
    G = fix_graph_weights(G)

    print("Loading battery model...")
    battery_model = load_battery_model()

    nodes = list(G.nodes())
    print(f"Graph: {len(nodes)} nodes, {len(list(G.edges()))} edges")
    print(f"\nRunning {n} deliveries with dynamic road closures...\n")

    random.seed(config.RANDOM_SEED)
    results = []
    skipped = 0

    for i in tqdm(range(n), desc="Testing deliveries"):
        start = random.choice(nodes)
        end = random.choice(nodes)

        if start == end:
            skipped += 1
            continue

        try:
            nx.shortest_path_length(G, start, end, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped += 1
            continue

        for scenario in ["no_closure", "with_closure"]:
            closure_step = None if scenario == "no_closure" else random.randint(15, 40)

            row = {
                "trial": i,
                "scenario": scenario,
                "closure_step": closure_step
            }

            # Method 1: Static Dijkstra
            res_dijkstra = simulate_delivery_with_dynamic_closure(
                G, start, end, 'dijkstra', closure_step)
            row["dijkstra_success"]      = res_dijkstra["success"]
            row["dijkstra_reason"]       = res_dijkstra["reason"]
            row["dijkstra_time_ms"]      = round(res_dijkstra["time_ms"], 4)
            row["dijkstra_battery_left"] = round(res_dijkstra["battery_left"], 2)

            # Method 2: Full A* Replan
            res_astar = simulate_delivery_with_dynamic_closure(
                G, start, end, 'astar_replan', closure_step)
            row["astar_success"]      = res_astar["success"]
            row["astar_reason"]       = res_astar["reason"]
            row["astar_time_ms"]      = round(res_astar["time_ms"], 4)
            row["astar_battery_left"] = round(res_astar["battery_left"], 2)
            row["astar_replan_count"] = res_astar["replan_count"]
            row["astar_expanded"]     = res_astar["expanded"]

            # Method 3: LPA* Incremental
            res_lpa = simulate_delivery_with_dynamic_closure(
                G, start, end, 'lpa_incremental', closure_step)
            row["lpa_success"]       = res_lpa["success"]
            row["lpa_reason"]        = res_lpa["reason"]
            row["lpa_total_ms"]      = round(res_lpa["time_ms"], 4)
            row["lpa_initial_ms"]    = round(res_lpa.get("initial_ms", 0), 4)
            row["lpa_replan_ms"]     = round(res_lpa.get("replan_ms", 0), 4)
            row["lpa_battery_left"]  = round(res_lpa["battery_left"], 2)
            row["lpa_replan_count"]  = res_lpa["replan_count"]
            row["lpa_expanded"]      = res_lpa["expanded"]

            # Speedup (replan times only — apples-to-apples comparison)
            lpa_replan_ms  = res_lpa.get("replan_ms", 0)
            astar_replan_ms = (res_astar["time_ms"] - res_lpa.get("initial_ms", 0)
                               if res_astar["replan_count"] > 0 else 0)
            if lpa_replan_ms > 0 and astar_replan_ms > 0:
                row["speedup_vs_astar"] = round(astar_replan_ms / lpa_replan_ms, 2)
            else:
                row["speedup_vs_astar"] = 0

            # Method 4: LPA* Battery-Aware
            res_lpa_bat = simulate_lpa_battery_delivery(G, start, end, closure_step)
            row["lpa_battery_success"]      = res_lpa_bat["success"]
            row["lpa_battery_reason"]       = res_lpa_bat["reason"]
            row["lpa_battery_total_ms"]     = round(res_lpa_bat["time_ms"], 4)
            row["lpa_battery_initial_ms"]   = round(res_lpa_bat.get("initial_ms", 0), 4)
            row["lpa_battery_replan_ms"]    = round(res_lpa_bat.get("replan_ms", 0), 4)
            row["lpa_battery_left"]         = round(res_lpa_bat["battery_left"], 2)
            row["lpa_battery_replan_count"] = res_lpa_bat["replan_count"]
            row["lpa_battery_expanded"]     = res_lpa_bat["expanded"]
            row["lpa_battery_feasible"]     = res_lpa_bat.get("battery_feasible", True)

            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(config.BENCHMARK_RESULTS, index=False)

    # ───────────────────────────────────────────────────────────────────────
    # RESULTS
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS — {len(df)} trials ({skipped} skipped)")
    print(f"{'='*70}")

    no_closure   = df[df['scenario'] == 'no_closure']
    with_closure = df[df['scenario'] == 'with_closure']

    for label, subset in [("NO ROAD CLOSURES", no_closure),
                          ("WITH DYNAMIC ROAD CLOSURES", with_closure)]:
        print(f"\n{label} ({len(subset)} trials)")
        print(f"{'Method':<25} {'Success%':>10} {'Avg Time':>12} {'Battery Fail':>14}")
        print("-" * 65)
        for method, prefix in [("Dijkstra (static)", "dijkstra"),
                                ("Full A* replan",    "astar"),
                                ("LPA* incremental",  "lpa"),
                                ("LPA* Battery-Aware","lpa_battery")]:
            success_pct = subset[f"{prefix}_success"].mean() * 100
            if prefix == "lpa":
                avg_time = subset["lpa_total_ms"].mean()
            elif prefix == "lpa_battery":
                avg_time = subset["lpa_battery_total_ms"].mean()
            else:
                avg_time = subset[f"{prefix}_time_ms"].mean()
            bat_fail = (subset[f"{prefix}_reason"] == "battery_empty").sum()
            print(f"{method:<25} {success_pct:>9.1f}% {avg_time:>10.2f} ms {bat_fail:>12}")

    # Replanning performance (only trials where LPA* actually replanned)
    replanned = with_closure[with_closure['lpa_replan_ms'] > 0]
    if len(replanned) > 0:
        avg_lpa_replan   = replanned['lpa_replan_ms'].mean()
        avg_astar_replan = (replanned['astar_time_ms'] - replanned['lpa_initial_ms']).mean()
        avg_speedup      = replanned['speedup_vs_astar'].replace(0, float('nan')).mean()
        avg_lpa_exp      = replanned['lpa_expanded'].mean()
        avg_astar_exp    = replanned['astar_expanded'].mean()
        reduction        = (1 - avg_lpa_exp / avg_astar_exp) * 100 if avg_astar_exp > 0 else 0

        print(f"\nREPLANNING PERFORMANCE ({len(replanned)} trials with actual replans)")
        print("-" * 45)
        print(f"LPA* avg replan time:      {avg_lpa_replan:.2f} ms")
        print(f"A*  avg replan time:       {avg_astar_replan:.2f} ms")
        print(f"LPA* speedup:              {avg_speedup:.1f}x faster")
        print(f"LPA* avg nodes expanded:   {avg_lpa_exp:,.0f}")
        print(f"A*  avg nodes expanded:    {avg_astar_exp:,.0f}")
        print(f"Node expansion reduction:  {reduction:.1f}%")
    else:
        print("\nWARNING: No trials with LPA* replans detected — check close_road() wiring")

    # Battery-Aware analysis
    print(f"\nBATTERY-AWARE LPA* ANALYSIS")
    print("-" * 45)
    print(f"LPA* battery failures:           {(df['lpa_reason']=='battery_empty').sum()}")
    print(f"LPA* Battery-Aware failures:     {(df['lpa_battery_reason']=='battery_empty').sum()}")
    if "lpa_battery_feasible" in df.columns:
        print(f"Battery-feasible routes:         {df['lpa_battery_feasible'].mean()*100:.1f}%")

    # Failure analysis
    print(f"\nFAILURE ANALYSIS (with_closure trials)")
    print("-" * 45)
    blocked = (with_closure["dijkstra_reason"] == "road_blocked_no_replan").sum()
    print(f"Dijkstra blocked by closures: {blocked}/{len(with_closure)} "
          f"({blocked/len(with_closure)*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"Results saved to: {config.BENCHMARK_RESULTS}")
    print(f"{'='*70}")

    return df


if __name__ == "__main__":
    run_full_benchmark(n=config.N_DELIVERIES)
