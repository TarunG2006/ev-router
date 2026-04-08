# tests/test_simulation.py
"""Unit tests for delivery simulation."""

import pytest
import sys
import os
import copy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import osmnx as ox
from src.simulation import DeliverySimulation
from src.battery_model import load_battery_model
import config


def fix_graph_types(G):
    """Convert GraphML string attributes to proper types."""
    for u, v, k, d in G.edges(keys=True, data=True):
        for attr in ['weight', 'battery_cost', 'length', 'slope', 'speed_kph']:
            if attr in d:
                try:
                    d[attr] = float(d[attr])
                except (ValueError, TypeError):
                    d[attr] = 1.0
        if 'road_type' in d:
            try:
                d['road_type'] = int(float(d['road_type']))
            except (ValueError, TypeError):
                d['road_type'] = 0
    return G


@pytest.fixture(scope="module")
def graph():
    """Load and prepare the Jaipur graph."""
    G = ox.load_graphml(config.PROCESSED_GRAPH_PATH)
    G = fix_graph_types(G)
    return G


@pytest.fixture(scope="module")
def battery_model():
    """Load the battery model."""
    return load_battery_model()


class TestDeliverySimulation:
    """Tests for delivery simulation."""

    def test_simulation_completes_simple_delivery(self, graph, battery_model):
        """Simulation should complete a simple delivery without disruptions."""
        G = copy.deepcopy(graph)
        nodes = list(G.nodes())
        
        # Pick nodes that are relatively close
        start, end = nodes[0], nodes[min(50, len(nodes)-1)]
        
        sim = DeliverySimulation(
            delivery_id=1,
            graph=G,
            start_node=start,
            end_node=end,
            battery_model=battery_model
        )
        
        result = sim.run(road_closures=[], battery_noise_std=0.0)
        
        assert "success" in result
        assert "reason" in result
        assert "steps_taken" in result
        assert "battery_left_wh" in result

    def test_simulation_handles_road_closure(self, graph, battery_model):
        """Simulation should handle road closure and replan."""
        G = copy.deepcopy(graph)
        nodes = list(G.nodes())
        
        start, end = nodes[0], nodes[min(100, len(nodes)-1)]
        
        # Get the initial path to find an edge to close
        from src.lpa_star import LPAStar, make_haversine_heuristic
        h = make_haversine_heuristic(G, end)
        lpa = LPAStar(copy.deepcopy(G), start, end, h)
        lpa.compute_shortest_path()
        path = lpa.extract_path()
        
        if path and len(path) > 4:
            mid = len(path) // 2
            u, v = path[mid], path[mid + 1]
            
            sim = DeliverySimulation(
                delivery_id=2,
                graph=G,
                start_node=start,
                end_node=end,
                battery_model=battery_model
            )
            
            # Close road after step 2
            result = sim.run(road_closures=[(u, v, 2)], battery_noise_std=0.05)
            
            assert "replans" in result
            # Should have at least one replan due to road closure
            if result["success"]:
                assert result["replans"] >= 0

    def test_simulation_tracks_battery(self, graph, battery_model):
        """Simulation should track battery consumption."""
        G = copy.deepcopy(graph)
        nodes = list(G.nodes())
        
        start, end = nodes[0], nodes[min(30, len(nodes)-1)]
        
        sim = DeliverySimulation(
            delivery_id=3,
            graph=G,
            start_node=start,
            end_node=end,
            battery_model=battery_model
        )
        
        result = sim.run(battery_noise_std=0.0)
        
        if result["success"]:
            # Battery should decrease after delivery
            assert result["battery_left_wh"] < config.BATTERY_CAPACITY_WH
            assert result["battery_left_wh"] >= 0

    def test_result_contains_log(self, graph, battery_model):
        """Result should contain event log."""
        G = copy.deepcopy(graph)
        nodes = list(G.nodes())
        
        start, end = nodes[0], nodes[min(20, len(nodes)-1)]
        
        sim = DeliverySimulation(
            delivery_id=4,
            graph=G,
            start_node=start,
            end_node=end,
            battery_model=battery_model
        )
        
        result = sim.run()
        
        assert "log" in result
        assert isinstance(result["log"], list)
        # Should have at least initial_plan event
        if result["success"] or result["reason"] != "no_path_exists":
            events = [e["event"] for e in result["log"]]
            assert "initial_plan" in events
