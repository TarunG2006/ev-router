# src/simulation.py

import time
import random
import math
import copy
import networkx as nx
from src.lpa_star import LPAStar, make_haversine_heuristic
from src.battery_model import load_battery_model, predict_edge_cost
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DeliverySimulation:

    def __init__(self, delivery_id, graph, start_node, end_node, battery_model):
        self.id            = delivery_id
        self.G             = graph
        self.start         = start_node
        self.end           = end_node
        self.battery_model = battery_model
        self.battery       = config.BATTERY_CAPACITY_WH
        self.position      = start_node
        self.path          = []
        self.lpa           = None
        self.log           = []

    def _get_edge_data(self, u, v):
        edge_dict = self.G[u][v]
        key = min(edge_dict.keys())
        return edge_dict[key]

    def _plan_initial_route(self):
        h = make_haversine_heuristic(self.G, self.end)
        self.lpa = LPAStar(self.G, self.start, self.end, h)
        cost, expanded = self.lpa.compute_shortest_path()
        self.path = self.lpa.extract_path()
        self.log.append({
            "event"    : "initial_plan",
            "cost_wh"  : cost,
            "path_len" : len(self.path) if self.path else 0,
            "expanded" : expanded
        })
        return self.path is not None

    def _do_replan(self, reason):
        t0 = time.perf_counter()
        cost, expanded = self.lpa.compute_shortest_path()
        new_path = self.lpa.extract_path()
        replan_ms = (time.perf_counter() - t0) * 1000
        self.path = new_path
        self.log.append({
            "event"        : "replan",
            "reason"       : reason,
            "replan_ms"    : round(replan_ms, 4),
            "expanded"     : expanded,
            "new_path_len" : len(new_path) if new_path else 0
        })
        return new_path is not None, replan_ms

    def run(self, road_closures=None, battery_noise_std=0.10):
        random.seed()

        if not nx.has_path(self.G, self.start, self.end):
            return self._result(success=False, reason="no_path_exists")

        if not self._plan_initial_route() or not self.path:
            return self._result(success=False, reason="initial_plan_failed")

        road_closures = road_closures or []
        closure_at = {}
        for u, v, after_step in road_closures:
            closure_at[after_step] = (u, v)

        steps_taken     = 0
        total_replan_ms = 0
        n_replans       = 0

        step = 0
        while self.path and step < len(self.path) - 1:
            if not self.path or step >= len(self.path) - 1:
                break

            if self.battery < config.BATTERY_LOW_WH:
                return self._result(success=False, reason="battery_empty",
                                    steps=steps_taken, replans=n_replans,
                                    total_replan_ms=total_replan_ms)

            u = self.path[step]
            v = self.path[step + 1]

            if not self.G.has_edge(u, v):
                ok, ms = self._do_replan("edge_missing")
                if not ok:
                    return self._result(success=False, reason="replan_failed")
                n_replans += 1
                total_replan_ms += ms
                step = 0
                continue

            edata = self._get_edge_data(u, v)

            # ALWAYS cast to float/int — GraphML loads everything as strings
            dist_km   = float(edata.get('length', 100)) / 1000
            slope     = float(edata.get('slope', 0.0))
            speed     = float(edata.get('speed_kph', 30.0))
            road_type = int(float(edata.get('road_type', 0)))
            predicted = float(edata.get('battery_cost', 1.0))

            noise  = random.gauss(1.0, battery_noise_std)
            actual = float(predict_edge_cost(
                self.battery_model, slope, dist_km, speed, road_type
            )) * max(0.5, noise)

            self.battery -= actual
            self.position = v
            steps_taken  += 1

            # Disruption 1: Road closure
            if step in closure_at:
                cu, cv = closure_at[step]
                self.lpa.close_road(cu, cv)
                self.log.append({"event": "road_closed", "edge": (cu, cv), "step": step})
                ok, ms = self._do_replan("road_closure")
                if not ok:
                    return self._result(success=False, reason="no_path_after_closure")
                n_replans += 1
                total_replan_ms += ms
                step = 0
                continue

            # Disruption 2: Battery deviation > 15%
            if predicted > 0:
                deviation = abs(actual - predicted) / predicted
                if deviation > config.BATTERY_DEVIATION_THRESHOLD:
                    scale     = actual / predicted
                    remaining = self.path[step + 1:]
                    for ru, rv in zip(remaining[:-1], remaining[1:]):
                        if self.G.has_edge(ru, rv):
                            for k in self.G[ru][rv]:
                                old_w = float(self.G[ru][rv][k].get('weight', 1.0))
                                new_w = old_w * scale
                                self.G[ru][rv][k]['weight'] = new_w
                                self.lpa.update_edge_weight(ru, rv, new_w)
                    self.log.append({"event": "battery_deviation",
                                     "deviation_pct": round(deviation * 100, 2),
                                     "scale": round(scale, 3), "step": step})
                    ok, ms = self._do_replan("battery_deviation")
                    if ok:
                        n_replans += 1
                        total_replan_ms += ms
                    step = 0
                    continue

            step += 1

        avg_replan = total_replan_ms / n_replans if n_replans > 0 else 0.0
        return self._result(
            success         = (self.battery > 0),
            reason          = "delivered",
            steps           = steps_taken,
            replans         = n_replans,
            total_replan_ms = total_replan_ms,
            avg_replan_ms   = avg_replan,
            battery_left    = max(0.0, self.battery)
        )

    def _result(self, success, reason="", steps=0, replans=0,
                total_replan_ms=0.0, avg_replan_ms=0.0, battery_left=0.0):
        return {
            "id"              : self.id,
            "success"         : success,
            "reason"          : reason,
            "steps_taken"     : steps,
            "replans"         : replans,
            "total_replan_ms" : round(total_replan_ms, 4),
            "avg_replan_ms"   : round(avg_replan_ms, 4),
            "battery_left_wh" : round(battery_left, 2),
            "log"             : self.log
        }
