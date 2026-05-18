# src/lpa_star_battery.py
#
# LPA* on expanded state space: (node_id, battery_bucket)
# battery_bucket: 0 = empty, 49 = full
# Each bucket = WH_PER_BUCKET Wh (BATTERY_CAPACITY_WH / N_BUCKETS)
# N_BUCKETS = 50, so each bucket = 800 Wh for a 40 kWh car battery

import heapq
import math
from collections import defaultdict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

N_BUCKETS        = 50
WH_PER_BUCKET    = config.BATTERY_CAPACITY_WH / N_BUCKETS   # 800 Wh per bucket for 40kWh battery


def wh_to_bucket(wh):
    """Convert watt-hours to bucket index 0-49."""
    bucket = int(wh / WH_PER_BUCKET)
    return max(0, min(N_BUCKETS - 1, bucket))


def bucket_to_wh(bucket):
    """Convert bucket index to watt-hours (midpoint of bucket)."""
    return (bucket + 0.5) * WH_PER_BUCKET


class LPAStarBattery:
    """
    LPA* on state space (node_id, battery_bucket).

    Key difference from plain LPA*:
    - A state is (node, battery_bucket) not just node
    - 50 discrete battery buckets (0=empty, 49=full, 800 Wh each)
    - Edge (u,b) → (v,b') only exists if battery after consuming
      edge cost is >= 0. Infeasible transitions are pruned in
      _get_successors(), so any returned path is battery-guaranteed.
    - Goal is reached when node == end_node (any battery bucket)
    """

    def __init__(self, graph, start_node, end_node,
                 start_battery_wh, heuristic_fn):
        self.G          = graph
        self.start_node = start_node
        self.end_node   = end_node
        self.h          = heuristic_fn

        start_bucket   = wh_to_bucket(start_battery_wh)
        self.start     = (start_node, start_bucket)

        self.goal      = None
        self.goal_cost = math.inf

        self.g   = defaultdict(lambda: math.inf)
        self.rhs = defaultdict(lambda: math.inf)

        self._heap    = []
        self._in_heap = {}

        self.nodes_expanded = 0

        self.rhs[self.start] = 0.0
        self._push(self.start, self._key(self.start))

    # ── Priority queue helpers ──────────────────────────────

    def _key(self, state):
        node, bucket = state
        m = min(self.g[state], self.rhs[state])
        return (m + self.h(node), m)

    def _push(self, state, key):
        heapq.heappush(self._heap, (key, state))
        self._in_heap[state] = key

    def _top_key(self):
        while self._heap:
            key, state = self._heap[0]
            if self._in_heap.get(state) == key:
                return key
            heapq.heappop(self._heap)
        return (math.inf, math.inf)

    def _pop_min(self):
        while self._heap:
            key, state = heapq.heappop(self._heap)
            if self._in_heap.get(state) == key:
                del self._in_heap[state]
                return state, key
        return None, None

    # ── Edge weight helpers ─────────────────────────────────

    def _get_edge_cost(self, u, v):
        """Get battery cost (Wh) for edge u→v."""
        if not self.G.has_edge(u, v):
            return math.inf
        edge_dict = self.G[u][v]
        min_cost  = math.inf
        for k in edge_dict:
            raw = edge_dict[k].get('battery_cost', None)
            try:
                cost = float(raw) if raw is not None else math.inf
            except (ValueError, TypeError):
                cost = math.inf
            if cost < min_cost:
                min_cost = cost
        return min_cost

    def _get_successors(self, state):
        """
        Returns list of (next_state, edge_cost) from current state.
        Transitions where battery would go below 0 are pruned here —
        this is what guarantees battery-feasible paths.
        """
        node, bucket = state
        current_wh   = bucket_to_wh(bucket)
        successors   = []

        for v in self.G.successors(node):
            edge_cost = self._get_edge_cost(node, v)
            if edge_cost == math.inf:
                continue

            remaining_wh = current_wh - edge_cost

            # Battery constraint: prune if battery goes negative
            if remaining_wh < 0:
                continue

            new_bucket = wh_to_bucket(remaining_wh)
            next_state = (v, new_bucket)
            successors.append((next_state, edge_cost))

        return successors

    def _get_predecessors(self, state):
        """
        Returns list of (prev_state, edge_cost) leading to state.
        Used to compute rhs values.
        """
        node, bucket = state
        current_wh   = bucket_to_wh(bucket)
        predecessors = []

        for u in self.G.predecessors(node):
            edge_cost = self._get_edge_cost(u, node)
            if edge_cost == math.inf:
                continue

            needed_wh   = current_wh + edge_cost
            prev_bucket = wh_to_bucket(needed_wh)

            if needed_wh > config.BATTERY_CAPACITY_WH:
                continue

            prev_state = (u, prev_bucket)
            predecessors.append((prev_state, edge_cost))

        return predecessors

    # ── Core LPA* ───────────────────────────────────────────

    def _update_vertex(self, state):
        if state != self.start:
            best_rhs = math.inf
            for prev_state, cost in self._get_predecessors(state):
                candidate = self.g[prev_state] + cost
                if candidate < best_rhs:
                    best_rhs = candidate
            self.rhs[state] = best_rhs

        if state in self._in_heap:
            del self._in_heap[state]

        if self.g[state] != self.rhs[state]:
            self._push(state, self._key(state))

    def _is_goal(self, state):
        return state[0] == self.end_node

    def compute_shortest_path(self):
        """
        Run LPA* until best goal state is consistent.
        Returns (best_cost, nodes_expanded).

        FIX: best goal state is now updated only when we expand a goal
        node, instead of scanning all N_BUCKETS on every iteration.
        """
        expanded = 0

        while True:
            top = self._top_key()

            # Termination: heap is empty or top key >= best consistent goal
            best_goal_key = self._key(self.goal) if self.goal else (math.inf, math.inf)
            if top >= best_goal_key and self.goal is not None and \
               self.rhs[self.goal] == self.g[self.goal]:
                break

            state, old_key = self._pop_min()
            if state is None:
                break

            expanded += 1
            self.nodes_expanded += 1

            if expanded > 500_000:
                print("WARNING: expansion limit hit")
                break

            # FIX: update best goal only when we actually expand a goal node
            # Previously scanned all 50 buckets on every loop iteration — O(50) overhead per step
            if self._is_goal(state):
                if self.rhs[state] == self.g[state] or self.g[state] > self.rhs[state]:
                    tentative_cost = self.rhs[state]
                    if tentative_cost < self.goal_cost:
                        self.goal_cost = tentative_cost
                        self.goal      = state

            new_key = self._key(state)

            if old_key < new_key:
                self._push(state, new_key)
            elif self.g[state] > self.rhs[state]:
                self.g[state] = self.rhs[state]
                for next_state, _ in self._get_successors(state):
                    self._update_vertex(next_state)
            else:
                self.g[state] = math.inf
                self._update_vertex(state)
                for next_state, _ in self._get_successors(state):
                    self._update_vertex(next_state)

        return self.goal_cost, expanded

    def extract_path(self):
        """
        Backtrack from best goal state to start.
        Returns list of (node_id, battery_wh) tuples.
        """
        if self.goal is None or self.goal_cost == math.inf:
            return None

        path    = [self.goal]
        current = self.goal
        visited = {self.goal}

        while current != self.start:
            best_prev = None
            best_cost = math.inf

            for prev_state, cost in self._get_predecessors(current):
                total = self.g[prev_state] + cost
                if total < best_cost:
                    best_cost = total
                    best_prev = prev_state

            if best_prev is None or best_prev in visited:
                return None

            visited.add(best_prev)
            path.append(best_prev)
            current = best_prev

        path.reverse()

        readable = [(state[0], round(bucket_to_wh(state[1]), 1))
                    for state in path]
        return readable

    def close_road(self, u, v):
        """Close road u→v and mark all states at v inconsistent."""
        if self.G.has_edge(u, v):
            for k in self.G[u][v]:
                self.G[u][v][k]['battery_cost'] = math.inf
                self.G[u][v][k]['weight']        = math.inf

        for b in range(N_BUCKETS):
            self._update_vertex((v, b))

    def get_battery_profile(self):
        """Returns battery level at each node along the path."""
        path = self.extract_path()
        if not path:
            return []
        return [(node, wh, round(wh / config.BATTERY_CAPACITY_WH * 100, 1))
                for node, wh in path]


def make_battery_heuristic(graph, end_node):
    """
    Admissible heuristic for (node, battery_bucket) state space.
    Uses straight-line distance × MIN_ENERGY_PER_KM.
    Must never overestimate — so we use a conservative lower bound.
    """
    end_lat = math.radians(graph.nodes[end_node]['y'])
    end_lon = math.radians(graph.nodes[end_node]['x'])
    MIN_ENERGY_PER_KM = 8.0  # Conservative lower bound (actual ~180 Wh/km for car)

    def h(node):
        lat  = math.radians(graph.nodes[node]['y'])
        lon  = math.radians(graph.nodes[node]['x'])
        dlat = end_lat - lat
        dlon = end_lon - lon
        a    = (math.sin(dlat / 2) ** 2
                + math.cos(lat) * math.cos(end_lat) * math.sin(dlon / 2) ** 2)
        dist_km = 6371.0 * 2.0 * math.asin(math.sqrt(max(0, a)))
        return dist_km * MIN_ENERGY_PER_KM

    return h