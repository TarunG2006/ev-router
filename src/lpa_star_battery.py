# src/lpa_star_battery.py
#
# LPA* on expanded state space: (node_id, battery_bucket)
# battery_bucket: 0 = empty, 9 = full (each bucket = 10% of capacity)

import heapq
import math
from collections import defaultdict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

N_BUCKETS        = 50
WH_PER_BUCKET    = config.BATTERY_CAPACITY_WH / N_BUCKETS   # 100 Wh per bucket


def wh_to_bucket(wh):
    """Convert watt-hours to bucket index 0-9."""
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
    - Edge (u,b) → (v,b') only exists if battery_bucket after
      consuming edge cost is >= 0
    - Goal is reached when node == end_node (any battery bucket)
    - This GUARANTEES the found route is battery-feasible
    """

    def __init__(self, graph, start_node, end_node,
                 start_battery_wh, heuristic_fn):
        self.G          = graph
        self.start_node = start_node
        self.end_node   = end_node
        self.h          = heuristic_fn

        # Starting state
        start_bucket   = wh_to_bucket(start_battery_wh)
        self.start     = (start_node, start_bucket)

        # Goal: any state where node == end_node
        # We track best goal state found
        self.goal      = None
        self.goal_cost = math.inf

        # g and rhs over (node, bucket) states
        self.g   = defaultdict(lambda: math.inf)
        self.rhs = defaultdict(lambda: math.inf)

        self._heap    = []
        self._in_heap = {}

        self.nodes_expanded = 0

        # Bootstrap
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
        next_state is (v, new_bucket).
        Transition is INVALID if battery would hit 0.
        """
        node, bucket = state
        current_wh   = bucket_to_wh(bucket)
        successors   = []

        for v in self.G.successors(node):
            edge_cost = self._get_edge_cost(node, v)
            if edge_cost == math.inf:
                continue

            remaining_wh = current_wh - edge_cost

            # Battery constraint: can't go below 0
            if remaining_wh < 0:
                continue

            new_bucket = wh_to_bucket(remaining_wh)
            next_state = (v, new_bucket)
            successors.append((next_state, edge_cost))

        return successors

    def _get_predecessors(self, state):
        """
        Returns list of (prev_state, edge_cost) leading to state.
        Used to compute rhs.
        """
        node, bucket = state
        current_wh   = bucket_to_wh(bucket)
        predecessors = []

        for u in self.G.predecessors(node):
            edge_cost = self._get_edge_cost(u, node)
            if edge_cost == math.inf:
                continue

            # What battery would we have needed at u to arrive here?
            needed_wh  = current_wh + edge_cost
            prev_bucket = wh_to_bucket(needed_wh)

            # Check it's a valid bucket
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
        """
        expanded = 0

        # Find best goal state key
        best_goal_key  = (math.inf, math.inf)
        best_goal_state = None

        # Check all possible goal states (end_node, bucket 0-9)
        for b in range(N_BUCKETS):
            gs  = (self.end_node, b)
            key = self._key(gs)
            if key < best_goal_key:
                best_goal_key   = key
                best_goal_state = gs

        if best_goal_state:
            self.goal = best_goal_state

        while True:
            top = self._top_key()

            # Recompute best goal key
            best_goal_key = (math.inf, math.inf)
            for b in range(N_BUCKETS):
                gs  = (self.end_node, b)
                gk  = self._key(gs)
                grhs = self.rhs[gs]
                gg   = self.g[gs]
                if grhs == gg and grhs < math.inf:
                    if gk < best_goal_key:
                        best_goal_key   = gk
                        self.goal       = gs
                        self.goal_cost  = grhs

            if top >= best_goal_key:
                break

            state, old_key = self._pop_min()
            if state is None:
                break

            expanded += 1
            self.nodes_expanded += 1

            if expanded > 500_000:
                print("WARNING: expansion limit hit")
                break

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

        # Convert to (node_id, battery_wh) for readability
        readable = [(state[0], round(bucket_to_wh(state[1]), 1))
                    for state in path]
        return readable

    def close_road(self, u, v):
        """Close road u→v and mark affected states inconsistent."""
        if self.G.has_edge(u, v):
            for k in self.G[u][v]:
                self.G[u][v][k]['battery_cost'] = math.inf
                self.G[u][v][k]['weight']        = math.inf

        # All states at v become potentially inconsistent
        for b in range(N_BUCKETS):
            self._update_vertex((v, b))

    def get_battery_profile(self):
        """
        Returns battery level at each node along the path.
        Useful for visualization.
        """
        path = self.extract_path()
        if not path:
            return []
        return [(node, wh, round(wh/config.BATTERY_CAPACITY_WH*100, 1))
                for node, wh in path]


def make_battery_heuristic(graph, end_node):
    """
    Admissible heuristic for (node, battery_bucket) state space.
    Uses straight-line distance × min energy per km.
    """
    end_lat = math.radians(graph.nodes[end_node]['y'])
    end_lon = math.radians(graph.nodes[end_node]['x'])
    MIN_ENERGY_PER_KM = 8.0  # Conservative lower bound

    def h(node):
        lat = math.radians(graph.nodes[node]['y'])
        lon = math.radians(graph.nodes[node]['x'])
        dlat = end_lat - lat
        dlon = end_lon - lon
        a = (math.sin(dlat/2)**2
             + math.cos(lat) * math.cos(end_lat) * math.sin(dlon/2)**2)
        dist_km = 6371.0 * 2.0 * math.asin(math.sqrt(max(0, a)))
        return dist_km * MIN_ENERGY_PER_KM

    return h
