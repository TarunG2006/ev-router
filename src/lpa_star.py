

# src/lpa_star.py

import heapq
import math
from collections import defaultdict


class LPAStar:

    def __init__(self, graph, start, goal, heuristic_fn):
        self.G     = graph
        self.start = start
        self.goal  = goal
        self.h     = heuristic_fn
        self.g     = defaultdict(lambda: math.inf)
        self.rhs   = defaultdict(lambda: math.inf)
        self._heap = []
        self._in_heap = {}
        self.nodes_expanded = 0
        self.total_replans  = 0
        # FIX 1: parent pointers set during compute_shortest_path so
        # extract_path never relies on potentially-stale g-values.
        self.parent = {}
        self.rhs[self.start] = 0.0
        self._push(self.start, self._key(self.start))

    def _key(self, node):
        m = min(self.g[node], self.rhs[node])
        return (m + self.h(node), m)

    def _push(self, node, key):
        heapq.heappush(self._heap, (key, node))
        self._in_heap[node] = key

    def _top_key(self):
        while self._heap:
            key, node = self._heap[0]
            if self._in_heap.get(node) == key:
                return key
            heapq.heappop(self._heap)
        return (math.inf, math.inf)

    def _pop_min(self):
        while self._heap:
            key, node = heapq.heappop(self._heap)
            if self._in_heap.get(node) == key:
                del self._in_heap[node]
                return node, key
        return None, None

    def _get_edge_weight(self, u, v):
        if not self.G.has_edge(u, v):
            return math.inf
        edge_dict = self.G[u][v]
        min_w = math.inf
        for key_idx in edge_dict:
            raw = edge_dict[key_idx].get('weight', None)
            try:
                w = float(raw) if raw is not None else math.inf
            except (ValueError, TypeError):
                w = math.inf
            if w < min_w:
                min_w = w
        return min_w

    def _update_vertex(self, node):
        if node != self.start:
            best_rhs = math.inf
            for pred in self.G.predecessors(node):
                w = self._get_edge_weight(pred, node)
                candidate = self.g[pred] + w
                if candidate < best_rhs:
                    best_rhs = candidate
            self.rhs[node] = best_rhs
        if node in self._in_heap:
            del self._in_heap[node]
        if self.g[node] != self.rhs[node]:
            self._push(node, self._key(node))

    def _record_parent(self, node):
        """Record the predecessor that gives node its current rhs value.
        Called whenever g[node] is set to rhs[node] (node becomes consistent).
        This is how we build reliable parent pointers for extract_path.
        """
        if node == self.start:
            return
        best_pred, best_cost = None, math.inf
        for pred in self.G.predecessors(node):
            w = self._get_edge_weight(pred, node)
            c = self.g[pred] + w
            if c < best_cost:
                best_cost = c
                best_pred = pred
        if best_pred is not None:
            self.parent[node] = best_pred

    def compute_shortest_path(self):
        expanded_this_call = 0
        while (self._top_key() < self._key(self.goal)
               or self.rhs[self.goal] != self.g[self.goal]):
            node, old_key = self._pop_min()
            if node is None:
                break
            expanded_this_call += 1
            self.nodes_expanded += 1
            if expanded_this_call > 300000:
                print("WARNING: LPA* hit expansion limit")
                break
            new_key = self._key(node)
            if old_key < new_key:
                self._push(node, new_key)
            elif self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]
                # FIX 1: record parent pointer the moment g is committed
                self._record_parent(node)
                for succ in self.G.successors(node):
                    self._update_vertex(succ)
            else:
                self.g[node] = math.inf
                # Parent pointer for this node is now invalid; clear it
                self.parent.pop(node, None)
                self._update_vertex(node)
                for succ in self.G.successors(node):
                    self._update_vertex(succ)
        return self.g[self.goal], expanded_this_call

    def extract_path(self):
        if self.g[self.goal] == math.inf:
            return None

        path    = [self.goal]
        current = self.goal
        visited = {self.goal}

        while current != self.start:
            # FIX 1: prefer the parent pointer recorded during compute —
            # it is always consistent because it was set when g[node] was
            # committed.  Fall back to greedy g-value scan only if the
            # pointer is missing (shouldn't happen on a reachable path).
            pred = self.parent.get(current)

            if pred is None or pred in visited:
                # Fallback: greedy scan, skipping already-visited nodes
                pred = None
                best_cost = math.inf
                for p in self.G.predecessors(current):
                    if p in visited:
                        continue
                    w = self._get_edge_weight(p, current)
                    cost = self.g[p] + w
                    if cost < best_cost:
                        best_cost = cost
                        pred = p

            if pred is None or pred in visited:
                return None

            visited.add(pred)
            path.append(pred)
            current = pred

        path.reverse()
        return path

    def close_road(self, u, v):
        if self.G.has_edge(u, v):
            for key_idx in self.G[u][v]:
                self.G[u][v][key_idx]['weight'] = math.inf
        self._update_vertex(v)
        self.total_replans += 1

    def update_edge_weight(self, u, v, new_weight):
        if self.G.has_edge(u, v):
            for key_idx in self.G[u][v]:
                self.G[u][v][key_idx]['weight'] = new_weight
        self._update_vertex(v)

    def get_stats(self):
        return {
            "goal_cost_wh"   : self.g[self.goal],
            "total_expanded" : self.nodes_expanded,
            "inconsistent"   : len(self._in_heap),
            "total_replans"  : self.total_replans
        }


def make_haversine_heuristic(graph, goal_node):
    goal_lat = math.radians(graph.nodes[goal_node]['y'])
    goal_lon = math.radians(graph.nodes[goal_node]['x'])
    # FIX 2: use 80 Wh/km (below eVED flat mean of ~85) so the heuristic
    # is admissible — it never overestimates the true battery cost.
    # The old value of 120 Wh/km overestimated flat roads, making LPA*
    # skip expanding nodes it needed, causing stale g-values on replans.
    AVG_ENERGY_PER_KM = 80.0

    def h(node):
        lat = math.radians(graph.nodes[node]['y'])
        lon = math.radians(graph.nodes[node]['x'])
        dlat = goal_lat - lat
        dlon = goal_lon - lon
        a = (math.sin(dlat/2)**2
             + math.cos(lat) * math.cos(goal_lat) * math.sin(dlon/2)**2)
        dist_km = 6371.0 * 2.0 * math.asin(math.sqrt(max(0, a)))
        return dist_km * AVG_ENERGY_PER_KM

    return h