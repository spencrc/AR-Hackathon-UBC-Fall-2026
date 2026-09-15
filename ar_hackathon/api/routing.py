"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name:
Email address:
*******************
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple
from ar_hackathon.models.graph_state import GraphState

_INF = float('inf')

_plan_ts: int = -1
_last_t: int = -1
_last_delivered: int = -1
_plan: Dict[int, Optional[int]] = {}
_prev_pickup: Dict[int, int] = {}
_spawn_counts: Dict[int, int] = {}
_seen_pods: Set[str] = set()


def _reset_module_state() -> None:
    global _plan_ts, _last_t, _last_delivered
    _plan_ts = -1
    _last_t = -1
    _last_delivered = -1
    _plan.clear()
    _prev_pickup.clear()
    _spawn_counts.clear()
    _seen_pods.clear()


def _build_adj(state: GraphState) -> Dict[int, list]:
    adj: Dict[int, list] = {node.id: [] for node in state.nodes}
    for edge in state.edges:
        adj.setdefault(edge.from_node, []).append((edge.to_node, edge))
        if edge.bidirectional:
            adj.setdefault(edge.to_node, []).append((edge.from_node, edge))
    return adj


def _edge_free_time(state: GraphState, edge) -> int:
    best = None
    for unit in state.drive_units:
        if unit.in_transit and edge.connects(unit.current_node, unit.transit_destination):
            r = max(1, unit.transit_remaining_time)
            if best is None or r < best:
                best = r
    return 1 if best is None else best


def _dijkstra(state: GraphState, adj: Dict[int, list], start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    dist = {nid: _INF for nid in adj}
    pred: Dict[int, Optional[int]] = {nid: None for nid in adj}
    dist[start] = 0
    heap = [(0, start)]
    nodes = {n.id: n for n in state.nodes}
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, edge in adj.get(u, ()):
            w = edge.weight
            if edge.capacity is not None and state.edge_occupancy(u, v) >= edge.capacity:
                w += _edge_free_time(state, edge)
            node = nodes.get(v)
            if node is not None and node.capacity is not None and state.node_occupancy(v) >= node.capacity:
                w += 2
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, pred


def _first_hop(pred: Dict[int, Optional[int]], start: int, target: int) -> Optional[int]:
    if target not in pred or pred[target] is None:
        return None
    node = target
    while pred[node] != start:
        node = pred[node]
        if node is None:
            return None
    return node


def _move_is_safe(state: GraphState, unit, next_node: int) -> bool:
    edge = state.get_edge(unit.current_node, next_node)
    if edge is None:
        return False
    if edge.capacity is not None and state.edge_occupancy(unit.current_node, next_node) >= edge.capacity:
        return False
    node = state.get_node(next_node)
    if node is not None and node.capacity is not None and state.node_occupancy(next_node) >= node.capacity:
        return False
    return True


def _delivery_target(state: GraphState, unit, dist_map: Dict[int, float]) -> Optional[int]:
    counts: Dict[int, int] = {}
    for pid in unit.carrying:
        pod = state.get_pod(pid)
        if pod is not None:
            counts[pod.destination_station] = counts.get(pod.destination_station, 0) + 1
    best_key = None
    best_station = None
    for station in counts:
        d = dist_map.get(station, _INF)
        if d == _INF:
            continue
        key = (d, -counts[station])
        if best_key is None or key < best_key:
            best_key = key
            best_station = station
    return best_station


def _compute_plan(state: GraphState) -> None:
    _plan.clear()
    adj = _build_adj(state)

    for pod in state.active_pods:
        if (pod.carried_by is None and pod.current_node is not None
                and pod.id not in _seen_pods):
            _seen_pods.add(pod.id)
            _spawn_counts[pod.current_node] = _spawn_counts.get(pod.current_node, 0) + 1

    units = {u.id: u for u in state.drive_units}
    dists: Dict[int, Dict[int, float]] = {}
    for uid, unit in units.items():
        dists[uid] = _dijkstra(state, adj, unit.current_node)[0]

    waiting = [p for p in state.active_pods
               if p.carried_by is None and p.current_node is not None]
    claimed: Set[str] = set()

    free_units: List = []
    for unit in sorted(state.drive_units, key=lambda u: u.id):
        if unit.carrying:
            _plan[unit.id] = _delivery_target(state, unit, dists[unit.id])
        else:
            free_units.append(unit)

    clusters: Dict[int, List] = {}
    for pod in waiting:
        clusters.setdefault(pod.current_node, []).append(pod)
    for pods in clusters.values():
        pods.sort(key=lambda p: (p.entry_time, p.id))

    pairs = []
    for unit in free_units:
        for node, pods in clusters.items():
            d = dists[unit.id].get(node, _INF)
            if d == _INF or not pods:
                continue
            grabs = min(len(pods), unit.capacity)
            cost = d / grabs
            if _prev_pickup.get(unit.id) == node:
                cost -= 0.5
            pairs.append((cost, -grabs, unit.id, node))
    pairs.sort()
    for _, _, uid, node in pairs:
        if _plan.get(uid) is not None:
            continue
        avail = [p for p in clusters[node] if p.id not in claimed]
        if not avail:
            continue
        take = min(len(avail), units[uid].capacity)
        for pod in avail[:take]:
            claimed.add(pod.id)
        _plan[uid] = node
        _prev_pickup[uid] = node
    for unit in free_units:
        if _plan.get(unit.id) is None:
            _prev_pickup.pop(unit.id, None)

    for unit in sorted(state.drive_units, key=lambda u: u.id):
        spare = unit.capacity - len(unit.carrying)
        if not unit.carrying or spare <= 0:
            continue
        dest = _plan.get(unit.id)
        if dest is None:
            continue
        base = dists[unit.id].get(dest, _INF)
        if base == _INF:
            continue
        best_pod = None
        best_key = None
        src_cache: Dict[int, Dict[int, float]] = {}
        for pod in waiting:
            if pod.id in claimed or pod.destination_station != dest:
                continue
            d_to_src = dists[unit.id].get(pod.current_node, _INF)
            if d_to_src == _INF or d_to_src > base:
                continue
            if pod.current_node not in src_cache:
                src_cache[pod.current_node] = _dijkstra(state, adj, pod.current_node)[0]
            extra = d_to_src + src_cache[pod.current_node].get(dest, _INF) - base
            if extra == _INF or extra > 2:
                continue
            key = (extra, pod.current_node)
            if best_key is None or key < best_key:
                best_key = key
                best_pod = pod
        if best_pod is not None:
            claimed.add(best_pod.id)
            _plan[unit.id] = best_pod.current_node

    candidates: Dict[int, float] = {}
    for node in state.nodes:
        if node.node_type == "storage":
            candidates[node.id] = 1.0 + _spawn_counts.get(node.id, 0)
    for nid, count in _spawn_counts.items():
        if nid not in candidates:
            candidates[nid] = 1.0 + count

    idle = [u for u in free_units if _plan.get(u.id) is None]
    if not candidates:
        for unit in idle:
            _plan[unit.id] = None
        return

    pressure: Dict[int, int] = {nid: 0 for nid in candidates}
    for tgt in _plan.values():
        if tgt in pressure:
            pressure[tgt] += 1

    for unit in idle:
        best_node = None
        best_cost = _INF
        for nid, weight in candidates.items():
            d = dists[unit.id].get(nid, _INF)
            if d == _INF:
                continue
            cost = d + 6.0 * pressure[nid] / weight
            if any(p.current_node == nid for p in waiting):
                cost += 2.0
            here = state.get_node(unit.current_node)
            if nid == unit.current_node and here is not None and here.node_type == "station":
                cost += 50.0
            if cost < best_cost:
                best_cost = cost
                best_node = nid
        _plan[unit.id] = best_node
        if best_node is not None:
            pressure[best_node] += 1


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    Returns the ID of an adjacent node to move to, or None to wait.
    """
    global _plan_ts, _last_t, _last_delivered

    t = state.current_time_step
    delivered = len(state.delivered_pods)
    if t < _last_t or delivered < _last_delivered:
        _reset_module_state()
    _last_t = t
    _last_delivered = delivered

    if t != _plan_ts:
        _compute_plan(state)
        _plan_ts = t

    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    target = _plan.get(drive_unit_id)
    if target is None or target == unit.current_node:
        return None

    adj = _build_adj(state)
    dist, pred = _dijkstra(state, adj, unit.current_node)
    if dist.get(target, _INF) == _INF:
        return None
    hop = _first_hop(pred, unit.current_node, target)
    if hop is None or not _move_is_safe(state, unit, hop):
        return None
    return hop
