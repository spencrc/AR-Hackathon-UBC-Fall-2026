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
from typing import Dict, Optional, Set, Tuple
from ar_hackathon.models.graph_state import GraphState

_seen_pod_sources: Set[str] = set()
_last_spawn_source: Optional[int] = None


def _dijkstra(state: GraphState, start: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    dist = {node.id: float('inf') for node in state.nodes}
    predecessor: Dict[int, int] = {node.id: None for node in state.nodes}
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in state.neighbors(u):
            edge = state.get_edge(u, v)
            if edge is None:
                continue
            nd = d + edge.weight
            if nd < dist[v]:
                dist[v] = nd
                predecessor[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, predecessor


def _first_hop(predecessor: Dict[int, int], start: int, target: int) -> Optional[int]:
    if target == start:
        return None
    if predecessor.get(target) is None:
        return None
    node = target
    while predecessor[node] is not None and predecessor[node] != start:
        node = predecessor[node]
    return node if predecessor[node] == start else None


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


def _idle_target(state: GraphState, dist: Dict[int, float]) -> Optional[int]:
    if _last_spawn_source is not None and dist.get(_last_spawn_source, float('inf')) != float('inf'):
        return _last_spawn_source
    best = None
    best_d = float('inf')
    for node in state.nodes:
        if node.node_type != "storage":
            continue
        d = dist.get(node.id, float('inf'))
        if d < best_d:
            best_d = d
            best = node.id
    return best


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    Returns the ID of an adjacent node to move to, or None to wait.
    """
    global _last_spawn_source

    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    for pod in state.active_pods:
        if (pod.carried_by is None and pod.current_node is not None
                and pod.id not in _seen_pod_sources):
            _seen_pod_sources.add(pod.id)
            _last_spawn_source = pod.current_node

    dist, predecessor = _dijkstra(state, unit.current_node)

    target = None
    if unit.carrying:
        pod = state.get_pod(unit.carrying[0])
        target = pod.destination_station if pod else None
    else:
        waiting = [p for p in state.active_pods
                   if p.carried_by is None and p.current_node is not None]
        if waiting:
            best = None
            for p in sorted(waiting, key=lambda p: (p.entry_time, p.id)):
                d = dist.get(p.current_node, float('inf'))
                if d == float('inf'):
                    continue
                if best is None or d < best:
                    best = d
                    target = p.current_node
        else:
            target = _idle_target(state, dist)

    if target is None or target == unit.current_node:
        return None
    if dist.get(target, float('inf')) == float('inf'):
        return None

    next_node = _first_hop(predecessor, unit.current_node, target)
    if next_node is None or not _move_is_safe(state, unit, next_node):
        return None
    return next_node
