import numpy as np
import heapq
from . import SPCITO


def dijkstra(start, end):
    """Find a shortest state path with Dijkstra, storing states as discovered."""
    # Allocate records as states are discovered instead of preallocating all states.
    dist = {start: 0.0}
    prev = {start: None}
    pq = [(0.0, start)]  # (distance, state)
    
    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            break
        if d > dist[u]:  # Skip stale priority-queue entries.
            continue
            
        available_actions = SPCITO.get_available_actions(u)
        for a in available_actions:
            v = u | (1 << a)  # Integrate class a to obtain the successor state.
            cost = SPCITO.calculate_scplx(u, a)
            new_dist = d + cost
            
            # Relax the edge if v is new or the candidate distance is smaller.
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    # Reconstruct the state path from the predecessor records.
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev.get(current, None)
    path.reverse()
    
    return dist.get(end, float('inf')), path

def generate_spdijkstra_sequence():
    """Full-order construction, including the original zero-cost preprocessing."""
    independent_classes = SPCITO.get_independent_classes()
    dependency_free_classes = [i for i in SPCITO.get_dependency_free_classes()
                               if i not in SPCITO.get_independent_classes()]
    initial_state = 0
    initial_sequence = []
    for cls in independent_classes:
        initial_state |= 1 << cls
        initial_sequence.append(cls)
    end_state = 0
    for i in range(SPCITO.n_classes):
        if i not in dependency_free_classes:
            end_state |= 1 << i
    min_cost, optimal_path = dijkstra(initial_state, end_state)
    if not np.isfinite(min_cost) or optimal_path[0] != initial_state:
        raise RuntimeError('SPDijkstra did not reach the target state.')
    test_sequence = initial_sequence[:]
    for previous, current in zip(optimal_path, optimal_path[1:]):
        test_sequence.append((current ^ previous).bit_length() - 1)
    test_sequence.extend(dependency_free_classes)
    SPCITO.validate_sequence(test_sequence)
    return test_sequence


