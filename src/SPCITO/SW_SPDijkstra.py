import numpy as np
import heapq
from . import SPCITO

def optimize_sequence(window_size, test_sequence):
    """Optimize an order using one left-to-right pass of exact window searches.

    Args:
        window_size (int): Maximum number of classes in each window.
        test_sequence (list): A complete permutation of zero-based class indices.

    Returns:
        list: The complete order after window optimization.

    Each window is solved exactly with Dijkstra, then the window moves one
    position to the right. Each move fixes one more position in the prefix;
    this does not guarantee a globally optimal complete order. Independent
    score evaluation is excluded from this function. The original three-value
    return interface is retained in Dijkstra_sliding_window.
    """
    
    SPCITO.validate_sequence(test_sequence)
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError('Window size must be a positive integer.')
    # Work on a copy so the input order remains unchanged.
    optimized_sequence = test_sequence[:]
    n = len(optimized_sequence)
    
    # Cap the window size at the sequence length.
    if window_size >= n:
        window_size = n
    
    # Visit each window position once, from left to right.
    for start_idx in range(n - window_size + 1):
        # Select the classes currently inside the window.
        window_classes = optimized_sequence[start_idx:start_idx + window_size]
        
        # Collect the classes outside the current window.
        remaining_classes = []
        for i, cls in enumerate(optimized_sequence):
            if i < start_idx or i >= start_idx + window_size:
                remaining_classes.append(cls)
        
        # Build the induced cost matrix for the window subproblem.
        # Only relative orderings within this window can change.
        sub_cost_matrix = np.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                sub_cost_matrix[i][j] = SPCITO.cost_matrix[window_classes[i]][window_classes[j]]
        
        # Define the state operations for exact optimization within the window.
        def sub_get_available_actions(state):
            binary_state = bin(state)[2:].zfill(window_size)[::-1]
            return [i for i in range(window_size) if binary_state[i] == '0']

        def sub_calculate_scplx(state, action):
            binary_state = bin(state)[2:].zfill(window_size)[::-1]
            scplx = 0.0
            for j in range(window_size):
                if binary_state[j] == '0':  # Local class j is not yet integrated.
                    scplx += sub_cost_matrix[action][j]
            return scplx

        def sub_dijkstra(start, end):
            dist = {start: 0.0}
            prev = {start: None}
            pq = [(0.0, start)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if u == end:
                    break
                if d > dist[u]:
                    continue
                    
                available_actions = sub_get_available_actions(u)
                for a in available_actions:
                    v = u | (1 << a)
                    cost = sub_calculate_scplx(u, a)
                    new_dist = d + cost
                    
                    if v not in dist or new_dist < dist[v]:
                        dist[v] = new_dist
                        prev[v] = u
                        heapq.heappush(pq, (new_dist, v))
            
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = prev.get(current, None)
            path.reverse()
            
            return dist.get(end, float('inf')), path

        # Define the initial and target states of the window subproblem.
        sub_start_state = 0
        sub_end_state = (1 << window_size) - 1  # All window classes are integrated.
        
        # Solve the window subproblem with Dijkstra.
        sub_min_cost, sub_optimal_path = sub_dijkstra(sub_start_state, sub_end_state)
        
        # Recover the optimal window order from its state path.
        sub_optimal_sequence = []
        for i in range(1, len(sub_optimal_path)):
            prev_state = sub_optimal_path[i-1]
            curr_state = sub_optimal_path[i]
            new_bit = curr_state ^ prev_state
            # Locate the bit corresponding to the newly integrated local class.
            action = 0
            temp = new_bit
            while temp & 1 == 0:
                action += 1
                temp >>= 1
            sub_optimal_sequence.append(window_classes[action])
        
        # Replace the current window with its optimized order.
        optimized_sequence[start_idx:start_idx + window_size] = sub_optimal_sequence[:]
    
    SPCITO.validate_sequence(optimized_sequence)
    return optimized_sequence


def Dijkstra_sliding_window(window_size, test_sequence):
    """Backward-compatible 3-value interface; benchmarks time optimize_sequence only."""
    optimized_sequence = optimize_sequence(window_size, test_sequence)
    optimized_cost, optimized_stubs = SPCITO.compute_sequence_cost(optimized_sequence)
    return optimized_sequence, optimized_cost, optimized_stubs


