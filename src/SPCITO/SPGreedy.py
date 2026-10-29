"""Greedy order generation with the original minimum-edge random tie rule."""
import numpy as np
from . import SPCITO


def generate_spgreedy_sequence(rng: np.random.Generator) -> list[int]:
    independent_classes = SPCITO.get_independent_classes()
    independent_set = set(independent_classes)
    dependency_free_classes = [cls for cls in SPCITO.get_dependency_free_classes() if cls not in independent_set]
    dependency_free_set = set(dependency_free_classes)
    current_state = 0
    sequence = list(independent_classes)
    for cls in independent_classes:
        current_state |= 1 << cls
    end_state = 0
    for cls in range(SPCITO.n_classes):
        if cls not in dependency_free_set:
            end_state |= 1 << cls
    while current_state != end_state:
        available_actions = [action for action in SPCITO.get_available_actions(current_state)
                             if action not in dependency_free_set]
        if not available_actions:
            raise RuntimeError('SPGreedy stopped before reaching the target state.')
        minimum_cost = float('inf')
        minimum_actions = []
        for action in available_actions:
            action_cost = SPCITO.calculate_scplx(current_state, action)
            if action_cost < minimum_cost:
                minimum_cost = action_cost
                minimum_actions = [action]
            elif action_cost == minimum_cost:
                minimum_actions.append(action)
        selected_action = int(rng.choice(minimum_actions))
        current_state |= 1 << selected_action
        sequence.append(selected_action)
    sequence.extend(dependency_free_classes)
    SPCITO.validate_sequence(sequence)
    return sequence


