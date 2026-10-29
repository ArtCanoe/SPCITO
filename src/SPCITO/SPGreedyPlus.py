"""Greedy construction followed by one SW-SPDijkstra pass."""

import numpy as np

from .SPGreedy import generate_spgreedy_sequence
from .SW_SPDijkstra import optimize_sequence


def generate_spgreedyplus_sequence(rng: np.random.Generator, window_size: int = 10) -> list[int]:
    """Return the optimized order without independent score evaluation.

    The experiment worker calls the two stages separately to record their
    continuous timing intervals; this convenience function follows the same
    construction and optimization steps.
    """
    initial = generate_spgreedy_sequence(rng)
    return optimize_sequence(window_size, initial)
