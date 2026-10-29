"""SPCITO algorithms. Load one project before calling a generator.

Use ``from SPCITO import SPCITO`` to access the current matrices and scores.
Algorithm modules share this explicitly configured model; importing the package
does not load a dataset or start an experiment.
"""

from .SPCITO import configure_matrices, load_project, compute_sequence_cost

__all__ = ["configure_matrices", "load_project", "compute_sequence_cost"]
