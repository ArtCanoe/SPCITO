"""SPCITO cost operations. Input loading is explicit and outside algorithm time."""
from pathlib import Path

import numpy as np

DEFAULT_PROJECT = Path(__file__).resolve().parents[2] / 'datasets' / 'Ding' / 'ANT'
cost_matrix = np.empty((0, 0), dtype=float)
dependency_matrix = np.empty((0, 0), dtype=bool)
n_classes = 0


def input_paths(project_dir):
    """Accept the original filename capitalization on Linux as well as Windows."""
    project = Path(project_dir).resolve()
    matches = {}
    for name in ('StubComplexityMatrix.csv', 'DependencyMatrix.csv'):
        found = [p for p in project.iterdir() if p.is_file() and p.name.lower() == name.lower()]
        if len(found) != 1:
            raise ValueError(f'Expected exactly 1 {name} in {project}; found {len(found)}.')
        matches[name] = found[0]
    return matches


def configure_matrices(costs, dependencies):
    global cost_matrix, dependency_matrix, n_classes
    costs = np.asarray(costs, dtype=float)
    dependencies = np.asarray(dependencies, dtype=float)
    if costs.ndim != 2 or not len(costs) or costs.shape[0] != costs.shape[1]:
        raise ValueError('Cost matrix must be nonempty and square.')
    if dependencies.shape != costs.shape:
        raise ValueError('Dependency and cost matrix shapes must match.')
    if not np.all(np.isfinite(costs)) or np.any(costs < 0) or np.any(np.diag(costs) != 0):
        raise ValueError('Costs must be finite and nonnegative with a zero diagonal.')
    if not np.all(np.isfinite(dependencies)) or not np.all((dependencies == 0) | (dependencies == 1)):
        raise ValueError('Dependency matrix entries must be 0 or 1.')
    if np.any(np.diag(dependencies) != 0):
        raise ValueError('Dependency matrix must have a zero diagonal.')
    cost_matrix = costs.copy()
    dependency_matrix = dependencies.astype(bool)
    n_classes = len(cost_matrix)


def load_project(project_dir=DEFAULT_PROJECT):
    paths = input_paths(project_dir)
    costs = np.loadtxt(paths['StubComplexityMatrix.csv'], delimiter=',', encoding='utf-8-sig', ndmin=2)
    deps = np.genfromtxt(paths['DependencyMatrix.csv'], delimiter=',', encoding='utf-8-sig',
                         dtype=float, autostrip=True, missing_values='', filling_values=0, ndmin=2)
    configure_matrices(costs, deps)
    return paths


def require_loaded():
    if n_classes < 1:
        raise RuntimeError('Call SPCITO.load_project(project_dir) before generating an order.')


def validate_sequence(sequence):
    require_loaded()
    if len(sequence) != n_classes or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in sequence) or set(sequence) != set(range(n_classes)):
        raise ValueError('The CITO must be a complete zero-based class permutation.')

def state_to_binary(state,):
    """Return the binary state string with the most significant bit first."""
    return bin(state)[2:].zfill(n_classes)

def get_available_actions(state):
    """Return the indices of classes that have not yet been integrated."""
    binary_state = state_to_binary(state)
    return [i for i in range(n_classes) if binary_state[n_classes - 1 - i] == '0']

def calculate_scplx(state, action):
    """Return the stubbing cost of integrating action in the current state."""
    binary_state = state_to_binary(state)
    scplx = 0.0
    # Sum costs to classes that have not yet been integrated.
    for j in range(n_classes):
        if binary_state[n_classes - 1 - j] == '0':  # Class j is not yet integrated.
            scplx += cost_matrix[action][j]
    return scplx

def get_independent_classes():
    """Return classes whose outgoing stubbing costs are all zero (zero rows)."""
    require_loaded()
    return [i for i in range(n_classes) if np.all(cost_matrix[i] == 0)]

def get_dependency_free_classes():
    """Return classes whose incoming stubbing costs are all zero (zero columns)."""
    return [i for i in range(n_classes) if np.all(cost_matrix[:, i] == 0)]

def compute_sequence_cost(test_sequence, *, verbose=False):
    """Compute total stubbing cost and the number of specific stubs.

    Args:
        test_sequence (list): A complete permutation of zero-based class indices.
        verbose (bool): Print a diagnostic for each required stub.

    Returns:
        tuple: Total stubbing cost (OCplx) and specific stub count (SStubs).
    """
    validate_sequence(test_sequence)
    
    tested = [False] * n_classes  # Track which classes have been integrated.
    total_ACplx = 0
    total_MCplx = 0
    total_cost = 0.0
    total_stubs = 0  # Total number of specific stubs.

    for action in test_sequence:
        # Compute the current class's cost over providers not yet integrated.
        scplx = 0.0
        stub_count = 0  # Initialize the per-step stub counter.
        for j in range(n_classes):
            if not tested[j] and j != action:  # Exclude integrated classes and the current class.
                cost = cost_matrix[action][j]
                if cost != 0:  # Accumulate costs; count dependencies separately below.
                    scplx += cost
                
                if dependency_matrix[action][j]:
                    total_stubs += 1
                    if verbose:
                        print(f"测试类 {action} 依赖于类 {j}，需要测试桩scplx[{action}][{j}]={cost}。")
        
        total_cost += scplx
        total_stubs += stub_count  # Add the per-step stub counter.
        tested[action] = True  # Mark the current class as integrated.

    return float(total_cost), total_stubs
