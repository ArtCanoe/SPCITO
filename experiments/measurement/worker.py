"""One isolated repetition; no input loading, logging, or metric evaluation in timers."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import numpy as np

from SPCITO import SPCITO
from SPCITO.SW_SPDijkstra import optimize_sequence
from SPCITO.SPDijkstra import generate_spdijkstra_sequence
from SPCITO.SPGreedy import generate_spgreedy_sequence
from experiments.measurement.peak_memory import read_peak_rss


def execute_algorithm(algorithm, seed, window_size, initial_order=None, *, clock=None):
    """Return orders and continuous timing intervals; evaluation happens later."""
    clock = time.perf_counter_ns if clock is None else clock
    rng = np.random.default_rng(seed)
    if algorithm == 'SPGreedy+':
        t0 = clock()
        initial = generate_spgreedy_sequence(rng)
        t1 = clock()
        order = optimize_sequence(window_size, initial)
        t2 = clock()
        return {'order': order, 'initial': initial, 'runtime_ns': t2 - t0,
                'greedy_ns': t1 - t0, 'sw_ns': t2 - t1}
    if algorithm == 'SPGreedy':
        t0 = clock()
        order = generate_spgreedy_sequence(rng)
        t1 = clock()
        return {'order': order, 'initial': None, 'runtime_ns': t1 - t0,
                'greedy_ns': t1 - t0, 'sw_ns': None}
    if algorithm == 'SPDijkstra':
        t0 = clock()
        order = generate_spdijkstra_sequence()
        t1 = clock()
        return {'order': order, 'initial': None, 'runtime_ns': t1 - t0,
                'greedy_ns': None, 'sw_ns': None}
    if algorithm == 'SW-SPDijkstra':
        if initial_order is None:
            raise ValueError('SW-SPDijkstra requires a supplied initial CITO.')
        SPCITO.validate_sequence(initial_order)
        t0 = clock()
        order = optimize_sequence(window_size, initial_order)
        t1 = clock()
        return {'order': order, 'initial': initial_order, 'runtime_ns': t1 - t0,
                'greedy_ns': None, 'sw_ns': t1 - t0}
    raise ValueError(f'Unknown algorithm: {algorithm}')


def evaluate_result(result):
    cost, stubs = SPCITO.compute_sequence_cost(result['order'])
    evaluated = {'Ocplx': cost, 'SStubs': stubs}
    if result['initial'] is not None:
        old_cost, old_stubs = SPCITO.compute_sequence_cost(result['initial'])
        if cost > old_cost + 1e-10 * max(1, abs(old_cost)):
            raise RuntimeError('Sliding-window optimization increased OCplx.')
        evaluated.update({'InitialOcplx': old_cost, 'InitialSStubs': old_stubs})
    return evaluated


def run(request):
    paths = SPCITO.input_paths(request['project_dir'])
    actual = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    if actual != request['input_sha256']:
        raise RuntimeError('Input files changed after the batch was planned.')
    read_peak_rss()  # Probe outside algorithm time; fail explicitly if unsupported.
    SPCITO.load_project(request['project_dir'])
    # A safety guard, not a pruning rule or an approximation of the exact method.
    if request['algorithm'] == 'SPDijkstra':
        remaining = SPCITO.n_classes - len(SPCITO.get_independent_classes())
        if remaining > request['max_exact_classes']:
            raise ValueError(f'SPDijkstra has {remaining} initially unset bits; '
                             f'limit is {request["max_exact_classes"]}. '
                             'Increase --max-exact-classes explicitly if resources permit.')
    for _ in range(request['warmup_runs']):
        execute_algorithm(request['algorithm'], request['seed'], request['window_size'], request.get('initial_order'))
    result = execute_algorithm(request['algorithm'], request['seed'], request['window_size'], request.get('initial_order'))
    if {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()} != actual:
        raise RuntimeError('Input files changed during the repetition.')
    metrics = evaluate_result(result)
    if result['runtime_ns'] <= 0:
        raise RuntimeError('Nonpositive runtime measurement.')
    payload = {
        'status': 'success', 'Algorithm': request['algorithm'], 'Seed': request['seed'],
        'NClasses': SPCITO.n_classes, 'CITO': result['order'], **metrics,
        'RuntimeNs': result['runtime_ns'], 'RuntimeMs': result['runtime_ns'] / 1e6,
        'SPGreedyRuntimeNs': result['greedy_ns'],
        'SWRuntimeNs': result['sw_ns'], 'InitialCITO': result['initial'],
        'WorkerPid': os.getpid(), 'WarmupRuns': request['warmup_runs'],
        'WindowSize': request['window_size'] if request['algorithm'] in ('SPGreedy+', 'SW-SPDijkstra') else None,
    }
    prepared = json.dumps(payload, ensure_ascii=True)
    payload['PeakRSSMiB'], payload['PeakRSSSource'] = read_peak_rss()
    del prepared
    return payload


if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin)), ensure_ascii=True), flush=True)
