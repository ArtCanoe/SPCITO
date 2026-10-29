"""Run with: python -B -m unittest discover -s experiments/tests -v"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from SPCITO import SPCITO, SPDijkstra, SPGreedy, SW_SPDijkstra as Dijkstra_SW
from experiments.measurement import worker, peak_memory
from experiments import run_generation as runner

TEMP_ROOT = ROOT / 'experiments' / 'tests' / '.tmp'
TEMP_ROOT.mkdir(exist_ok=True)
tempfile.tempdir = str(TEMP_ROOT)


def make_project(parent):
    project = Path(parent) / 'Tiny'
    project.mkdir()
    # Lowercase names intentionally exercise Linux-compatible path lookup.
    np.savetxt(project / 'stubcomplexitymatrix.csv',
               [[0, 2, 0], [1, 0, 3], [4, 0, 0]], delimiter=',')
    np.savetxt(project / 'dependencymatrix.csv',
               [[0, 1, 1], [1, 0, 1], [1, 0, 0]], delimiter=',', fmt='%d')
    return project


def make_request(project, algorithm='SPGreedy+', warmup=0):
    return {'project_dir': str(project), 'algorithm': algorithm, 'seed': 117,
            'window_size': 2, 'warmup_runs': warmup, 'max_exact_classes': 24,
            'input_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest()
                             for key, path in SPCITO.input_paths(project).items()}}


class CoreTests(unittest.TestCase):
    def setUp(self):
        SPCITO.configure_matrices([[0, 2, 0], [1, 0, 3], [4, 0, 0]],
                                 [[0, 1, 1], [1, 0, 1], [1, 0, 0]])

    def test_metrics_count_zero_cost_dependency(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            self.assertEqual(SPCITO.compute_sequence_cost([0, 1, 2]), (5.0, 3))
        self.assertEqual(capture.getvalue(), '')

    def test_verbose_metrics_are_opt_in(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            SPCITO.compute_sequence_cost([0, 1, 2], verbose=True)
        self.assertEqual(len(capture.getvalue().splitlines()), 3)

    def test_reject_invalid_permutations(self):
        for order in ([0, 1], [0, 1, 1], [0, 1, 3], [0, 1, 2.0], [False, 1, 2]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                SPCITO.compute_sequence_cost(order)

    def test_reject_invalid_costs(self):
        for costs in ([[1]], [[0, -1], [1, 0]], [[0, np.nan], [1, 0]], [[0, np.inf], [1, 0]], [[0, 1]]):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                SPCITO.configure_matrices(costs, np.zeros(np.asarray(costs).shape))

    def test_reject_invalid_dependencies(self):
        for deps in ([[0]], [[0, 2], [1, 0]], [[1, 0], [0, 0]], [[0, np.nan], [1, 0]]):
            with self.subTest(deps=deps), self.assertRaises(ValueError):
                SPCITO.configure_matrices([[0, 1], [1, 0]], deps)

    def test_sw_does_not_mutate_input_or_global_matrix(self):
        initial = [0, 1, 2]
        matrix = SPCITO.cost_matrix
        before = matrix.copy()
        result = Dijkstra_SW.optimize_sequence(2, initial)
        self.assertEqual(initial, [0, 1, 2])
        self.assertIs(SPCITO.cost_matrix, matrix)
        np.testing.assert_array_equal(matrix, before)
        self.assertLessEqual(SPCITO.compute_sequence_cost(result)[0], 5)
        self.assertEqual(Dijkstra_SW.Dijkstra_sliding_window(2, initial),
                         (result, *SPCITO.compute_sequence_cost(result)))

    def test_window_sizes(self):
        for size in (0, -1, 2.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                Dijkstra_SW.optimize_sequence(size, [0, 1, 2])
        self.assertEqual(Dijkstra_SW.optimize_sequence(1, [0, 1, 2]), [0, 1, 2])
        self.assertEqual(Dijkstra_SW.optimize_sequence(30, [0, 1, 2]),
                         Dijkstra_SW.optimize_sequence(3, [0, 1, 2]))

    def test_single_class_and_all_zero_costs(self):
        for n in (1, 3):
            SPCITO.configure_matrices(np.zeros((n, n)), np.ones((n, n)) - np.eye(n))
            order = list(range(n))
            self.assertEqual(SPDijkstra.generate_spdijkstra_sequence(), order)
            self.assertEqual(SPGreedy.generate_spgreedy_sequence(np.random.default_rng(117)), order)
            result = Dijkstra_SW.optimize_sequence(10, order)
            SPCITO.validate_sequence(result)
            self.assertEqual(SPCITO.compute_sequence_cost(result), (0.0, n * (n - 1) // 2))

    def test_exact_solution_against_brute_force(self):
        for seed in (0, 11, 117):
            costs = np.random.default_rng(seed).integers(0, 6, size=(6, 6)).astype(float)
            np.fill_diagonal(costs, 0)
            if seed == 11:
                costs[0] = 0
                costs[:, 1] = 0
            SPCITO.configure_matrices(costs, costs != 0)
            optimum = min(SPCITO.compute_sequence_cost(list(p))[0] for p in itertools.permutations(range(6)))
            with self.subTest(seed=seed):
                self.assertEqual(SPCITO.compute_sequence_cost(SPDijkstra.generate_spdijkstra_sequence())[0], optimum)
                self.assertEqual(SPCITO.compute_sequence_cost(Dijkstra_SW.optimize_sequence(6, list(range(6))))[0], optimum)

    def test_spgreedyplus_timing_is_continuous(self):
        events = []
        timestamps = iter((100, 350, 700))
        def clock():
            events.append('clock')
            return next(timestamps)
        def greedy(rng):
            events.append('greedy')
            return [0, 1, 2]
        def optimize(size, initial):
            events.append('sw')
            return [2, 1, 0]
        with mock.patch.object(worker, 'generate_spgreedy_sequence', side_effect=greedy), \
             mock.patch.object(worker, 'optimize_sequence', side_effect=optimize), \
             mock.patch.object(SPCITO, 'compute_sequence_cost', side_effect=AssertionError('evaluation inside timer')):
            measured = worker.execute_algorithm('SPGreedy+', 117, 2, clock=clock)
        self.assertEqual(events, ['clock', 'greedy', 'clock', 'sw', 'clock'])
        self.assertEqual((measured['runtime_ns'], measured['greedy_ns'], measured['sw_ns']), (600, 250, 350))

    def test_all_timed_interfaces_exclude_metric_evaluation_and_output(self):
        for algorithm in runner.ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                capture = io.StringIO()
                with mock.patch.object(SPCITO, 'compute_sequence_cost', side_effect=AssertionError('evaluation inside timer')), \
                     contextlib.redirect_stdout(capture):
                    result = worker.execute_algorithm(algorithm, 117, 2, [0, 1, 2])
                self.assertEqual(capture.getvalue(), '')
                self.assertGreater(result['runtime_ns'], 0)
                self.assertIn('Ocplx', worker.evaluate_result(result))

    def test_sw_never_generates_its_initial_order(self):
        with mock.patch.object(worker, 'generate_spgreedy_sequence', side_effect=AssertionError('initial generation')):
            worker.execute_algorithm('SW-SPDijkstra', 117, 2, [0, 1, 2])
        with self.assertRaises(ValueError):
            worker.execute_algorithm('SW-SPDijkstra', 117, 2)

    def test_sstubs_increase_is_not_misreported_as_algorithm_failure(self):
        result = {'order': [0, 1, 2], 'initial': [2, 1, 0]}
        with mock.patch.object(SPCITO, 'compute_sequence_cost', side_effect=[(1, 4), (2, 3)]):
            evaluated = worker.evaluate_result(result)
        self.assertEqual(evaluated['SStubs'], 4)
        with mock.patch.object(SPCITO, 'compute_sequence_cost', side_effect=[(3, 2), (2, 3)]), self.assertRaises(RuntimeError):
            worker.evaluate_result(result)


class MemoryTests(unittest.TestCase):
    def test_linux_units(self):
        self.assertEqual(peak_memory.parse_linux_peak_rss('Name:\tpython\nVmHWM:\t2048 kB\n'), 2.0)

    def test_linux_invalid_or_missing_peak(self):
        for value in ('VmRSS: 1024 kB', 'VmHWM: 0 kB', 'VmHWM: 10 MB', 'VmHWM: nope kB'):
            with self.subTest(value=value), self.assertRaises((RuntimeError, ValueError)):
                peak_memory.parse_linux_peak_rss(value)

    def test_native_peak(self):
        peak, source = peak_memory.read_peak_rss()
        self.assertGreater(peak, 0)
        self.assertIn(source, ('LinuxVmHWM', 'WindowsPeakWorkingSetSize'))

    def test_unsupported_os_fails_explicitly(self):
        with mock.patch.object(peak_memory.sys, 'platform', 'unsupported'), self.assertRaises(RuntimeError):
            peak_memory.read_peak_rss()


class ExperimentTests(unittest.TestCase):
    def test_batch_defaults_and_command_line_overrides(self):
        args = runner.parse_args([], default_runs=7, default_window_size=4)
        self.assertEqual((args.runs, args.window_size), (7, 4))
        args = runner.parse_args(['--runs', '3', '--window-size', '2'], default_runs=7, default_window_size=4)
        self.assertEqual((args.runs, args.window_size), (3, 2))

    def test_imports_have_no_dataset_load_or_experiment_side_effect(self):
        result = subprocess.run([sys.executable, '-B', '-c',
                                 "import sys; sys.path.insert(0, 'src'); "
                                 "from SPCITO import SPCITO, SPDijkstra, SW_SPDijkstra, SPGreedy, SPGreedyPlus; "
                                 "assert SPCITO.n_classes == 0"],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')

    def test_seeds_are_unique_and_replayable(self):
        a = runner.generate_unique_seeds(30)
        b = runner.generate_unique_seeds(30)
        self.assertEqual(len(set(a)), 30)
        self.assertNotEqual(a, b)
        self.assertTrue(all(0 <= seed < 2**64 for seed in a))
        self.assertEqual(runner.generate_unique_seeds(30, 117), runner.generate_unique_seeds(30, 117))
        with self.assertRaises(ValueError):
            runner.generate_unique_seeds(0)

    def test_fresh_process_and_warmup_rng_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            project = make_project(folder)
            cold = runner.run_worker(make_request(project), 30)
            warm = runner.run_worker(make_request(project, warmup=1), 30)
        self.assertNotEqual(cold['WorkerPid'], os.getpid())
        self.assertNotEqual(warm['WorkerPid'], os.getpid())
        self.assertNotEqual(cold['WorkerPid'], warm['WorkerPid'])
        for key in ('CITO', 'InitialCITO', 'Ocplx', 'SStubs'):
            self.assertEqual(cold[key], warm[key])
        for result in (cold, warm):
            self.assertEqual(result['RuntimeNs'], result['SPGreedyRuntimeNs'] + result['SWRuntimeNs'])
            self.assertEqual(result['RuntimeMs'], result['RuntimeNs'] / 1e6)
            self.assertGreater(result['PeakRSSMiB'], 0)

    def test_other_worker_algorithms(self):
        with tempfile.TemporaryDirectory() as folder:
            project = make_project(folder)
            for algorithm in ('SPGreedy', 'SPDijkstra', 'SW-SPDijkstra'):
                request = make_request(project, algorithm)
                request['initial_order'] = [0, 1, 2] if algorithm == 'SW-SPDijkstra' else None
                with self.subTest(algorithm=algorithm):
                    result = runner.run_worker(request, 30)
                    self.assertEqual(sorted(result['CITO']), [0, 1, 2])
                    self.assertGreater(result['PeakRSSMiB'], 0)

    def test_changed_inputs_and_exact_guard_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            project = make_project(folder)
            request = make_request(project, 'SPDijkstra')
            request['max_exact_classes'] = 2
            with mock.patch.object(worker, 'execute_algorithm', side_effect=AssertionError('must not run')), self.assertRaises(ValueError):
                worker.run(request)
            request['input_sha256'] = {}
            with self.assertRaises(RuntimeError):
                worker.run(request)

    def test_timeout_is_recorded_as_an_error(self):
        with mock.patch.object(runner.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 1)), self.assertRaisesRegex(RuntimeError, 'terminated'):
            runner.run_worker({}, 1)

    def test_csv_and_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            project = make_project(folder)
            destination = Path(folder) / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                status = runner.main(['--project-dir', str(project), '--runs', '2', '--window-size', '2',
                                      '--master-seed', '117', '--warmup-runs', '0', '--output-dir', str(destination)])
            self.assertEqual(status, 0)
            batch, = destination.iterdir()
            config = json.loads((batch / 'configuration.json').read_text(encoding='utf-8'))
            with (batch / 'runs.csv').open(encoding='utf-8-sig', newline='') as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames[:8], ['Run', 'Project', 'Algorithm', 'CITO', 'Ocplx', 'SStubs', 'RuntimeMs', 'PeakRSSMiB'])
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertTrue(config['completed'])
            self.assertEqual([int(row['Seed']) for row in rows], runner.generate_unique_seeds(2, 117))
            for row in rows:
                self.assertEqual(row['Status'], 'success')
                self.assertEqual(sorted(json.loads(row['CITO'])), [0, 1, 2])
                self.assertEqual(int(row['RuntimeNs']), int(row['SPGreedyRuntimeNs']) + int(row['SWRuntimeNs']))
            with (batch / 'summary.csv').open(encoding='utf-8-sig', newline='') as file:
                summary, = csv.DictReader(file)
            self.assertEqual(summary['SuccessfulRuns'], '2')
            expected_mean = sum(float(row['RuntimeMs']) for row in rows) / 2
            self.assertAlmostEqual(float(summary['RuntimeMs_mean']), expected_mean)

    def test_failure_output_is_not_zero_or_success(self):
        with tempfile.TemporaryDirectory() as folder:
            project = make_project(folder)
            destination = Path(folder) / 'out'
            with mock.patch.object(runner, 'run_worker', side_effect=RuntimeError('synthetic worker failure')), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = runner.main(['--project-dir', str(project), '--runs', '2', '--output-dir', str(destination)])
            self.assertEqual(status, 1)
            batch, = destination.iterdir()
            config = json.loads((batch / 'configuration.json').read_text(encoding='utf-8'))
            with (batch / 'runs.csv').open(encoding='utf-8-sig', newline='') as file:
                row, = csv.DictReader(file)
            self.assertFalse(config['completed'])
            self.assertEqual(row['Status'], 'error')
            for key in ('CITO', 'Ocplx', 'SStubs', 'RuntimeMs', 'PeakRSSMiB'):
                self.assertEqual(row[key], '')
            self.assertTrue(row['Seed'])


class OriginalResultRegressionTests(unittest.TestCase):
    def test_all_8_ding_original_orders_and_metrics(self):
        fixture = json.loads((Path(__file__).parent / 'regression_before_measurement.json').read_text(encoding='utf-8'))
        # Retain the historical fixture unchanged; this release bundles Ding only.
        records = [record for record in fixture['records'] if record['dataset'] == 'Ding']
        self.assertEqual(len(records), 8)
        self.assertEqual({record['project'] for record in records},
                         {'ANT', 'Email-spl', 'DNS', 'jboss', 'xml_security', 'jmeter', 'JHotDraw', 'xstream_spl'})
        self.assertEqual(np.__version__, fixture['numpy_version'], 'Use the recorded NumPy version for exact seeded-order regression.')
        for record in records:
            with self.subTest(dataset=record['dataset'], project=record['project']):
                SPCITO.load_project(ROOT / 'datasets' / 'Ding' / record['project'])
                initial = SPGreedy.generate_spgreedy_sequence(np.random.default_rng(record['seed']))
                final = Dijkstra_SW.optimize_sequence(record['window'], initial)
                self.assertEqual(initial, record['initial'])
                self.assertEqual(final, record['final'])
                self.assertEqual(SPCITO.compute_sequence_cost(initial), (record['initial_cost'], record['initial_stubs']))
                self.assertEqual(SPCITO.compute_sequence_cost(final), (record['cost'], record['stubs']))


if __name__ == '__main__':
    unittest.main()
