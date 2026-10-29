"""Paired-input, validation, persistence, and real-worker regression tests."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from SPCITO import SPCITO
from experiments import run_postoptimization as rq3

TEMP_ROOT = ROOT / 'experiments' / 'tests' / '.tmp'
TEMP_ROOT.mkdir(exist_ok=True)
tempfile.tempdir = str(TEMP_ROOT)


class RQ3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.inputs = self.base / 'inputs'
        self.dataset = self.base / 'dataset'
        self.outputs = self.base / 'outputs'
        project = self.dataset / 'Tiny'
        project.mkdir(parents=True)
        np.savetxt(project / 'StubComplexityMatrix.csv', [[0, 2, 0], [1, 0, 3], [4, 0, 0]], delimiter=',')
        np.savetxt(project / 'DependencyMatrix.csv', [[0, 1, 1], [1, 0, 1], [1, 0, 0]], delimiter=',', fmt='%d')
        (self.inputs / 'Tiny').mkdir(parents=True)
        self.write_input('GA')
        self.write_input('SPGreedy')
        self.write_input('SPGreedy+')
        self.argv = ['--input-root', str(self.inputs), '--dataset-root', str(self.dataset),
                     '--output-dir', str(self.outputs), '--expected-runs', '2', '--window-size', '2', '--warmup-runs', '0']

    def write_input(self, algorithm, *, rows=None, suffix=None):
        suffix = suffix or ('SPGreedyPlus' if algorithm == 'SPGreedy+' else algorithm)
        filename = self.inputs / 'Tiny' / f'Tiny_{suffix}.csv'
        if rows is None:
            # Deliberate duplicate CITO from separate recorded runs: never deduplicate.
            rows = [dict(zip(rq3.FIELDS, [run, 'Tiny', algorithm, '[0,1,2]', '5.000000000000', '3', '2', '30']))
                    for run in (1, 2)]
        with filename.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rq3.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return filename

    def read_rows(self, filename):
        with filename.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def plan(self, extra=()):
        settings = rq3.settings_from_args(rq3.parse_args([*self.argv, *extra]))
        return settings, *rq3.build_plan(settings)

    def main(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return rq3.main([*self.argv, *extra])

    def fake_result(self, request, timeout):
        # Cost decreases while SStubs increases: valid for an Ocplx optimizer.
        return dict(InitialCITO=request['initial_order'][:], CITO=[2, 1, 0], InitialOcplx=5.0,
                    InitialSStubs=3, Ocplx=4.0, SStubs=4, RuntimeMs=1.234,
                    RuntimeNs=1234000, PeakRSSMiB=35.5, PeakRSSSource='test', WorkerPid=123)

    def test_plan_excludes_spgreedyplus_preserves_run_pairs(self):
        settings, groups, sources, matrices = self.plan()
        self.assertEqual(len(groups), 2)
        self.assertEqual({g['Algorithm'] for g in groups}, {'GA+', 'SPGreedy+'})
        self.assertEqual(sum(len(g['records']) for g in groups), 4)
        self.assertEqual(len(sources), 2)
        self.assertEqual(matrices['Tiny']['n_classes'], 3)
        self.assertEqual(groups[0]['records'][0]['InitialSeed'], '')
        self.assertEqual(groups[0]['records'][0]['BeforeCITO'], groups[0]['records'][1]['BeforeCITO'])

    def test_check_only_does_not_write_or_execute(self):
        with mock.patch.object(rq3, 'run_worker', side_effect=AssertionError('must not run')):
            self.assertEqual(self.main(['--check-only']), 0)
        self.assertFalse(self.outputs.exists())

    def test_run_filter_does_not_renumber_source_runs(self):
        _, groups, _, _ = self.plan(['--run-ids', '2', '--algorithms', 'SPGreedy'])
        self.assertEqual(len(groups), 1)
        self.assertEqual([row['Run'] for row in groups[0]['records']], [2])

    def test_missing_runs_duplicate_runs_and_mixed_labels_rejected(self):
        filename = self.inputs / 'Tiny' / 'Tiny_GA.csv'
        original = self.read_rows(filename)
        variants = [original[:1], [original[0], original[0]],
                    [original[0], {**original[1], 'Project': 'Elsewhere'}],
                    [original[0], {**original[1], 'Algorithm': 'PSO'}]]
        for rows in variants:
            self.write_input('GA', rows=rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.plan()

    def test_bad_metrics_and_orders_rejected_before_worker(self):
        filename = self.inputs / 'Tiny' / 'Tiny_GA.csv'
        original = self.read_rows(filename)
        for update in ({'Ocplx': '999'}, {'SStubs': '2'}, {'Ocplx': 'nan'}, {'SStubs': '-1'},
                       {'CITO': '[0,0,2]'}, {'CITO': '[0,1,true]'}, {'CITO': '[1,2,3]'}):
            self.write_input('GA', rows=[{**original[0], **update}, original[1]])
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.plan()

    def test_duplicate_method_and_unknown_filter_rejected(self):
        self.write_input('GA', suffix='duplicate')
        with self.assertRaisesRegex(ValueError, 'multiple source'):
            self.plan()
        with self.assertRaisesRegex(ValueError, 'no eligible'):
            self.plan(['--algorithms', 'SPGreedy+'])

    def test_vectorized_evaluation_counts_zero_cost_dependencies(self):
        SPCITO.load_project(self.dataset / 'Tiny')
        self.assertEqual(rq3.evaluate_loaded([0, 1, 2]), (5.0, 3))
        self.assertEqual(rq3.evaluate_loaded([2, 1, 0]), SPCITO.compute_sequence_cost([2, 1, 0]))

    def test_zero_denominator_and_change_labels(self):
        self.assertEqual(rq3.reduction(0, 0), '')
        self.assertEqual(rq3.reduction(2, 3), -50)
        self.assertEqual(rq3.outcome(5, 5 + 1e-12, cost=True), 'unchanged')
        self.assertEqual(rq3.outcome(3, 4), 'worsened')

    def test_success_outputs_pairing_names_negative_stubs_and_resume(self):
        originals = {file: file.read_bytes() for file in self.inputs.rglob('*.csv')}
        with mock.patch.object(rq3, 'run_worker', side_effect=self.fake_result) as worker:
            self.assertEqual(self.main(), 0)
        self.assertEqual(worker.call_count, 4)
        for call in worker.call_args_list:
            request = call.args[0]
            self.assertEqual(request['algorithm'], 'SW-SPDijkstra')
            self.assertEqual(request['initial_order'], [0, 1, 2])
            self.assertEqual(request['seed'], 0)  # unused by this deterministic postprocessor
        batch, = self.outputs.iterdir()
        for filename in ('Tiny_GA+.csv', 'Tiny_SPGreedy+.csv'):
            with (batch / 'Tiny' / filename).open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, rq3.FIELDS)
                self.assertEqual(len(list(reader)), 2)
        audit = self.read_rows(batch / 'paired_runs.csv')
        self.assertEqual(len(audit), 4)
        self.assertTrue(all(row['SStubsReduction'] == '-1' and row['SStubsOutcome'] == 'worsened' for row in audit))
        self.assertTrue(all(row['BeforeCITO'] == '[0,1,2]' and row['InitialSeed'] == '' for row in audit))
        summary = self.read_rows(batch / 'summary.csv')
        self.assertTrue(all(row['SuccessfulRuns'] == '2' and row['SStubsWorsened'] == '2' for row in summary))
        self.assertTrue(all(float(row['OcplxReductionPctOfMeans']) == 20 for row in summary))
        with mock.patch.object(rq3, 'run_worker', side_effect=AssertionError('success must not rerun')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq3.main(['--resume', str(batch)]), 0)
        self.assertEqual(len(self.read_rows(batch / 'paired_runs.csv')), 4)
        for file, content in originals.items():
            self.assertEqual(file.read_bytes(), content)

    def test_failed_record_remains_in_audit_then_can_be_retried(self):
        def fail_second(request, timeout):
            if failed.call_count == 2:
                raise RuntimeError('synthetic failure')
            return self.fake_result(request, timeout)
        with mock.patch.object(rq3, 'run_worker', side_effect=fail_second) as failed:
            self.assertEqual(self.main(), 1)
        batch, = self.outputs.iterdir()
        config = json.loads((batch / 'configuration.json').read_text(encoding='utf-8'))
        self.assertFalse(config['completed'])
        self.assertEqual((config['successful_runs'], config['failed_runs'], config['pending_runs']), (1, 1, 2))
        audit = self.read_rows(batch / 'paired_runs.csv')
        self.assertEqual(audit[-1]['Status'], 'error')
        self.assertEqual(audit[-1]['Ocplx'], '')
        with mock.patch.object(rq3, 'run_worker', side_effect=self.fake_result) as resumed, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq3.main(['--resume', str(batch)]), 0)
        self.assertEqual(resumed.call_count, 3)
        audit = self.read_rows(batch / 'paired_runs.csv')
        self.assertEqual(len(audit), 5)
        self.assertEqual(audit[2]['Attempt'], '2')
        self.assertEqual(len(self.read_rows(batch / 'Tiny' / 'Tiny_GA+.csv')), 2)

    def test_resume_rejects_changed_environment_or_sources(self):
        with mock.patch.object(rq3, 'run_worker', side_effect=self.fake_result):
            self.main()
        batch, = self.outputs.iterdir()
        with mock.patch.object(rq3, 'environment_signature', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'environment'):
            rq3.main(['--resume', str(batch)])
        file = self.inputs / 'Tiny' / 'Tiny_GA.csv'
        rows = self.read_rows(file)
        rows[0]['RuntimeMs'] = '99'
        self.write_input('GA', rows=rows)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'sources_sha256'):
            rq3.main(['--resume', str(batch)])

    def test_existing_batch_lock_rejects_concurrent_resume(self):
        with mock.patch.object(rq3, 'run_worker', side_effect=self.fake_result):
            self.main()
        batch, = self.outputs.iterdir()
        lock = batch / '.running.lock'
        self.assertFalse(lock.exists())
        lock.write_text('existing owner', encoding='utf-8')
        before = (batch / 'paired_runs.csv').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'locked'):
            rq3.main(['--resume', str(batch)])
        self.assertEqual(lock.read_text(encoding='utf-8'), 'existing owner')
        self.assertEqual((batch / 'paired_runs.csv').read_bytes(), before)

    def test_real_worker_smoke_and_complete_window(self):
        self.assertEqual(self.main(['--algorithms', 'GA', '--run-ids', '1', '--window-size', '3']), 0)
        batch, = self.outputs.iterdir()
        row, = self.read_rows(batch / 'Tiny' / 'Tiny_GA+.csv')
        self.assertEqual(sorted(json.loads(row['CITO'])), [0, 1, 2])
        self.assertLessEqual(float(row['Ocplx']), 5)
        self.assertGreater(float(row['RuntimeMs']), 0)
        self.assertGreater(float(row['PeakRSSMiB']), 0)
        SPCITO.load_project(self.dataset / 'Tiny')
        self.assertEqual(SPCITO.compute_sequence_cost(json.loads(row['CITO'])), (float(row['Ocplx']), int(row['SStubs'])))


if __name__ == '__main__':
    unittest.main()
