"""Window sweeps preserve original inputs and isolated measurement records."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from experiments import run_window_sweep as sweep
from SPCITO import SPCITO

TEMP_ROOT = ROOT / 'experiments' / 'tests' / '.tmp'
TEMP_ROOT.mkdir(exist_ok=True)


class WindowSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.dataset = self.folder / 'data'
        project = self.dataset / 'Tiny'
        project.mkdir(parents=True)
        np.savetxt(project / 'StubComplexityMatrix.csv', [[0, 2, 0], [1, 0, 3], [4, 0, 0]], delimiter=',')
        np.savetxt(project / 'DependencyMatrix.csv', [[0, 1, 1], [1, 0, 1], [1, 0, 0]], delimiter=',', fmt='%d')
        self.output = self.folder / 'output'
        self.argv = ['--dataset-root', str(self.dataset), '--windows', '1', '3',
                     '--warmup-runs', '0', '--output-dir', str(self.output)]

    def test_paper_defaults_and_invalid_windows(self):
        args = sweep.parse_args([])
        self.assertEqual(args.windows, list(range(1, 19)))
        self.assertEqual((args.runs, args.warmup_runs), (1, 1))
        for values in (['0'], ['1', '1']):
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sweep.parse_args(['--windows', *values])

    def test_sequential_and_explicit_input_validation(self):
        args = sweep.parse_args(self.argv)
        self.assertEqual(sweep.prepare_inputs(args)[0]['initial_order'], [0, 1, 2])
        source = self.folder / 'orders'
        source.mkdir()
        order_file = source / 'Tiny.json'
        order_file.write_text('[2,0,1]', encoding='utf-8')
        args.initial_orders_dir = source
        entry, = sweep.prepare_inputs(args)
        self.assertEqual(entry['initial_order'], [2, 0, 1])
        self.assertIsNotNone(entry['original_order_sha256'])
        order_file.write_text('[2,2,1]', encoding='utf-8')
        with self.assertRaises(ValueError):
            sweep.prepare_inputs(args)
        self.assertFalse(self.output.exists())

    def test_real_sweep_restarts_every_setting(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.main(self.argv), 0)
        batch, = self.output.iterdir()
        config = json.loads((batch / 'configuration.json').read_text(encoding='utf-8'))
        self.assertTrue(config['completed'])
        self.assertEqual(config['recorded_runs'], 2)
        with (batch / 'wsize_sweep_ding_all.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row['Window_Size']) for row in rows], [1, 3])
        self.assertEqual(json.loads(rows[0]['CITO']), [0, 1, 2])
        self.assertLess(float(rows[1]['Ocplx']), float(rows[0]['Ocplx']))
        for row in rows:
            with (batch / row['SourceRunFile']).open(encoding='utf-8-sig', newline='') as stream:
                raw, = csv.DictReader(stream)
            self.assertEqual(json.loads(raw['InitialCITO']), [0, 1, 2])
            self.assertEqual(raw['Status'], 'success')
            self.assertGreater(float(raw['PeakRSSMiB']), 0)
            self.assertEqual(raw['RuntimeNs'], raw['SWRuntimeNs'])
            SPCITO.load_project(self.dataset / 'Tiny')
            self.assertEqual(SPCITO.compute_sequence_cost(json.loads(raw['CITO'])),
                             (float(raw['Ocplx']), int(raw['SStubs'])))

    def test_failure_stops_and_retains_failed_record(self):
        with mock.patch.object(sweep.run_generation, 'run_worker', side_effect=RuntimeError('test failure')), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sweep.main(self.argv), 1)
        batch, = self.output.iterdir()
        config = json.loads((batch / 'configuration.json').read_text(encoding='utf-8'))
        self.assertFalse(config['completed'])
        self.assertEqual(len(config['batches']), 1)
        with (batch / 'wsize_sweep_ding_all.csv').open(encoding='utf-8-sig', newline='') as stream:
            row, = csv.DictReader(stream)
        self.assertEqual(row['Status'], 'error')
        self.assertEqual(row['RuntimeMs'], '')


if __name__ == '__main__':
    unittest.main()
