"""Measure SW-SPDijkstra across window sizes, restarting from each original order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from SPCITO import SPCITO
from experiments import run_generation

FIELDS = ['Window_Size', 'Run', 'Project', 'Algorithm', 'CITO', 'Ocplx',
          'SStubs', 'RuntimeMs', 'PeakRSSMiB', 'Status', 'SourceRunFile']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'datasets' / 'Ding')
    parser.add_argument('--projects', nargs='+')
    parser.add_argument('--windows', nargs='+', type=int, default=list(range(1, 19)))
    parser.add_argument('--runs', type=int, default=1,
                        help='Resource repetitions per project/window (paper sweep: 1).')
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--worker-timeout', type=float, default=300)
    parser.add_argument('--initial-orders-dir', type=Path,
                        help='Optional Project.json files. Default: Sequential [0,1,...,n-1].')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result' / 'raw' / 'window_size')
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error('runs must be positive; warmup-runs must be nonnegative.')
    if any(window < 1 for window in args.windows) or len(set(args.windows)) != len(args.windows):
        parser.error('windows must be distinct positive integers.')
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        parser.error('worker-timeout must be positive and finite.')
    if args.projects and len(set(args.projects)) != len(args.projects):
        parser.error('projects must be distinct.')
    return args


def prepare_inputs(args):
    """Validate all inputs before creating outputs or launching any worker."""
    projects = sorted(path.resolve() for path in args.dataset_root.iterdir() if path.is_dir())
    if args.projects:
        missing = set(args.projects) - {project.name for project in projects}
        if missing:
            raise ValueError(f'Unknown projects: {sorted(missing)}')
        projects = [project for project in projects if project.name in args.projects]
    if not projects:
        raise ValueError('No projects selected.')
    inputs = []
    for project in projects:
        paths = SPCITO.load_project(project)
        order = list(range(SPCITO.n_classes))
        original_path = None
        if args.initial_orders_dir is not None:
            original_path = args.initial_orders_dir / f'{project.name}.json'
            order = json.loads(original_path.read_text(encoding='utf-8-sig'))
            if not isinstance(order, list):
                raise ValueError(f'{original_path}: expected a JSON array.')
        SPCITO.validate_sequence(order)
        inputs.append({
            'project': project.name, 'project_path': str(project), 'initial_order': order,
            'n_classes': SPCITO.n_classes,
            'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in paths.items()},
            'original_order_path': str(original_path.resolve()) if original_path else None,
            'original_order_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest()
                                     if original_path else None,
        })
    return inputs


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    inputs = prepare_inputs(args)
    destination = args.output_dir.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True)
    order_dir = destination / 'initial_orders'
    order_dir.mkdir()
    for item in inputs:
        write_json(order_dir / f'{item["project"]}.json', item['initial_order'])
    config = {
        'created_at': datetime.now().astimezone().isoformat(), 'completed': False,
        'finished': False, 'algorithm': 'SW-SPDijkstra', 'windows': args.windows,
        'runs_per_project_window': args.runs, 'warmup_runs': args.warmup_runs,
        'worker_timeout_seconds': args.worker_timeout, 'stride': 1, 'passes': 1,
        'initial_order_source': 'Sequential' if args.initial_orders_dir is None else 'JSON files',
        'restart_policy': 'Every window setting and repetition starts from its original input order.',
        'inputs': inputs, 'source_sha256': run_generation.source_hashes(),
        'batches': [],
    }
    config_path = destination / 'configuration.json'
    write_json(config_path, config)
    combined = destination / 'wsize_sweep_ding_all.csv'
    print(f'Sweep results: {destination}', flush=True)
    status = 0
    rows_written = 0
    with combined.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        try:
            for item in inputs:
                project_dir = destination / f'wsize_sweep_{item["project"]}'
                project_dir.mkdir()
                for window in args.windows:
                    before = set(project_dir.iterdir())
                    status = run_generation.main([
                        '--algorithm', 'SW-SPDijkstra', '--project-dir', item['project_path'],
                        '--initial-order', str(order_dir / f'{item["project"]}.json'),
                        '--window-size', str(window), '--runs', str(args.runs),
                        '--warmup-runs', str(args.warmup_runs),
                        '--worker-timeout', str(args.worker_timeout),
                        '--output-dir', str(project_dir),
                    ])
                    created = set(project_dir.iterdir()) - before
                    if len(created) != 1:
                        raise RuntimeError('Expected exactly one new generation batch per setting.')
                    batch = created.pop()
                    run_file = batch / 'runs.csv'
                    config['batches'].append({'Project': item['project'], 'WindowSize': window,
                                              'directory': batch.relative_to(destination).as_posix(),
                                              'exit_status': status})
                    with run_file.open(encoding='utf-8-sig', newline='') as source:
                        for row in csv.DictReader(source):
                            writer.writerow({
                                'Window_Size': window,
                                **{key: row[key] for key in FIELDS[1:-1]},
                                'SourceRunFile': run_file.relative_to(destination).as_posix(),
                            })
                            rows_written += 1
                    stream.flush()
                    write_json(config_path, config)
                    if status != 0:
                        break
                if status != 0:
                    break
        except (KeyboardInterrupt, Exception) as error:
            status = 130 if isinstance(error, KeyboardInterrupt) else 1
            config['error'] = f'{type(error).__name__}: {error}'
            print(config['error'], file=sys.stderr)
        finally:
            expected = len(inputs) * len(args.windows) * args.runs
            config.update(finished=True, completed=status == 0 and rows_written == expected,
                          recorded_runs=rows_written, expected_runs=expected)
            write_json(config_path, config)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
