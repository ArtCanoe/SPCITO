"""Portable, isolated time/PeakRSS experiments for the SPCITO algorithms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import secrets
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from SPCITO import SPCITO

ALGORITHMS = ('SPGreedy+', 'SPGreedy', 'SPDijkstra', 'SW-SPDijkstra')
FIELDS = ['Run', 'Project', 'Algorithm', 'CITO', 'Ocplx', 'SStubs', 'RuntimeMs', 'PeakRSSMiB',
          'Seed', 'Status', 'NClasses', 'WindowSize', 'RuntimeNs', 'SPGreedyRuntimeNs', 'SWRuntimeNs',
          'SPGreedyRuntimeMs', 'SWRuntimeMs', 'InitialCITO', 'InitialOcplx', 'InitialSStubs',
          'OcplxReduction', 'SStubsReduction', 'PeakRSSSource', 'WorkerPid', 'WarmupRuns', 'Error']
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
              'PYTHONHASHSEED': '0', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}


def generate_unique_seeds(count, master_seed=None):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('Seed count must be a positive integer.')
    rng = random.Random(master_seed) if master_seed is not None else None
    used, seeds = set(), []
    while len(seeds) < count:
        seed = rng.getrandbits(64) if rng is not None else secrets.randbits(64)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    return seeds


def parse_args(argv=None, *, default_algorithm='SPGreedy+', default_runs=30, default_window_size=10):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--project-dir', type=Path, help='One project directory containing the 2 input matrices.')
    source.add_argument('--dataset-root', type=Path, help='A dataset root; process all its project directories unless --projects is given.')
    parser.add_argument('--projects', nargs='+')
    parser.add_argument('--algorithm', choices=ALGORITHMS, default=default_algorithm)
    parser.add_argument('--runs', type=int, default=default_runs)
    parser.add_argument('--window-size', type=int, default=default_window_size)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--worker-timeout', type=float, default=300)
    parser.add_argument('--max-exact-classes', type=int, default=24,
                        help='SPDijkstra guard on initially unset bits; explicitly increase only if resources permit.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result' / 'raw' / 'generation')
    parser.add_argument('--initial-order', type=Path, help='JSON array for standalone SW-SPDijkstra; exactly 1 project required.')
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument('--seed', type=int, help='Replay an exact seed; requires 1 project and --runs 1.')
    seed.add_argument('--master-seed', type=int, help='Optional repeatable batch; omit for fresh OS-generated seeds.')
    args = parser.parse_args(argv)
    if args.runs < 1 or args.window_size < 1 or args.warmup_runs < 0 or args.max_exact_classes < 1:
        parser.error('runs/window-size/max-exact-classes must be positive; warmup-runs must be nonnegative.')
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        parser.error('worker-timeout must be positive and finite.')
    if (args.seed is not None and args.seed < 0) or (args.master_seed is not None and args.master_seed < 0):
        parser.error('Seeds must be nonnegative.')
    if args.projects and args.dataset_root is None:
        parser.error('--projects requires --dataset-root.')
    if (args.algorithm == 'SW-SPDijkstra') != (args.initial_order is not None):
        parser.error('Use --initial-order exactly when selecting standalone SW-SPDijkstra.')
    return args


def run_worker(request, timeout):
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    try:
        proc = subprocess.run([sys.executable, '-B', str(ROOT / 'experiments' / 'measurement' / 'worker.py')], cwd=ROOT,
                              input=json.dumps(request), capture_output=True, text=True, encoding='utf-8',
                              env=environment, timeout=timeout, check=False,
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'Worker exceeded {timeout:g} seconds; the worker was terminated.') from error
    if proc.returncode != 0:
        raise RuntimeError(f'Worker exit={proc.returncode}: {proc.stderr[-6000:]}')
    result = json.loads(proc.stdout)
    if result.get('status') != 'success' or result.get('Algorithm') != request['algorithm'] or result.get('Seed') != request['seed']:
        raise RuntimeError('Worker response does not match the request.')
    for key in ('RuntimeMs', 'PeakRSSMiB'):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f'Invalid worker measurement: {key}')
    if result['RuntimeMs'] != result['RuntimeNs'] / 1e6:
        raise RuntimeError('Inconsistent time units.')
    if request['algorithm'] == 'SPGreedy+' and result['RuntimeNs'] != result['SPGreedyRuntimeNs'] + result['SWRuntimeNs']:
        raise RuntimeError('Stage times do not sum to the continuous total.')
    return result


def source_hashes():
    """Hash executable collection sources, using portable repository paths."""
    files = list((ROOT / 'src' / 'SPCITO').glob('*.py'))
    files += list((ROOT / 'experiments').glob('*.py'))
    files += list((ROOT / 'experiments' / 'measurement').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def write_summary(rows, destination):
    metrics = ('Ocplx', 'SStubs', 'RuntimeMs', 'PeakRSSMiB', 'SPGreedyRuntimeMs', 'SWRuntimeMs')
    stats = ('min', 'mean', 'median', 'stdev', 'max')
    fields = ['Project', 'Algorithm', 'SuccessfulRuns', 'FailedRuns'] + [f'{metric}_{stat}' for metric in metrics for stat in stats]
    with destination.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for project in sorted({row['Project'] for row in rows}):
            selected = [row for row in rows if row['Project'] == project]
            success = [row for row in selected if row['Status'] == 'success']
            summary = {'Project': project, 'Algorithm': selected[0]['Algorithm'],
                       'SuccessfulRuns': len(success), 'FailedRuns': len(selected) - len(success)}
            for metric in metrics:
                values = [float(row[metric]) for row in success if row.get(metric) not in (None, '')]
                if values:
                    summary.update(dict(zip((f'{metric}_{stat}' for stat in stats),
                                            (min(values), statistics.fmean(values), statistics.median(values),
                                             statistics.stdev(values) if len(values) > 1 else 0.0, max(values)))))
            writer.writerow(summary)


def main(argv=None, *, default_algorithm='SPGreedy+', default_runs=30, default_window_size=10):
    args = parse_args(argv, default_algorithm=default_algorithm, default_runs=default_runs,
                      default_window_size=default_window_size)
    if args.dataset_root is not None:
        projects = sorted(p.resolve() for p in args.dataset_root.iterdir() if p.is_dir())
        if args.projects:
            missing = set(args.projects) - {p.name for p in projects}
            if missing:
                raise ValueError(f'Unknown projects: {sorted(missing)}')
            projects = [p for p in projects if p.name in args.projects]
    else:
        projects = [(args.project_dir or SPCITO.DEFAULT_PROJECT).resolve()]
    if not projects:
        raise ValueError('No projects selected.')
    if args.seed is not None and (len(projects) != 1 or args.runs != 1):
        raise ValueError('--seed requires exactly 1 project and --runs 1.')
    initial_order = None
    if args.initial_order is not None:
        if len(projects) != 1:
            raise ValueError('Standalone SW-SPDijkstra requires exactly 1 project.')
        initial_order = json.loads(args.initial_order.read_text(encoding='utf-8-sig'))
        if not isinstance(initial_order, list) or any(type(value) is not int for value in initial_order):
            raise ValueError('Initial order must be a JSON array of integer class indices.')
    fingerprints = {project.name: {key: hashlib.sha256(file.read_bytes()).hexdigest()
                                  for key, file in SPCITO.input_paths(project).items()} for project in projects}
    seeds = [args.seed] if args.seed is not None else generate_unique_seeds(len(projects) * args.runs, args.master_seed)
    planned = [{'Project': project.name, 'Run': run, 'Seed': seeds[index * args.runs + run - 1]}
               for index, project in enumerate(projects) for run in range(1, args.runs + 1)]
    directory = args.output_dir.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    config = {
        'created_at': datetime.now().astimezone().isoformat(), 'completed': False, 'finished': False,
        'algorithm': args.algorithm, 'runs_per_project': args.runs, 'planned_runs': planned,
        'seed_source': 'explicit' if args.seed is not None else 'master-seed' if args.master_seed is not None else 'os-random',
        'project_paths': {project.name: str(project) for project in projects}, 'input_sha256': fingerprints,
        'source_sha256': source_hashes(),
        'python_version': sys.version, 'python_executable': sys.executable,
        'numpy_version': importlib.metadata.version('numpy'), 'platform': platform.platform(),
        'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'worker_environment': THREAD_ENV, 'execution_mode': 'sequential fresh Python process per repetition',
        'window_size': args.window_size, 'warmup_runs': args.warmup_runs,
        'worker_timeout_seconds': args.worker_timeout, 'max_exact_classes': args.max_exact_classes,
        'initial_order': initial_order,
        'initial_order_sha256': hashlib.sha256(args.initial_order.read_bytes()).hexdigest() if args.initial_order else None,
        'timing_clock': 'time.perf_counter_ns',
        'timing_included': 'algorithm preprocessing/pruning, decision cost calculations, search, window construction/update, sequence recovery and internal permutation checks',
        'timing_excluded': 'imports, dataset loading, RNG initialization, warmup, independent initial/final cost and stub evaluation, input hashing, memory queries, logging, process startup/transfer/shutdown and file output',
        'spgreedyplus_timing': 'continuous t0->SPGreedy->t1->SW-SPDijkstra->t2; total=t2-t0; stages sum exactly in nanoseconds',
        'memory_metric': 'PeakRSSMiB', 'memory_unit': 'MiB (1048576 bytes)',
        'memory_scope': 'worker lifetime through result preparation/readout; includes Python/NumPy/native runtime, current project matrices, warmup, algorithm, verification and result preparation; excludes parent process, final transfer/CSV writing and subsequent shutdown; not exclusive algorithm memory',
        'memory_source': 'Linux /proc/self/status VmHWM or Windows GetProcessMemoryInfo PeakWorkingSetSize; no polling/tracemalloc',
        'warmup_rng': 'recreate numpy.random.default_rng(recorded_seed) for every warmup and the measured run',
        'pruning': 'original zero-cost row/column rules retained; dependency matrix still determines SStubs, including zero-cost dependency pairs',
        'summary_stdev': 'sample standard deviation; 0 for a single observed run',
    }
    config_path = directory / 'configuration.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Results: {directory}', flush=True)
    rows = []
    with (directory / 'runs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for item in planned:
            row = {'Run': item['Run'], 'Project': item['Project'], 'Algorithm': args.algorithm, 'Seed': item['Seed']}
            print(f"Starting {item['Project']} run={item['Run']} seed={item['Seed']}", flush=True)
            request = {'project_dir': config['project_paths'][item['Project']], 'algorithm': args.algorithm,
                       'seed': item['Seed'], 'window_size': args.window_size, 'initial_order': initial_order,
                       'warmup_runs': args.warmup_runs, 'max_exact_classes': args.max_exact_classes,
                       'input_sha256': fingerprints[item['Project']]}
            try:
                result = run_worker(request, args.worker_timeout)
                row.update({key: value for key, value in result.items() if key in FIELDS})
                row['Status'] = 'success'
                for key in ('CITO', 'InitialCITO'):
                    row[key] = json.dumps(result[key], separators=(',', ':')) if result.get(key) is not None else ''
                for ns, ms in (('SPGreedyRuntimeNs', 'SPGreedyRuntimeMs'), ('SWRuntimeNs', 'SWRuntimeMs')):
                    row[ms] = result[ns] / 1e6 if result[ns] is not None else ''
                if result.get('InitialCITO') is not None:
                    row['OcplxReduction'] = result['InitialOcplx'] - result['Ocplx']
                    row['SStubsReduction'] = result['InitialSStubs'] - result['SStubs']
                print(f"OCplx={row['Ocplx']:.6f} SStubs={row['SStubs']} "
                      f"time={row['RuntimeMs']:.3f} ms PeakRSS={row['PeakRSSMiB']:.3f} MiB", flush=True)
            except Exception as error:
                row.update({'Status': 'error', 'Error': str(error)})
                print(f'ERROR: {error}', file=sys.stderr, flush=True)
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            if row['Status'] != 'success':
                break  # Preserve the failed seed; do not keep spawning failing workers.
    write_summary(rows, directory / 'summary.csv')
    config.update({'finished': True, 'completed': len(rows) == len(planned) and all(row['Status'] == 'success' for row in rows),
                   'recorded_runs': len(rows), 'successful_runs': sum(row['Status'] == 'success' for row in rows)})
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Completed={config['completed']}; recorded={len(rows)}/{len(planned)}", flush=True)
    return 0 if config['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
