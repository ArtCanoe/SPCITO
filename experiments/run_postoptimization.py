"""Paired SW-SPDijkstra experiments on recorded CITOs; no baseline regeneration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from SPCITO import SPCITO
from experiments.run_generation import THREAD_ENV, run_worker, source_hashes

SCHEMA_VERSION = 1
BASE = ROOT / 'result'
FIELDS = ['Run', 'Project', 'Algorithm', 'CITO', 'Ocplx', 'SStubs', 'RuntimeMs', 'PeakRSSMiB']
AUDIT_FIELDS = FIELDS + [
    'InitialAlgorithm', 'Attempt', 'Status', 'BeforeCITO', 'BeforeOcplx', 'BeforeSStubs',
    'SourceOcplx', 'SourceSStubs', 'SourceFile', 'SourceSHA256', 'InitialSeed',
    'WindowSize', 'EffectiveWindowSize', 'WarmupRuns', 'RuntimeNs', 'PeakRSSSource', 'WorkerPid',
    'OcplxReduction', 'SStubsReduction', 'OcplxReductionPct', 'SStubsReductionPct',
    'OcplxOutcome', 'SStubsOutcome', 'Error',
]
SETTING_NAMES = ('input_root', 'dataset_root', 'output_dir', 'projects', 'algorithms',
                 'run_ids', 'expected_runs', 'window_size', 'warmup_runs', 'worker_timeout')


def digest(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def finite_nonnegative(value, label):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{label} must be finite and nonnegative: {value!r}')
    return number


def integer(value, label, minimum=0):
    if not re.fullmatch(r'\d+', str(value)) or int(value) < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}: {value!r}')
    return int(value)


def cost_close(left, right):
    # Only accommodates CSV/binary arithmetic rounding, not different cost models.
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-9)


def order_json(order):
    return json.dumps(order, separators=(',', ':'))


def validate_order(order, n):
    if (not isinstance(order, list) or len(order) != n
            or any(type(value) is not int for value in order) or set(order) != set(range(n))):
        raise ValueError(f'CITO must be a complete zero-based permutation of {n} classes.')


def evaluate_loaded(order):
    """Independent vectorized preflight evaluator, never inside a worker timer."""
    validate_order(order, SPCITO.n_classes)
    index = np.asarray(order, dtype=int)
    costs = SPCITO.cost_matrix[np.ix_(index, index)]
    deps = SPCITO.dependency_matrix[np.ix_(index, index)]
    return float(np.triu(costs, 1).sum()), int(np.count_nonzero(np.triu(deps, 1)))


def require_safe_name(value, label):
    if not value or value in ('.', '..') or re.search(r'[<>:"/\\|?*\x00-\x1f]', value) or value[-1] in '. ':
        raise ValueError(f'Unsafe {label}: {value!r}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, help='Recorded results: Project/Project_Algorithm.csv.')
    parser.add_argument('--dataset-root', type=Path, help='Project directories containing cost/dependency matrices.')
    parser.add_argument('--output-dir', type=Path, help='A new timestamped batch is created beneath this directory.')
    parser.add_argument('--projects', nargs='+')
    parser.add_argument('--algorithms', nargs='+', help='Initial method names, e.g. GA SPGreedy; never SPGreedy+.')
    parser.add_argument('--run-ids', nargs='+', type=int, help='Optional source Run selection for a smoke test.')
    parser.add_argument('--expected-runs', type=int, help='Required source runs per file (default: 30).')
    parser.add_argument('--window-size', type=int, help='Default: 10; stride 1 and one left-to-right sweep.')
    parser.add_argument('--warmup-runs', type=int, help='Default: 1; always restart from the original CITO.')
    parser.add_argument('--worker-timeout', type=float, help='Per-worker timeout in seconds (default: 300).')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and metrics without optimizing or writing results.')
    parser.add_argument('--resume', type=Path, help='Resume an existing batch; do not run it concurrently elsewhere.')
    args = parser.parse_args(argv)
    if args.resume and any(getattr(args, key) is not None for key in SETTING_NAMES):
        parser.error('--resume reads its saved settings; do not combine it with input/output or experiment settings.')
    if not args.resume:
        defaults = dict(input_root=BASE / 'processed' / 'Ding',
                        dataset_root=ROOT / 'datasets' / 'Ding',
                        output_dir=BASE / 'raw' / 'postoptimization',
                        expected_runs=30, window_size=10, warmup_runs=1, worker_timeout=300.0)
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        if args.expected_runs < 1 or args.window_size < 1 or args.warmup_runs < 0:
            parser.error('expected-runs/window-size must be positive; warmup-runs must be nonnegative.')
        if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
            parser.error('worker-timeout must be finite and positive.')
        if args.run_ids and (len(set(args.run_ids)) != len(args.run_ids)
                             or any(run < 1 or run > args.expected_runs for run in args.run_ids)):
            parser.error('run-ids must be unique source Run numbers within 1..expected-runs.')
    return args


def settings_from_args(args):
    return {key: str(getattr(args, key).resolve()) if key.endswith('_root') or key == 'output_dir'
            else getattr(args, key) for key in SETTING_NAMES}


def build_plan(settings):
    source_root = Path(settings['input_root']).resolve()
    dataset_root = Path(settings['dataset_root']).resolve()
    projects = sorted(entry for entry in source_root.iterdir() if entry.is_dir())
    requested = set(settings['projects'] or (entry.name for entry in projects))
    if requested - {entry.name for entry in projects}:
        raise ValueError(f'Unknown source projects: {sorted(requested - {entry.name for entry in projects})}')
    groups, sources, matrices = [], {}, {}
    metric_mismatches = []
    for directory in projects:
        project = directory.name
        if project not in requested:
            continue
        require_safe_name(project, 'project name')
        selected_algorithms = set()
        for filename in sorted(directory.glob('*.csv')):
            content = filename.read_bytes()
            reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)) or not set(FIELDS).issubset(headers):
                raise ValueError(f'{filename}: missing or duplicated input columns.')
            rows = list(reader)
            if not rows:
                raise ValueError(f'{filename}: empty result file.')
            algorithm = rows[0]['Algorithm']
            if algorithm == 'SPGreedy+':
                continue
            if settings['algorithms'] and algorithm not in settings['algorithms']:
                continue
            require_safe_name(algorithm, 'algorithm name')
            if algorithm.endswith('+'):
                raise ValueError(f'{filename}: already postoptimized algorithm {algorithm!r}.')
            if algorithm in selected_algorithms:
                raise ValueError(f'{project}: multiple source files for {algorithm}.')
            selected_algorithms.add(algorithm)
            if project not in matrices:
                project_dir = dataset_root / project
                paths = SPCITO.input_paths(project_dir)
                hashes = {key: digest(value) for key, value in paths.items()}
                SPCITO.load_project(project_dir)
                matrices[project] = {'directory': str(project_dir), 'n_classes': SPCITO.n_classes,
                                     'sha256': hashes}
            seen, records = set(), []
            relative = filename.relative_to(source_root).as_posix()
            source_hash = hashlib.sha256(content).hexdigest()
            for row in rows:
                if None in row or any(row.get(key) is None for key in FIELDS):
                    raise ValueError(f'{filename}: malformed CSV row.')
                if row['Project'] != project or row['Algorithm'] != algorithm:
                    raise ValueError(f'{filename}: mixed project/algorithm records.')
                if row.get('Status', 'success') != 'success':
                    raise ValueError(f'{filename}: unsuccessful source experiment.')
                run = integer(row['Run'], 'Run', 1)
                if run in seen:
                    raise ValueError(f'{filename}: duplicate Run {run}.')
                seen.add(run)
                before_cost = finite_nonnegative(row['Ocplx'], 'Ocplx')
                before_stubs = integer(row['SStubs'], 'SStubs')
                order = json.loads(row['CITO'])
                cost, stubs = evaluate_loaded(order)
                if not cost_close(cost, before_cost) or stubs != before_stubs:
                    metric_mismatches.append((project, algorithm, run, before_cost, before_stubs, cost, stubs))
                initial_seed = row.get('Seed') or row.get('InitialSeed') or ''
                if initial_seed:
                    integer(initial_seed, 'InitialSeed')
                if settings['run_ids'] and run not in settings['run_ids']:
                    continue
                records.append({'Run': run, 'Project': project, 'InitialAlgorithm': algorithm,
                                'Algorithm': algorithm + '+', 'BeforeCITO': order,
                                'SourceOcplx': row['Ocplx'], 'SourceSStubs': row['SStubs'],
                                'SourceFile': relative, 'SourceSHA256': source_hash, 'InitialSeed': initial_seed})
            if seen != set(range(1, settings['expected_runs'] + 1)):
                raise ValueError(f'{filename}: expected exactly Runs 1..{settings["expected_runs"]}.')
            sources[relative] = source_hash
            groups.append({'Project': project, 'InitialAlgorithm': algorithm, 'Algorithm': algorithm + '+',
                           'records': sorted(records, key=lambda record: record['Run'])})
        if not selected_algorithms:
            raise ValueError(f'{project}: no eligible initial methods.')
        if settings['algorithms'] and selected_algorithms != set(settings['algorithms']):
            raise ValueError(f'{project}: missing initial methods {set(settings["algorithms"]) - selected_algorithms}.')
    if not groups:
        raise ValueError('No eligible source records.')
    # A common method set is required across the selected projects.
    method_sets = [{group['InitialAlgorithm'] for group in groups if group['Project'] == project} for project in requested]
    if any(methods != method_sets[0] for methods in method_sets):
        raise ValueError('Selected projects do not contain the same set of initial methods.')
    if metric_mismatches:
        counts = Counter((project, algorithm) for project, algorithm, *_ in metric_mismatches)
        groups_text = '; '.join(f'{project}/{algorithm}: {count}' for (project, algorithm), count in counts.items())
        p, a, run, old_cost, old_stubs, cost, stubs = metric_mismatches[0]
        raise ValueError(f'{len(metric_mismatches)} source records disagree with matrix evaluation ({groups_text}). '
                         f'First: {p}/{a}, Run {run}, source=({old_cost}, {old_stubs}), '
                         f'recomputed=({cost}, {stubs}). No results written; source files not modified.')
    for relative, expected in sources.items():
        if digest(source_root / relative) != expected:
            raise RuntimeError('Source results changed during preflight.')
    for project, meta in matrices.items():
        if {key: digest(file) for key, file in SPCITO.input_paths(meta['directory']).items()} != meta['sha256']:
            raise RuntimeError(f'{project}: input matrices changed during preflight.')
    return groups, sources, matrices


def environment_signature():
    return {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
            'machine': platform.machine(), 'processor': platform.processor(),
            'hostname': platform.node(), 'logical_cpu_count': os.cpu_count(), 'worker_environment': THREAD_ENV}


def code_hashes():
    return source_hashes()


def atomic_text(filename, text):
    temporary = filename.with_name(filename.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, filename)


def write_csv(filename, fields, rows):
    content = io.StringIO(newline='')
    writer = csv.DictWriter(content, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(filename, '\ufeff' + content.getvalue())


def key_of(row):
    return row['Project'], row['InitialAlgorithm'], int(row['Run'])


def reduction(before, after):
    return (before - after) / before * 100 if before != 0 else ''


def outcome(before, after, *, cost=False):
    equal = cost_close(before, after) if cost else before == after
    if equal:
        return 'unchanged'
    return 'improved' if after < before else 'worsened'


def complete_record(source, result, attempt, settings, n):
    validate_order(result['CITO'], n)
    if result['InitialCITO'] != source['BeforeCITO']:
        raise RuntimeError('Worker optimized a different initial CITO.')
    before_cost = finite_nonnegative(result['InitialOcplx'], 'BeforeOcplx')
    before_stubs = integer(result['InitialSStubs'], 'BeforeSStubs')
    after_cost = finite_nonnegative(result['Ocplx'], 'Ocplx')
    after_stubs = integer(result['SStubs'], 'SStubs')
    if not cost_close(before_cost, source['SourceOcplx']) or before_stubs != int(source['SourceSStubs']):
        raise RuntimeError('Worker baseline evaluation differs from the source.')
    if after_cost > before_cost and not cost_close(after_cost, before_cost):
        raise RuntimeError('SW-SPDijkstra increased Ocplx.')
    for metric in ('RuntimeMs', 'PeakRSSMiB'):
        if finite_nonnegative(result[metric], metric) <= 0:
            raise RuntimeError(f'Nonpositive {metric}.')
    if integer(result['RuntimeNs'], 'RuntimeNs', 1) / 1e6 != result['RuntimeMs']:
        raise RuntimeError('Inconsistent runtime units.')
    return {**source, 'Attempt': attempt, 'Status': 'success', 'CITO': order_json(result['CITO']),
            'BeforeCITO': order_json(source['BeforeCITO']), 'BeforeOcplx': before_cost, 'BeforeSStubs': before_stubs,
            'Ocplx': after_cost, 'SStubs': after_stubs, 'RuntimeMs': result['RuntimeMs'], 'PeakRSSMiB': result['PeakRSSMiB'],
            'RuntimeNs': result['RuntimeNs'], 'PeakRSSSource': result['PeakRSSSource'], 'WorkerPid': result['WorkerPid'],
            'WindowSize': settings['window_size'], 'EffectiveWindowSize': min(settings['window_size'], n),
            'WarmupRuns': settings['warmup_runs'], 'OcplxReduction': before_cost - after_cost,
            'SStubsReduction': before_stubs - after_stubs,
            'OcplxReductionPct': reduction(before_cost, after_cost), 'SStubsReductionPct': reduction(before_stubs, after_stubs),
            'OcplxOutcome': outcome(before_cost, after_cost, cost=True), 'SStubsOutcome': outcome(before_stubs, after_stubs)}


def read_journal(filename, planned):
    latest = {}
    if not filename.exists():
        raise ValueError('Missing paired_runs.csv; cannot resume safely.')
    with filename.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != AUDIT_FIELDS:
            raise ValueError('Unexpected journal columns.')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Incomplete journal row; inspect it before resuming.')
            key = key_of(row)
            if key not in planned or row['Status'] not in ('success', 'error'):
                raise ValueError('Journal contains an unknown record/status.')
            source = planned[key]
            for name in ('SourceFile', 'SourceSHA256', 'InitialSeed', 'Algorithm', 'SourceOcplx', 'SourceSStubs'):
                if str(row[name]) != str(source[name]):
                    raise ValueError(f'Journal/source mismatch: {name}.')
            if json.loads(row['BeforeCITO']) != source['BeforeCITO']:
                raise ValueError('Journal/source CITO mismatch.')
            previous = latest.get(key)
            if previous and previous['Status'] == 'success':
                raise ValueError('Journal repeats an already successful record.')
            if int(row['Attempt']) != (int(previous['Attempt']) + 1 if previous else 1):
                raise ValueError('Unexpected journal attempt number.')
            if row['Status'] == 'success':
                validate_order(json.loads(row['CITO']), len(source['BeforeCITO']))
                for metric in ('Ocplx', 'SStubs', 'RuntimeMs', 'PeakRSSMiB', 'BeforeOcplx', 'BeforeSStubs'):
                    finite_nonnegative(row[metric], metric)
                if int(row['RuntimeNs']) / 1e6 != float(row['RuntimeMs']):
                    raise ValueError('Journal contains inconsistent timing units.')
            latest[key] = row
    return latest


def export_group(directory, group, latest):
    folder = directory / group['Project']
    folder.mkdir(exist_ok=True)
    filename = folder / f'{group["Project"]}_{group["Algorithm"]}.csv'
    rows = [latest[key_of(source)] for source in group['records']
            if key_of(source) in latest and latest[key_of(source)]['Status'] == 'success']
    write_csv(filename, FIELDS, rows)


def export_summary(directory, groups, latest):
    metrics = ('BeforeOcplx', 'Ocplx', 'BeforeSStubs', 'SStubs', 'RuntimeMs', 'PeakRSSMiB')
    fields = ['Project', 'InitialAlgorithm', 'Algorithm', 'PlannedRuns', 'SuccessfulRuns', 'FailedRuns', 'PendingRuns']
    fields += [f'{metric}_{stat}' for metric in metrics for stat in ('mean', 'stdev', 'max')]
    fields += [f'{metric}{suffix}' for metric in ('Ocplx', 'SStubs')
               for suffix in ('ReductionPctOfMeans', 'Improved', 'Unchanged', 'Worsened')]
    rows = []
    for group in groups:
        records = [latest.get(key_of(source)) for source in group['records']]
        success = [row for row in records if row and row['Status'] == 'success']
        row = {name: group[name] for name in ('Project', 'InitialAlgorithm', 'Algorithm')}
        row.update(PlannedRuns=len(records), SuccessfulRuns=len(success),
                   FailedRuns=sum(item is not None and item['Status'] == 'error' for item in records),
                   PendingRuns=sum(item is None for item in records))
        for metric in metrics:
            values = [float(item[metric]) for item in success]
            if values:
                row.update({f'{metric}_mean': statistics.fmean(values),
                            f'{metric}_stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                            f'{metric}_max': max(values)})
        for metric in ('Ocplx', 'SStubs'):
            if success:
                row[f'{metric}ReductionPctOfMeans'] = reduction(row[f'Before{metric}_mean'], row[f'{metric}_mean'])
            for change in ('improved', 'unchanged', 'worsened'):
                row[f'{metric}{change.capitalize()}'] = sum(item[f'{metric}Outcome'] == change for item in success)
        rows.append(row)
    write_csv(directory / 'summary.csv', fields, rows)


def main(argv=None):
    args = parse_args(argv)
    config = None
    if args.resume:
        directory = args.resume.resolve()
        config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
        if config['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported postoptimization batch version.')
        settings = config['settings']
    else:
        settings = settings_from_args(args)
    groups, sources, matrices = build_plan(settings)
    planned = {key_of(row): row for group in groups for row in group['records']}
    print(f'Validated {len(matrices)} projects, {len(groups)} groups, {len(planned)} selected source runs.', flush=True)
    if config:
        for key, current in [('sources_sha256', sources), ('matrices', matrices),
                             ('code_sha256', code_hashes()), ('environment', environment_signature())]:
            if config[key] != current:
                raise ValueError(f'Cannot mix changed {key} into an existing batch. Start a new batch instead.')
    if args.check_only:
        print('Check only: no optimization and no result files written.')
        return 0
    if not config:
        directory = Path(settings['output_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        source_root = Path(settings['input_root']).resolve()
        dataset_root = Path(settings['dataset_root']).resolve()
        if directory.resolve().is_relative_to(source_root) or directory.resolve().is_relative_to(dataset_root):
            raise ValueError('Output must not be inside the input results or dataset directory.')
        directory.mkdir(parents=True)
        config = {'schema_version': SCHEMA_VERSION, 'created_at': datetime.now().astimezone().isoformat(),
                  'settings': settings, 'sources_sha256': sources, 'matrices': matrices,
                  'code_sha256': code_hashes(), 'environment': environment_signature(),
                  'planned_runs': len(planned), 'completed': False, 'finished': False,
                  'postprocessor': 'SW-SPDijkstra', 'stride': 1, 'sweeps': 1,
                  'rng': 'Postprocessor is deterministic; internal seed 0 is unused. InitialSeed is retained only if present in source CSV.',
                  'timing_scope': 'Optimization only, including window construction/search/update and internal validation; excludes input loading, initial generation, warmup, independent cost/stub evaluation, hashing, memory readout, logging and output.',
                  'memory_scope': 'Fresh worker process peak RSS through result preparation/readout; includes interpreter, matrices, warmup and evaluation, not exclusive auxiliary memory. Never added to initial-method memory.',
                  'memory_unit': 'MiB (1048576 bytes)', 'time_unit': 'ms',
                  'rounding_tolerance': {'rel_tol': 1e-10, 'abs_tol': 1e-9},
                  'summary': 'Means and sample standard deviations of successful source-run pairs; reduction is relative reduction of means. Check counts for incomplete batches.',
                  'naming': 'Project/Project_InitialAlgorithm+.csv; Algorithm=InitialAlgorithm+; initial SPGreedy+ is excluded.'}
        write_csv(directory / 'paired_runs.csv', AUDIT_FIELDS, [])
    lock = directory / '.running.lock'
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError as error:
        raise RuntimeError(f'Batch is locked: {lock}. Do not run simultaneous resumes; '
                           'after a hard termination, verify the old process has exited before removing its stale lock.') from error
    try:
        with stream:
            json.dump({'pid': os.getpid(), 'hostname': platform.node(),
                       'started_at': datetime.now().astimezone().isoformat()}, stream)
        return execute_batch(directory, config, groups, planned, matrices)
    finally:
        lock.unlink()


def execute_batch(directory, config, groups, planned, matrices):
    settings = config['settings']
    latest = read_journal(directory / 'paired_runs.csv', planned)
    config.update(completed=False, finished=False)
    atomic_text(directory / 'configuration.json', json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(f'Results: {directory}', flush=True)
    failed = interrupted = False
    try:
        with (directory / 'paired_runs.csv').open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, extrasaction='ignore')
            for group in groups:
                for source in group['records']:
                    key = key_of(source)
                    previous = latest.get(key)
                    if previous and previous['Status'] == 'success':
                        continue
                    attempt = int(previous['Attempt']) + 1 if previous else 1
                    meta = matrices[source['Project']]
                    print(f'{source["Project"]} {source["Algorithm"]} Run={source["Run"]} Attempt={attempt}', flush=True)
                    try:
                        if digest(Path(settings['input_root']) / source['SourceFile']) != source['SourceSHA256']:
                            raise RuntimeError('Source CSV changed after preflight.')
                        if code_hashes() != config['code_sha256']:
                            raise RuntimeError('Experiment code changed after preflight.')
                        request = {'project_dir': meta['directory'], 'input_sha256': meta['sha256'],
                                   'algorithm': 'SW-SPDijkstra', 'seed': 0,
                                   'initial_order': source['BeforeCITO'], 'window_size': settings['window_size'],
                                   'warmup_runs': settings['warmup_runs'], 'max_exact_classes': 24}
                        result = run_worker(request, settings['worker_timeout'])
                        if digest(Path(settings['input_root']) / source['SourceFile']) != source['SourceSHA256']:
                            raise RuntimeError('Source CSV changed during optimization.')
                        if code_hashes() != config['code_sha256']:
                            raise RuntimeError('Experiment code changed during optimization.')
                        record = complete_record(source, result, attempt, settings, meta['n_classes'])
                    except Exception as error:
                        record = {**source, 'BeforeCITO': order_json(source['BeforeCITO']),
                                  'Attempt': attempt, 'Status': 'error', 'Error': str(error),
                                  'WindowSize': settings['window_size'], 'WarmupRuns': settings['warmup_runs']}
                        failed = True
                        print(f'ERROR: {error}', file=sys.stderr, flush=True)
                    writer.writerow(record)
                    stream.flush()
                    os.fsync(stream.fileno())
                    latest[key] = record
                    export_group(directory, group, latest)
                    if failed:
                        break
                if failed:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print('Interrupted; recorded successes can be resumed.', file=sys.stderr)
    finally:
        for group in groups:
            export_group(directory, group, latest)
        export_summary(directory, groups, latest)
        successes = sum(row['Status'] == 'success' for row in latest.values())
        config.update(finished=True, completed=successes == len(planned), successful_runs=successes,
                      failed_runs=sum(row['Status'] == 'error' for row in latest.values()),
                      pending_runs=len(planned) - len(latest), interrupted=interrupted,
                      updated_at=datetime.now().astimezone().isoformat())
        atomic_text(directory / 'configuration.json', json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(f'Completed={config["completed"]}; successful={config["successful_runs"]}/{len(planned)}', flush=True)
    return 0 if config['completed'] else 130 if interrupted else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
