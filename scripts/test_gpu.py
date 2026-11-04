"""Run a regression suite on CUDA, one pytest process per test file.

Usage: bash scripts/run_gpu.sh scripts/test_gpu.py --output results/gpu-tests/RUN [--full]
The core set must finish within ten minutes; --full runs every test within an hour and
is used only on the user's direct instruction. Reuse the local JAX compilation cache by
default; --cold-cache measures a fresh compile. All fixtures, cache loading and
compilation stay inside the timed region.
"""
import argparse
from contextlib import suppress
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib
from xml.etree import ElementTree

WALL_LIMITS = dict(core=600,full=3600)
SELECTIONS = dict(core='core',full='core or not core')
# Compilation takes most of the time and runs on the CPU while the GPU stays mostly
# idle, so processes share it well. Each one also adds up to 3 GB of host memory.
WORKERS = 3
SHARED_FILES = 8


def test_batches(root,suite):
    """Core files on the session's current map share a process and trace it once; others run alone."""
    paths = tomllib.loads((root/'pyproject.toml').read_text())['tool']['pytest']['ini_options']['testpaths']
    everything = [path.relative_to(root).as_posix() for path in (root/'stoix/tests').glob('*_test.py')]
    marked = {path for path in everything if 'pytest.mark.core' in (root/path).read_text(encoding='utf-8')}
    if len(paths) != len(set(paths)) or set(paths) != marked:
        raise SystemExit(f'testpaths must list exactly the files with core tests; differs by {sorted(set(paths)^marked)}')
    # Large files first keep the processes evenly loaded until the end.
    if suite == 'full':
        # Shared processes outgrow host memory here: 7.8 GB plus all swap, for a 3% gain.
        return [[path] for path in sorted(everything,key=lambda path:(root/path).stat().st_size,reverse=True)]
    paths = sorted(paths,key=lambda path:(root/path).stat().st_size,reverse=True)
    shared = [path for path in paths if 'current_game' in (root/path).read_text(encoding='utf-8')]
    return [[path] for path in paths if path not in shared]+[
        shared[i:i+SHARED_FILES] for i in range(0,len(shared),SHARED_FILES)]


def read_report(path):
    """Count a finished batch's cases and name its failures from the JUnit report."""
    tests,skipped,failed = 0,0,[]
    if path.exists():
        for case in ElementTree.parse(path).getroot().iter('testcase'):
            tests += 1
            skipped += case.find('skipped') is not None
            if case.find('failure') is not None or case.find('error') is not None:
                failed.append(case.get('classname')+'::'+case.get('name'))
    return dict(tests=tests,skipped=skipped,failed=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full',action='store_true',help='Run every test, only on the user\'s direct instruction')
    parser.add_argument('--workers',type=int,default=WORKERS,help='Pytest processes running at the same time')
    parser.add_argument('--cold-cache',action='store_true',help='Use an empty, isolated compilation cache for this run')
    parser.add_argument('--wall-started-at',type=float,default=os.environ.get('NUMBERGRID_TEST_STARTED_AT'),
                        help='Parent launch epoch seconds, including WSL startup')
    args = parser.parse_args()
    if args.wall_started_at is not None and (not math.isfinite(args.wall_started_at) or args.wall_started_at > time.time()+1):
        parser.error('--wall-started-at must be a finite timestamp no later than this launch')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if os.environ.get('JAX_PLATFORMS') != 'cuda':
        raise SystemExit('Run through scripts/run_gpu.sh; CUDA is mandatory.')
    root = Path(__file__).resolve().parents[1]
    suite = 'full' if args.full else 'core'
    wall_limit = WALL_LIMITS[suite]
    pending = test_batches(root,suite)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    for name in ('logs','junit','durations'):
        (output/name).mkdir()
    started = time.monotonic()
    # Reuse compiled executables, never test results. JAX keys the cache by
    # computation, compiler configuration and device; changed graphs recompile.
    # Separate processes share even small kernels through it, hence no minimum time.
    cache = output/'compilation-cache' if args.cold_cache else root/'results/gpu-tests/compilation-cache'
    cache.mkdir(parents=True,exist_ok=True)
    cache_populated = any(cache.glob('*-cache'))
    worker_env = {**os.environ,'NUMBERGRID_TEST_RUN':'1','JAX_ENABLE_COMPILATION_CACHE':'true',
                  'JAX_COMPILATION_CACHE_DIR':str(cache),
                  'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS':'0'}
    workers = {}
    results = {}
    # A parent timestamp measures the actual end-to-end command. Keep six
    # seconds for termination/reporting. Native WSL invocations use the same
    # budget; Windows launchers can include their startup with the timestamp.
    execution_limit = (wall_limit-6. if args.wall_started_at is None else
                       max(0.,wall_limit-6.-(time.time()-args.wall_started_at)))
    deadline = started+execution_limit
    wall_deadline = time.time()+execution_limit
    total_batches = len(pending)
    try:
        while pending or workers:
            for name,(worker,stream,began,batch) in list(workers.items()):
                code = worker.poll()
                if code is None:
                    continue
                stream.close()
                del workers[name]
                results[name] = dict(exit_code=code,seconds=time.monotonic()-began,test_files=batch,
                                     log=str(output/'logs'/(name+'.log')),**read_report(output/'junit'/(name+'.xml')))
                print(json.dumps(dict(phase='finished',batch=name,done=len(results),of=total_batches,
                                      elapsed=time.monotonic()-started,**results[name])),flush=True)
            while pending and len(workers) < args.workers:
                batch = pending.pop(0)
                name = Path(batch[0]).stem
                stream = (output/'logs'/(name+'.log')).open('w')
                command = [sys.executable,'-m','pytest','-v','-m',SELECTIONS[suite],f'--junitxml={output/"junit"/(name+".xml")}',*batch]
                env = {**worker_env,'NUMBERGRID_TEST_DURATIONS':str(output/'durations'/(name+'.jsonl'))}
                worker = subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                workers[name] = worker,stream,time.monotonic(),batch
            if not workers:
                break
            if time.monotonic() >= deadline or time.time() >= wall_deadline:
                print(f'The {suite} suite exceeded its {execution_limit:.1f}s execution budget.',flush=True)
                break
            time.sleep(.25)
    finally:
        # Terminate only the process groups created above, never other GPU work.
        for worker,*_ in workers.values():
            if worker.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid,signal.SIGTERM)
        for worker,stream,*_ in workers.values():
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid,signal.SIGKILL)
                worker.wait()
            stream.close()
        total_seconds = time.monotonic()-started if args.wall_started_at is None else time.time()-args.wall_started_at
        failed = [case for result in results.values() for case in result['failed']]
        summary = dict(seconds=time.monotonic()-started,wall_seconds=total_seconds,execution_limit=execution_limit,wall_limit=wall_limit,
                       parent_started_at=args.wall_started_at,backend='cuda',suite=suite,workers=args.workers,
                       batches=results,tests=sum(r['tests'] for r in results.values()),
                       skipped=sum(r['skipped'] for r in results.values()),failed=failed,
                       compilation_cache=str(cache),cold_cache=not cache_populated,
                       isolated_cache=args.cold_cache,
                       selection=SELECTIONS[suite],durations=str(output/'durations'),
                       passed=total_seconds <= wall_limit and len(results)==total_batches and all(r['exit_code']==0 for r in results.values()))
        (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({key:value for key,value in summary.items() if key != 'batches'}),flush=True)
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
