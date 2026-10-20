"""Run the mandatory regression suite in one CUDA process within ten minutes.

Usage: bash scripts/run_gpu.sh scripts/test_gpu.py --output results/gpu-tests/RUN
Reuse the local JAX compilation cache by default; --cold-cache measures a fresh
compile. All fixtures, cache loading and compilation stay inside the timed region.
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

WALL_LIMIT = 600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cold-cache',action='store_true',help='Use an empty, isolated compilation cache for this run')
    parser.add_argument('--wall-started-at',type=float,default=os.environ.get('NUMBERGRID_TEST_STARTED_AT'),
                        help='Parent launch epoch seconds, including WSL startup')
    args = parser.parse_args()
    if args.wall_started_at is not None and (not math.isfinite(args.wall_started_at) or args.wall_started_at > time.time()+1):
        parser.error('--wall-started-at must be a finite timestamp no later than this launch')
    if os.environ.get('JAX_PLATFORMS') != 'cuda':
        raise SystemExit('Run through scripts/run_gpu.sh; CUDA is mandatory.')
    root = Path(__file__).resolve().parents[1]
    paths = tomllib.loads((root/'pyproject.toml').read_text())['tool']['pytest']['ini_options']['testpaths']
    actions = 'stoix/tests/number_grid_reference_actions_test.py'
    if actions not in paths or len(paths) != len(set(paths)):
        raise SystemExit('Unexpected testpaths; refuse an incomplete/duplicate suite.')
    groups = {'suite': paths}
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    # Reuse compiled executables, never test results. JAX keys the cache by
    # computation, compiler configuration and device; changed graphs recompile.
    cache = output/'compilation-cache' if args.cold_cache else root/'results/gpu-tests/compilation-cache'
    cache.mkdir(parents=True,exist_ok=True)
    cache_populated = any(cache.glob('*-cache'))
    worker_env = {**os.environ,'NUMBERGRID_TEST_RUN':'1','JAX_ENABLE_COMPILATION_CACHE':'true',
                  'JAX_COMPILATION_CACHE_DIR':str(cache),
                  'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS':'1',
                  'NUMBERGRID_TEST_DURATIONS':str(output/'durations.jsonl')}
    workers = {}
    streams = []
    results = {}
    # A parent timestamp measures the actual end-to-end command. Keep six
    # seconds for termination/reporting. Native WSL invocations use the same
    # budget; Windows launchers can include their startup with the timestamp.
    execution_limit = (WALL_LIMIT-6. if args.wall_started_at is None else
                       max(0.,WALL_LIMIT-6.-(time.time()-args.wall_started_at)))
    deadline = started+execution_limit
    wall_deadline = time.time()+execution_limit
    try:
        for name,files in groups.items():
            stream = (output/(name+'.log')).open('w')
            streams.append(stream)
            command = [sys.executable,'-m','pytest','-v','-m','core','--maxfail=1',
                       '--durations=20',f'--junitxml={output / "junit.xml"}',*files]
            worker = subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=worker_env)
            workers[name] = worker
            print(json.dumps(dict(phase='started',group=name,pid=worker.pid,testpaths=files)),flush=True)
        while len(results) < len(workers):
            for name,worker in workers.items():
                code = worker.poll()
                if code is not None and name not in results:
                    results[name] = dict(exit_code=code,seconds=time.monotonic()-started,log=str(output/(name+'.log')))
                    print(json.dumps(dict(phase='finished',group=name,**results[name])),flush=True)
            if len(results) == len(workers):
                break
            if time.monotonic() >= deadline or time.time() >= wall_deadline:
                print(f'Full CUDA suite exceeded its {execution_limit:.1f}s execution budget.',flush=True)
                break
            time.sleep(.25)
    finally:
        # Terminate only the process groups created above, never other GPU work.
        for worker in workers.values():
            if worker.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid,signal.SIGTERM)
        for worker in workers.values():
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid,signal.SIGKILL)
                worker.wait()
        for stream in streams:
            stream.close()
        total_seconds = time.monotonic()-started if args.wall_started_at is None else time.time()-args.wall_started_at
        summary = dict(seconds=time.monotonic()-started,wall_seconds=total_seconds,execution_limit=execution_limit,wall_limit=WALL_LIMIT,
                       parent_started_at=args.wall_started_at,backend='cuda',groups=results,
                       compilation_cache=str(cache),cold_cache=not cache_populated,
                       isolated_cache=args.cold_cache,
                       selection='core',durations=str(output/'durations.jsonl'),
                       passed=total_seconds <= WALL_LIMIT and len(results)==len(groups) and all(r['exit_code']==0 for r in results.values()))
        (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
