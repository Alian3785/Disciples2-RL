"""Ordered, bounded report delivery to a dedicated CPU process.

Only already-transferred host batches cross this pipe. The worker does not import
JAX, run an environment, or receive learner state. Pickle is a private local IPC
protocol between our own processes, never an input file format.
"""
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading
import time
import traceback


class AsyncEpisodeReports:
    """One ordered worker, at most eight queued commands plus one in flight.

    The caller owns submitted host arrays until submission and must not mutate
    them afterwards. Queue saturation applies backpressure; nothing is dropped.
    Always use as a context manager so even a learner error drains accepted data.
    """

    def __init__(self, output, game_map, combat, *, max_pending=8):
        if max_pending < 1:
            raise ValueError('Report queue capacity must be positive')
        started = time.perf_counter()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=max_pending)
        self._error = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._finishing = self._closed = False
        self._expected = self._events = 0
        self.summary = None
        self.worker_seconds = self.queue_wait_seconds = self.drain_seconds = 0.0
        # Start a clean interpreter instead of fork/spawn of the CUDA learner.
        # BLAS is unnecessary for decoding the NumPy host arrays in the pipe.
        environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1'}
        options = ({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt'
                   else {'start_new_session': True})
        # Validate metadata before starting a process that would otherwise await stdin.
        initial = (str(self.output), game_map,
                   {name: combat[name] for name in ('catalogue', 'enemies')})
        self._log = (self.output / 'episode_reports.worker.log').open('ab')
        try:
            self._process = subprocess.Popen(
                [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                env=environment, **options)
        except BaseException:
            self._log.close()
            raise
        # Only the catalogue and rosters are needed; never serialize device arrays.
        self._thread = threading.Thread(target=self._deliver, args=(initial,),
                                        name='episode-report-transport', daemon=True)
        self._thread.start()
        try:
            while not self._ready.wait(0.1):
                self._check()
            self._check()
        except BaseException:
            self._close()
            raise
        self.startup_seconds = time.perf_counter() - started

    def _exchange(self, sequence, command, arguments):
        pickle.dump((sequence, command, arguments), self._process.stdin, protocol=5)
        self._process.stdin.flush()
        reply = pickle.load(self._process.stdout)
        if reply['sequence'] != sequence:
            raise RuntimeError('Out-of-order episode report acknowledgement')
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        self.worker_seconds = reply['worker_seconds']
        return reply.get('summary')

    def _deliver(self, initial):
        try:
            self._exchange(0, 'init', initial)
            self._ready.set()
            sequence = 1
            while True:
                command, arguments = self._queue.get()
                if command == '_stop':
                    break
                result = self._exchange(sequence, command, arguments)
                sequence += 1
                if command == 'finish':
                    self.summary = result
                    if self._process.wait() != 0:
                        raise RuntimeError('Episode report worker exited unsuccessfully')
                    break
        except BaseException as error:
            self._error = error
        finally:
            self._ready.set()
            self._done.set()

    def _check(self):
        if self._error is not None:
            raise RuntimeError(
                f'Episode report worker failed; see {self.output / "episode_reports.worker.log"}'
            ) from self._error
        code = self._process.poll()
        if code is not None and not self._finishing:
            raise RuntimeError(f'Episode report worker stopped unexpectedly (exit {code})')

    def _put(self, command, arguments):
        self._check()
        try:
            self._queue.put_nowait((command, arguments))
        except queue.Full:
            started = time.perf_counter()
            try:
                while True:
                    self._check()
                    try:
                        self._queue.put((command, arguments), timeout=0.1)
                        break
                    except queue.Full:
                        pass
            finally:
                self.queue_wait_seconds += time.perf_counter() - started
        self._check()

    def submit(self, batch, *, transition_base, expected_completed):
        if self._finishing or self._closed:
            raise RuntimeError('Episode report writer is already finishing')
        self._put('batch', (batch, transition_base, expected_completed))
        self._expected = expected_completed
        self._events += len(batch['data'])

    def write_summary(self, *, expected_completed):
        if self._finishing or self._closed:
            raise RuntimeError('Episode report writer is already finishing')
        self._put('summary', (expected_completed,))

    def finish(self):
        if self._closed:
            self._check()
            return self.summary
        started = time.perf_counter()
        try:
            self._check()
            self._finishing = True
            self._put('finish', (self._expected, self._events))
            while not self._done.wait(0.1):
                self._check()
            self._check()
            if self.summary is None:
                raise RuntimeError('Episode report worker did not return a final summary')
            if (self.summary['completed_episodes'] != self._expected
                    or self.summary['events'] != self._events):
                raise RuntimeError('Episode report final coverage mismatch')
            return self.summary
        except BaseException as error:
            if self._error is None:
                self._error = error
            raise
        finally:
            self.drain_seconds += time.perf_counter() - started
            self._close()

    def _close(self):
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()
        try:
            self._queue.put_nowait(('_stop', ()))
        except queue.Full:
            pass  # The transport will observe the closed pipe on its current command.
        self._thread.join()
        for stream in (self._process.stdin, self._process.stdout, self._log):
            try:
                stream.close()
            except OSError:
                pass  # A dead worker can leave buffered pipe bytes; keep the original error.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._closed:
            try:
                self.finish()
            except BaseException as report_error:
                if exc is None:
                    raise
                exc.add_note(f'Additionally, episode report finalization failed: {report_error}')


def _worker():
    # No imports from the training entry point or GPU environment modules.
    from numbergrid_episode_reports import EpisodeReports

    writer = None
    sequence = -1
    expected_sequence = 0
    worker_seconds = 0.0
    try:
        while True:
            sequence, command, arguments = pickle.load(sys.stdin.buffer)
            started = time.perf_counter()
            if sequence != expected_sequence:
                raise RuntimeError('Episode report command sequence mismatch')
            expected_sequence += 1
            summary = None
            if command == 'init' and writer is None:
                writer = EpisodeReports(*arguments)
            elif command == 'batch' and writer is not None:
                batch, transition_base, expected = arguments
                lengths = {len(batch[name]) for name in ('data', 'returns', 'stream', 'offset')}
                if len(lengths) != 1:
                    raise ValueError('Episode report batch fields have different lengths')
                writer.consume(batch, transition_base=transition_base)
                if writer.completed != expected:
                    raise RuntimeError('Episode reports lost a terminal record')
            elif command == 'summary' and writer is not None:
                writer.write_summary(expected_completed=arguments[0])
            elif command == 'finish' and writer is not None:
                expected, events = arguments
                if writer.event_count != events:
                    raise RuntimeError('Episode reports lost or duplicated events')
                summary = writer.write_summary(expected_completed=expected, final=True)
            else:
                raise ValueError(f'Invalid episode report command: {command}')
            worker_seconds += time.perf_counter() - started
            pickle.dump(dict(sequence=sequence, worker_seconds=worker_seconds, summary=summary),
                        sys.stdout.buffer, protocol=5)
            sys.stdout.buffer.flush()
            if command == 'finish':
                return
    except BaseException:
        error = traceback.format_exc()
        # Retain successfully consumed records and partial histories on failure.
        if writer is not None:
            try:
                writer.write_summary(final=True)
                writer._json('episode_reports.failure.json', dict(error=error, sequence=sequence))
            except BaseException:
                error += '\nWhile flushing reports:\n' + traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        try:
            pickle.dump(dict(sequence=sequence, error=error), sys.stdout.buffer, protocol=5)
            sys.stdout.buffer.flush()
        except (BrokenPipeError, OSError):
            pass
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('This module is an internal episode report worker')
    _worker()
