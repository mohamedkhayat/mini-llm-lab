"""Graceful shutdown for the training loop.

The first SIGINT / SIGTERM / SIGHUP sets ``stop.requested``; the loop
finishes the current optimizer step, checkpoints, and exits cleanly. A
second signal restores the previous handlers and re-raises the signal
(``KeyboardInterrupt`` for SIGINT), so a hung run can still be killed
with Ctrl-C twice.

Usage (spec 5.2)::

    with GracefulShutdown() as stop:      # SIGINT, SIGTERM, SIGHUP
        while state.step < total and not stop.requested:
            ...

Installs handlers only in the main thread (``signal.signal`` is
main-thread only); in a worker thread it is a no-op, so tests can run the
loop off-thread. Previous handlers are restored in ``__exit__``, including
on exception.
"""

import os
import signal
import threading


class StopFlag:
    """The object the loop polls: ``stop.requested`` / ``stop.signal_name``."""

    def __init__(self):
        self.requested = False
        self.signal_name: str | None = None


class GracefulShutdown:
    """Context manager mapping the first signal to a graceful stop request."""

    SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")

    def __init__(self):
        self.stop = StopFlag()
        self._previous: dict = {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in self.SIGNALS:
                number = getattr(signal, name, None)
                if number is None:
                    continue
                self._previous[number] = signal.getsignal(number)
                signal.signal(number, self._handle_signal)
        return self.stop

    def __exit__(self, exc_type, exc, tb):
        self._restore()
        return False

    def _handle_signal(self, signum, frame):
        if self.stop.requested:
            # Second signal: abandon the graceful path — restore the
            # previous handlers and let the signal take its normal effect
            # (KeyboardInterrupt for SIGINT, process exit for the rest).
            self._restore()
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            signal.pthread_kill(os.getpid(), signum)
            return
        self.stop.requested = True
        self.stop.signal_name = signal.Signals(signum).name
        print(
            f"Received {self.stop.signal_name}; "
            "checkpointing after the current optimizer step."
        )

    def _restore(self):
        if threading.current_thread() is not threading.main_thread():
            return
        for number, handler in self._previous.items():
            signal.signal(number, handler)
        self._previous.clear()
