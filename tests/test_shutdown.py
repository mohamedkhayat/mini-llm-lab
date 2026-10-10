"""T1.1: GracefulShutdown — first signal requests a stop, second re-raises."""

import os
import signal
import time

import pytest

from training.shutdown import GracefulShutdown


def _wait(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the signal handler")
        time.sleep(0.005)


def test_shutdown_first_and_second_signal(capsys):
    """A real first SIGINT sets the stop flag (no re-raise); the second
    signal restores the previous handlers and re-raises KeyboardInterrupt;
    __exit__ restores the original handlers."""
    gs = GracefulShutdown()
    with gs as stop:
        assert stop.requested is False
        assert stop.signal_name is None

        os.kill(os.getpid(), signal.SIGINT)
        _wait(lambda: stop.requested)
        assert stop.signal_name == "SIGINT"
        assert "SIGINT" in capsys.readouterr().out

        # Second signal: previous handlers restored, KeyboardInterrupt
        # re-raised (deterministic: drive the handler directly).
        with pytest.raises(KeyboardInterrupt):
            gs._handle_signal(signal.SIGINT, None)

    # __exit__ restored the original handlers even after the second-signal
    # path already restored them (restore is idempotent).
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_shutdown_is_a_noop_off_the_main_thread():
    """Off the main thread no handlers are installed (signal.signal is
    main-thread only) and exit restores nothing; the flag is still usable
    as a plain stop request."""
    import threading

    result = {}

    def worker():
        gs = GracefulShutdown()
        with gs as stop:
            stop.requested = True
            result["installed"] = len(gs._previous)
            result["requested"] = stop.requested

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result == {"installed": 0, "requested": True}
