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


def test_second_signal_restores_handlers_then_uses_os_kill(monkeypatch):
    """F5: a second non-INT signal re-raises via
    os.kill(os.getpid(), signum) AFTER the previous handlers are
    restored (the old pthread_kill passed a pid where a thread id
    belongs). A first signal never calls os.kill, and a second SIGINT
    still raises KeyboardInterrupt without it."""
    order = []
    monkeypatch.setattr(
        os, "kill",
        lambda pid, signum: order.append(("kill", pid, signum)),
    )
    real_signal = signal.signal

    def spy_signal(num, handler):
        order.append(("handler", num, handler))
        return real_signal(num, handler)

    monkeypatch.setattr(signal, "signal", spy_signal)

    gs = GracefulShutdown()
    with gs as stop:
        gs._handle_signal(signal.SIGTERM, None)  # first: graceful
        assert stop.requested is True
        assert not [e for e in order if e[0] == "kill"], (
            "a first signal never calls os.kill"
        )

        gs._handle_signal(signal.SIGTERM, None)  # second: force kill
        kills = [e for e in order if e[0] == "kill"]
        assert kills == [("kill", os.getpid(), signal.SIGTERM)]
        # The kill is the LAST event: _restore() ran first.
        assert order.index(kills[0]) == len(order) - 1

    order.clear()
    gs2 = GracefulShutdown()
    with gs2 as stop:
        gs2._handle_signal(signal.SIGINT, None)  # first: graceful
        with pytest.raises(KeyboardInterrupt):
            gs2._handle_signal(signal.SIGINT, None)  # second: re-raise
    assert not [e for e in order if e[0] == "kill"], (
        "a second SIGINT raises KeyboardInterrupt, not os.kill"
    )


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
