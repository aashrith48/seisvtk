"""Per-session viewer lifecycle: launch, keep-alive, reap, cap, teardown.

Uses a fake process + an injected clock and port allocator, so no trame server
or real socket is involved.
"""
import itertools

import pytest

from seisvtk.viz.sessions import SessionManager


class FakeProc:
    """Stand-in for a Popen: alive until terminated, or pre-set as exited."""

    def __init__(self, returncode=None):
        self._rc = returncode

    def poll(self):
        return self._rc

    def terminate(self):
        self._rc = -15

    def kill(self):
        self._rc = -9


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _manager(**kw):
    ports = itertools.count(8920)
    launched = []

    def launch(scene_ref, port):
        proc = FakeProc()
        launched.append((scene_ref, port, proc))
        return proc

    clock = Clock()
    mgr = SessionManager(launch=launch, allocate_port=lambda: next(ports),
                         clock=clock, **kw)
    return mgr, clock, launched


def test_create_launches_and_tracks():
    mgr, _, launched = _manager()
    s = mgr.create("scene.json")
    assert len(mgr) == 1
    assert launched[0][0] == "scene.json" and launched[0][1] == s.port
    assert mgr.get(s.id) is s and s.alive()


def test_concurrent_sessions_get_distinct_ports():
    mgr, _, _ = _manager()
    a = mgr.create("a.json")
    b = mgr.create("b.json")
    assert a.port != b.port
    assert len(mgr) == 2


def test_dead_process_is_dropped_on_get():
    mgr, _, launched = _manager()
    s = mgr.create("scene.json")
    launched[0][2]._rc = 0                     # the viewer process exited
    assert mgr.get(s.id) is None
    assert len(mgr) == 0


def test_idle_session_is_reaped_but_touch_keeps_it():
    mgr, clock, _ = _manager(idle_timeout=300.0)
    s = mgr.create("scene.json")
    clock.advance(200.0)
    mgr.touch(s.id)                            # still active
    clock.advance(200.0)                       # 200s since touch < 300s
    assert mgr.reap_idle() == []
    assert mgr.get(s.id) is s
    clock.advance(400.0)                       # now well past the timeout
    assert mgr.reap_idle() == [s.id]
    assert mgr.get(s.id) is None


def test_max_sessions_enforced_after_reap():
    mgr, _, launched = _manager(max_sessions=2)
    a = mgr.create("a.json")
    mgr.create("b.json")
    with pytest.raises(RuntimeError):
        mgr.create("c.json")
    # once one dies, create reaps it and succeeds
    launched[0][2]._rc = 0                     # 'a' exited
    c = mgr.create("c.json")
    assert c.alive() and mgr.get(a.id) is None


def test_close_and_shutdown_terminate():
    mgr, _, launched = _manager()
    s = mgr.create("scene.json")
    assert mgr.close(s.id) is True
    assert launched[0][2].poll() == -15        # terminated
    assert mgr.close(s.id) is False            # already gone
    mgr.create("x.json")
    mgr.create("y.json")
    mgr.shutdown()
    assert len(mgr) == 0
