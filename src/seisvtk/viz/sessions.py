"""Per-session lifecycle for the viewer service.

A streaming VTK viewer renders with one GL context per viewer, so the service
runs **one trame process per session** rather than sharing a renderer. This
manager owns that fleet: it launches a process bound to a scene manifest on its
own port, tracks it, keeps it alive while the viewer is active, and reaps it when
the viewer goes away (idle past a timeout, or the process died). It is render- and
transport-agnostic — the process launcher and the clock are injected — so the
logic is testable without actually starting trame or binding real sockets, and
the HTTP front that creates/proxies sessions layers on top.
"""
from __future__ import annotations

import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _os_free_port() -> int:
    """An ephemeral port the OS reports free (bind to 0, read it back, release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _default_launch(scene_ref: str, port: int) -> Any:
    """Start a ``seisvtk-viewer`` process serving *scene_ref* on *port*."""
    import subprocess

    return subprocess.Popen(
        [sys.executable, "-m", "seisvtk.viz.service", scene_ref,
         "--port", str(port), "--host", "0.0.0.0"])


@dataclass
class Session:
    """One running viewer process for one scene."""

    id: str
    port: int
    scene_ref: str
    proc: Any                       # anything with poll()/terminate()/kill()
    created: float
    last_seen: float = field(default=0.0)

    def alive(self) -> bool:
        """True while the underlying process is still running."""
        return self.proc.poll() is None


class SessionManager:
    """Launch, track, keep-alive and reap one viewer process per session.

    ``launch(scene_ref, port) -> proc`` starts a viewer (default: a
    ``seisvtk-viewer`` subprocess); ``allocate_port`` hands out a free port
    (default: an OS ephemeral port); ``clock`` is the monotonic time source.
    Sessions idle longer than ``idle_timeout`` seconds, or whose process has
    exited, are torn down; ``max_sessions`` caps how many run at once (a reap is
    attempted first, then it raises).
    """

    def __init__(self, *, launch: Callable[[str, int], Any] = _default_launch,
                 allocate_port: Optional[Callable[[], int]] = None,
                 idle_timeout: float = 900.0, max_sessions: int = 8,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._launch = launch
        self._allocate_port = allocate_port or _os_free_port
        self._idle_timeout = float(idle_timeout)
        self._max_sessions = int(max_sessions)
        self._clock = clock
        self._sessions: dict[str, Session] = {}

    # -- queries ---------------------------------------------------------------
    def get(self, session_id: str) -> Optional[Session]:
        """The live session for *session_id*, or ``None`` (dropping a dead one)."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if not session.alive():
            self._drop(session)
            return None
        return session

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def sessions(self) -> list[Session]:
        return list(self._sessions.values())

    # -- lifecycle -------------------------------------------------------------
    def create(self, scene_ref: str) -> Session:
        """Launch a viewer for *scene_ref* and return its :class:`Session`.

        Reaps dead/idle sessions first; raises ``RuntimeError`` if the fleet is
        still at ``max_sessions`` afterwards.
        """
        self.reap_idle()
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(
                f"viewer session limit reached ({self._max_sessions})")
        now = self._clock()
        port = self._allocate_port()
        proc = self._launch(scene_ref, port)
        session = Session(id=uuid.uuid4().hex, port=port, scene_ref=scene_ref,
                          proc=proc, created=now, last_seen=now)
        self._sessions[session.id] = session
        return session

    def touch(self, session_id: str) -> bool:
        """Mark a session active (resets its idle clock). False if it's gone."""
        session = self.get(session_id)
        if session is None:
            return False
        session.last_seen = self._clock()
        return True

    def reap_idle(self) -> list[str]:
        """Tear down sessions that have exited or been idle too long.

        Returns the ids reaped. Idle is measured from ``last_seen``.
        """
        now = self._clock()
        doomed = [s for s in self._sessions.values()
                  if not s.alive() or (now - s.last_seen) > self._idle_timeout]
        for session in doomed:
            self._drop(session)
        return [s.id for s in doomed]

    def close(self, session_id: str) -> bool:
        """Terminate and forget one session. False if it was not tracked."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        self._drop(session)
        return True

    def shutdown(self) -> None:
        """Terminate every session (service teardown)."""
        for session in list(self._sessions.values()):
            self._drop(session)

    # -- internals -------------------------------------------------------------
    def _drop(self, session: Session) -> None:
        try:
            if session.proc.poll() is None:
                session.proc.terminate()
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
        self._sessions.pop(session.id, None)
