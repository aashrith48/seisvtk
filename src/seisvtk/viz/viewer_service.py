"""HTTP front for the viewer service: create / keep-alive / close sessions.

Wraps a :class:`~seisvtk.viz.sessions.SessionManager` in a small aiohttp app a
host can run as the viewer service. A caller (e.g. the QEarth backend, after it
has authorized the user for a chart) POSTs a scene reference and gets back a
session id + the port its trame process is on; the browser then connects to that
session to receive the streamed frames.

Authorization is injected (``authorize(request)``) so the transport stays policy
free — the host decides how a request is vetted (a signed token, a shared secret).

Out of scope here, by design: **proxying** the browser's HTTP/WebSocket traffic to
the per-session trame port. That is deployment-coupled (an ingress path/host rule,
or a dedicated reverse proxy) and is wired at deploy time; this app manages the
session fleet and reports where each one lives.
"""
from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Optional

from aiohttp import web

from seisvtk.viz.sessions import Session, SessionManager

# authorize(request) -> None/ok, or raises aiohttp.web.HTTPException (e.g. 401).
Authorize = Callable[[web.Request], "Optional[Awaitable[Any]]"]


async def _apply(authorize: Optional[Authorize], request: web.Request) -> None:
    if authorize is None:
        return
    result = authorize(request)
    if inspect.isawaitable(result):
        await result


def _session_payload(session: Session, public_base: str) -> dict:
    return {
        "id": session.id,
        "port": session.port,
        "scene": session.scene_ref,
        "url": f"{public_base.rstrip('/')}/view/{session.id}" if public_base else None,
    }


def make_app(manager: SessionManager, *, authorize: Optional[Authorize] = None,
             public_base: str = "") -> web.Application:
    """Build the viewer-service aiohttp app around *manager*.

    ``authorize`` vets each mutating request (default: allow). ``public_base`` is
    the externally reachable base URL, used to report a ``url`` for each session;
    leave empty if routing is resolved elsewhere.
    """

    async def healthz(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "sessions": len(manager)})

    async def create_session(request: web.Request) -> web.Response:
        await _apply(authorize, request)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - malformed body -> 400
            raise web.HTTPBadRequest(reason="body must be JSON")
        scene = (body or {}).get("scene")
        if not scene:
            raise web.HTTPBadRequest(reason="missing 'scene'")
        try:
            session = manager.create(scene)
        except RuntimeError as exc:                    # fleet at capacity
            raise web.HTTPServiceUnavailable(reason=str(exc))
        return web.json_response(_session_payload(session, public_base), status=201)

    async def touch_session(request: web.Request) -> web.Response:
        await _apply(authorize, request)
        if not manager.touch(request.match_info["id"]):
            raise web.HTTPNotFound(reason="no such session")
        return web.json_response({"ok": True})

    async def get_session(request: web.Request) -> web.Response:
        session = manager.get(request.match_info["id"])
        if session is None:
            raise web.HTTPNotFound(reason="no such session")
        return web.json_response(_session_payload(session, public_base))

    async def close_session(request: web.Request) -> web.Response:
        await _apply(authorize, request)
        manager.close(request.match_info["id"])
        return web.json_response({"ok": True})

    app = web.Application()
    app.add_routes([
        web.get("/healthz", healthz),
        web.post("/sessions", create_session),
        web.get("/sessions/{id}", get_session),
        web.post("/sessions/{id}/touch", touch_session),
        web.delete("/sessions/{id}", close_session),
    ])
    app.on_cleanup.append(lambda _app: _shutdown(manager))
    return app


async def _shutdown(manager: SessionManager) -> None:
    manager.shutdown()
