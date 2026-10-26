"""HTTP front for the viewer service: mint a session, then reverse-proxy to it.

Wraps a :class:`~seisvtk.viz.sessions.SessionManager` in an aiohttp app a host
runs as the long-lived viewer service. One container serves many viewers:

* ``GET /view?token=…`` — the browser's entry point. The token (minted by the
  host with the shared secret) carries the *scene* to open; this creates a
  session (a trame subprocess on an internal 127.0.0.1 port), pins it to the
  browser with an http-only cookie, and redirects to ``/``.
* Everything else (``/``, ``/ws``, the trame client assets) is **reverse-proxied**
  — HTTP and WebSocket — to the cookie's session, so the browser talks only to
  this service and never to the per-session port directly. This is what makes a
  dynamically-allocated session reachable behind a single ingress host.
* ``POST /sessions`` / ``GET|DELETE /sessions/{id}`` / ``POST …/touch`` — an
  **admin-scoped** management API (create from an explicit scene, inspect,
  keep-alive, close) for trusted callers and tests; a browser's view token does
  not carry that scope.

Authorization is injected (``authorize(request)``): the host decides how a
request is vetted. :func:`jwt_authorizer` is the default — it verifies the
shared-secret token and returns its claims.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web
from multidict import CIMultiDict

from seisvtk.viz.sessions import Session, SessionManager
from seisvtk.viz.tokens import TokenError, verify_hs256

logger = logging.getLogger("seisvtk.viewer")

SESSION_COOKIE = "sv_session"

# Hop-by-hop headers are connection-specific and must not be forwarded by a proxy
# (RFC 7230 §6.1). ``host`` is dropped so aiohttp sets it for the upstream, and
# ``content-length`` because the Response recomputes it from the body.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host",
    "content-length",
})
# Also never leaked to the local worker: the browser's auth material.
_NEVER_FORWARD = frozenset({"cookie", "authorization"})

# A freshly-created session's trame process needs a moment to bind its port;
# retry connection-refused briefly so the first proxied request doesn't 502.
_CONNECT_RETRIES = 40
_CONNECT_DELAY = 0.25

# authorize(request) -> claims/None (optionally awaitable), or raises HTTPException.
Authorize = Callable[[web.Request], "Optional[Awaitable[Any]]"]

_CLIENT = web.AppKey("proxy_client", aiohttp.ClientSession)
_REAPER = web.AppKey("reaper_task", asyncio.Task)
_MANAGER = web.AppKey("manager", SessionManager)


# -- authorization ------------------------------------------------------------
def jwt_authorizer(secret: str, *, leeway: float = 30.0) -> Authorize:
    """An ``authorize`` that verifies a shared-secret HS256 token.

    The token is read from ``?token=`` (browsers cannot set WebSocket headers,
    so a query param is the portable carrier) or an ``Authorization: Bearer``
    header, and must carry an ``exp``. On success the decoded claims are
    returned; on failure a generic 401 is raised (details are logged, not echoed).
    """
    def authorize(request: web.Request) -> dict:
        token = request.query.get("token")
        if not token:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth[len("Bearer "):]
        try:
            return verify_hs256(token or "", secret, leeway=leeway, require_exp=True)
        except TokenError as exc:
            logger.info("viewer token rejected: %s", exc)
            raise web.HTTPUnauthorized(reason="invalid or expired token")

    return authorize


async def _apply(authorize: Optional[Authorize], request: web.Request) -> Any:
    """Run the injected authorizer; return its claims (or ``None`` if unset)."""
    if authorize is None:
        return None
    result = authorize(request)
    if inspect.isawaitable(result):
        result = await result
    return result


def _session_payload(session: Session, public_base: str) -> dict:
    return {
        "id": session.id,
        "port": session.port,
        "scene": session.scene_ref,
        "url": f"{public_base.rstrip('/')}/view/{session.id}" if public_base else None,
    }


def _set_session_cookie(resp: web.StreamResponse, sid: str, request: web.Request) -> None:
    """Pin the browser to *sid*. A cross-site iframe needs SameSite=None+Secure,
    which browsers only honour over https; fall back to Lax for plain-http dev."""
    secure = request.secure or \
        request.headers.get("X-Forwarded-Proto", "").lower() == "https"
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, path="/",
                    samesite="None" if secure else "Lax", secure=secure)


# -- reverse proxy ------------------------------------------------------------
def _forward_headers(request: web.Request) -> dict:
    drop = set(_HOP_BY_HOP) | set(_NEVER_FORWARD)
    conn = request.headers.get("Connection", "")
    drop |= {t.strip().lower() for t in conn.split(",") if t.strip()}
    return {k: v for k, v in request.headers.items() if k.lower() not in drop}


async def _open_http(client: aiohttp.ClientSession, method: str, url: str, **kw):
    """Issue an upstream request, retrying connection-refused during trame start."""
    last: Optional[Exception] = None
    for _ in range(_CONNECT_RETRIES):
        try:
            return await client.request(method, url, allow_redirects=False, **kw)
        except aiohttp.ClientConnectorError as exc:
            last = exc
            await asyncio.sleep(_CONNECT_DELAY)
    raise web.HTTPBadGateway(reason="viewer session not reachable")


async def _open_ws(client: aiohttp.ClientSession, url: str, protocols):
    last: Optional[Exception] = None
    for _ in range(_CONNECT_RETRIES):
        try:
            return await client.ws_connect(url, protocols=protocols, max_msg_size=0)
        except aiohttp.ClientConnectorError as exc:
            last = exc
            await asyncio.sleep(_CONNECT_DELAY)
    assert last is not None
    raise last


async def _proxy_ws(request: web.Request, ws_url: str, sid: Optional[str]) -> web.StreamResponse:
    """Bridge the browser WebSocket to the session's, relaying both directions."""
    proto = request.headers.get("Sec-WebSocket-Protocol")
    protocols = [p.strip() for p in proto.split(",")] if proto else ()
    # max_msg_size=0: trame streams rendered frames as large binary messages that
    # exceed aiohttp's 4 MB default, which would otherwise drop the connection.
    ws_server = web.WebSocketResponse(protocols=protocols, max_msg_size=0)
    await ws_server.prepare(request)
    client: aiohttp.ClientSession = request.app[_CLIENT]
    manager: SessionManager = request.app[_MANAGER]
    try:
        ws_up = await _open_ws(client, ws_url, protocols)
    except aiohttp.ClientError:
        await ws_server.close(code=aiohttp.WSCloseCode.TRY_AGAIN_LATER,
                              message=b"viewer session unavailable")
        return ws_server

    async def pump(src, dst):
        async for msg in src:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await dst.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await dst.send_bytes(msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                              aiohttp.WSMsgType.ERROR):
                break
            if sid:                         # an open, active socket keeps the session alive
                manager.touch(sid)

    tasks = [asyncio.ensure_future(pump(ws_up, ws_server)),
             asyncio.ensure_future(pump(ws_server, ws_up))]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await ws_up.close()
        await ws_server.close()
    return ws_server


async def _proxy(request: web.Request, session: Session) -> web.StreamResponse:
    """Forward *request* to *session*'s trame process (HTTP, or WebSocket upgrade).

    The target is always ``127.0.0.1:<session port>`` + the request's own
    (already-encoded) ``raw_path`` — the client cannot steer host, port or path."""
    sid = session.id
    base = f"127.0.0.1:{session.port}"
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await _proxy_ws(request, f"ws://{base}{request.raw_path}", sid)
    url = f"http://{base}{request.raw_path}"
    body = await request.read()
    client: aiohttp.ClientSession = request.app[_CLIENT]
    try:
        up = await _open_http(client, request.method, url,
                              headers=_forward_headers(request), data=body)
    except aiohttp.ClientError as exc:
        logger.info("upstream proxy error: %s", exc)
        raise web.HTTPBadGateway(reason="viewer session error")
    try:
        out = CIMultiDict((k, v) for k, v in up.headers.items()
                          if k.lower() not in _HOP_BY_HOP)
        return web.Response(status=up.status, body=await up.read(), headers=out)
    finally:
        up.release()


def make_app(manager: SessionManager, *, authorize: Optional[Authorize] = None,
             public_base: str = "", reap_interval: float = 60.0) -> web.Application:
    """Build the viewer-service aiohttp app around *manager*.

    ``authorize`` vets the ``/view`` entry and the admin API (default: allow —
    only safe behind a trusted network). ``public_base`` is the externally
    reachable base URL; when set, its origin is enforced on WebSocket upgrades
    (cross-site WebSocket-hijacking defence). A background task reaps idle/dead
    sessions every ``reap_interval`` seconds.
    """
    allowed_origin = None
    if public_base:
        parts = urlsplit(public_base)
        if parts.scheme and parts.netloc:
            allowed_origin = f"{parts.scheme}://{parts.netloc}"

    async def _require_admin(request: web.Request) -> None:
        claims = await _apply(authorize, request)
        if authorize is not None and (claims or {}).get("scope") != "admin":
            raise web.HTTPForbidden(reason="admin scope required")

    async def healthz(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "sessions": len(manager)})

    async def view(request: web.Request) -> web.Response:
        """Browser entry: token -> new session -> cookie -> redirect to the viewer."""
        claims = await _apply(authorize, request) or {}
        scene = claims.get("scene")
        if not isinstance(scene, str) or not scene:
            raise web.HTTPBadRequest(reason="token does not carry a scene")
        previous = request.cookies.get(SESSION_COOKIE)
        if previous:                         # don't orphan a prior session for this browser
            manager.close(previous)
        try:
            session = manager.create(scene)
        except RuntimeError:
            raise web.HTTPServiceUnavailable(reason="viewer capacity reached")
        except OSError as exc:
            logger.warning("failed to launch viewer session: %s", exc)
            raise web.HTTPBadGateway(reason="could not start viewer")
        resp = web.HTTPFound("/")
        resp.headers["Cache-Control"] = "no-store"     # token is in the entry URL
        resp.headers["Referrer-Policy"] = "no-referrer"
        _set_session_cookie(resp, session.id, request)
        raise resp

    async def create_session(request: web.Request) -> web.Response:
        await _require_admin(request)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - malformed body -> 400
            raise web.HTTPBadRequest(reason="body must be JSON")
        scene = (body or {}).get("scene")
        if not isinstance(scene, str) or not scene:
            raise web.HTTPBadRequest(reason="missing or non-string 'scene'")
        try:
            session = manager.create(scene)
        except RuntimeError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc))
        except OSError as exc:
            raise web.HTTPBadGateway(reason=f"could not start viewer: {exc}")
        return web.json_response(_session_payload(session, public_base), status=201)

    async def touch_session(request: web.Request) -> web.Response:
        await _require_admin(request)
        if not manager.touch(request.match_info["id"]):
            raise web.HTTPNotFound(reason="no such session")
        return web.json_response({"ok": True})

    async def get_session(request: web.Request) -> web.Response:
        await _require_admin(request)
        session = manager.get(request.match_info["id"])
        if session is None:
            raise web.HTTPNotFound(reason="no such session")
        return web.json_response(_session_payload(session, public_base))

    async def close_session(request: web.Request) -> web.Response:
        await _require_admin(request)
        manager.close(request.match_info["id"])
        return web.json_response({"ok": True})

    async def proxy_handler(request: web.Request) -> web.StreamResponse:
        """Catch-all: route the browser to its cookie-pinned session."""
        is_ws = request.headers.get("Upgrade", "").lower() == "websocket"
        if is_ws and allowed_origin:                   # block cross-site WS hijacking
            origin = request.headers.get("Origin")
            if origin is not None and origin != allowed_origin:
                raise web.HTTPForbidden(reason="origin not allowed")
        sid = request.cookies.get(SESSION_COOKIE)
        session = manager.get(sid) if sid else None
        if session is None:
            raise web.HTTPNotFound(reason="no active viewer session")
        manager.touch(sid)
        return await _proxy(request, session)

    app = web.Application()
    app.add_routes([
        web.get("/healthz", healthz),
        web.get("/view", view),
        web.post("/sessions", create_session),
        web.get("/sessions/{id}", get_session),
        web.post("/sessions/{id}/touch", touch_session),
        web.delete("/sessions/{id}", close_session),
        web.route("*", "/{tail:.*}", proxy_handler),   # reverse proxy (registered last)
    ])

    async def _on_startup(app: web.Application) -> None:
        app[_MANAGER] = manager
        # auto_decompress off: pass bytes + Content-Encoding through verbatim.
        # A short sock_connect bound surfaces a dead worker quickly; no total cap
        # so a long-lived streaming connection is not killed.
        app[_CLIENT] = aiohttp.ClientSession(
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=5))

        async def reaper() -> None:
            while True:
                await asyncio.sleep(reap_interval)
                try:
                    manager.reap_idle()
                except Exception:  # noqa: BLE001 - a reap error must not kill the loop
                    logger.exception("idle-session reap failed")
        app[_REAPER] = asyncio.create_task(reaper())

    async def _on_cleanup(app: web.Application) -> None:
        task = app.get(_REAPER)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        client = app.get(_CLIENT)
        if client is not None:
            await client.close()
        manager.shutdown()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def main(argv=None) -> None:
    """Run the viewer service from environment configuration (the image CMD).

    ``SEISVTK_VIEWER_SECRET`` enables token auth and is **required** unless
    ``SEISVTK_VIEWER_ALLOW_INSECURE`` is set (local dev only — the service binds
    0.0.0.0). ``SEISVTK_VIEWER_PUBLIC_BASE`` is the external URL; plus
    idle-timeout / max-sessions knobs.
    """
    logging.basicConfig(level=logging.INFO)
    secret = os.environ.get("SEISVTK_VIEWER_SECRET")
    allow_insecure = os.environ.get("SEISVTK_VIEWER_ALLOW_INSECURE", "").lower() \
        in ("1", "true", "yes")
    if not secret and not allow_insecure:
        raise SystemExit(
            "SEISVTK_VIEWER_SECRET is required; set SEISVTK_VIEWER_ALLOW_INSECURE=true "
            "to run without token auth (local dev only).")

    public_base = os.environ.get("SEISVTK_VIEWER_PUBLIC_BASE", "")
    port = int(os.environ.get("PORT", os.environ.get("SEISVTK_VIEWER_PORT", "8914")))
    idle = float(os.environ.get("SEISVTK_VIEWER_IDLE_TIMEOUT", "900"))
    max_sessions = int(os.environ.get("SEISVTK_VIEWER_MAX_SESSIONS", "8"))

    manager = SessionManager(idle_timeout=idle, max_sessions=max_sessions)
    authorize = jwt_authorizer(secret) if secret else None
    if authorize is None:
        logger.warning("SEISVTK_VIEWER_SECRET unset — token auth DISABLED (insecure)")
    app = make_app(manager, authorize=authorize, public_base=public_base)
    logger.info("viewer service on :%s (idle=%ss, max=%s)", port, idle, max_sessions)
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
