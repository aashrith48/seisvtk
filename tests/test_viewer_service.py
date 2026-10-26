"""The viewer-service HTTP app: session create/get/touch/close + auth hook.

Driven without pytest-asyncio by running each coroutine on a fresh event loop,
with a fake process launcher so no trame server starts.
"""
import asyncio
import itertools
import time

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from seisvtk.viz.sessions import SessionManager
from seisvtk.viz.tokens import sign_hs256
from seisvtk.viz.viewer_service import make_app, jwt_authorizer


class FakeProc:
    def __init__(self):
        self._rc = None

    def poll(self):
        return self._rc

    def terminate(self):
        self._rc = -15


def _manager(**kw):
    ports = itertools.count(8920)
    return SessionManager(launch=lambda scene, port: FakeProc(),
                          allocate_port=lambda: next(ports), **kw)


def _run(app, coro_factory):
    async def go():
        async with TestClient(TestServer(app)) as client:
            await coro_factory(client)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(go())
    finally:
        loop.close()


def test_create_get_touch_close_flow():
    mgr = _manager()
    app = make_app(mgr, public_base="https://viewer.example")

    async def steps(client):
        r = await client.post("/sessions", json={"scene": "s.json"})
        assert r.status == 201
        body = await r.json()
        assert body["scene"] == "s.json" and body["port"] == 8920
        assert body["url"] == f"https://viewer.example/view/{body['id']}"
        sid = body["id"]

        r = await client.get(f"/sessions/{sid}")
        assert r.status == 200

        r = await client.post(f"/sessions/{sid}/touch")
        assert r.status == 200 and (await r.json())["ok"] is True

        r = await client.delete(f"/sessions/{sid}")
        assert r.status == 200
        r = await client.get(f"/sessions/{sid}")
        assert r.status == 404

    _run(app, steps)


def test_create_requires_scene():
    app = make_app(_manager())

    async def steps(client):
        r = await client.post("/sessions", json={})
        assert r.status == 400

    _run(app, steps)


def test_capacity_returns_503():
    app = make_app(_manager(max_sessions=1))

    async def steps(client):
        assert (await client.post("/sessions", json={"scene": "a"})).status == 201
        assert (await client.post("/sessions", json={"scene": "b"})).status == 503

    _run(app, steps)


def test_authorize_hook_blocks():
    def deny(_request):
        raise web.HTTPUnauthorized(reason="nope")

    app = make_app(_manager(), authorize=deny)

    async def steps(client):
        r = await client.post("/sessions", json={"scene": "s"})
        assert r.status == 401
        assert (await client.get("/healthz")).status == 200   # healthz is open

    _run(app, steps)


def test_healthz_reports_session_count():
    app = make_app(_manager())

    async def steps(client):
        await client.post("/sessions", json={"scene": "a"})
        r = await client.get("/healthz")
        assert (await r.json()) == {"status": "ok", "sessions": 1}

    _run(app, steps)


# -- /view token entry + reverse proxy ----------------------------------------
# A real upstream aiohttp server stands in for a session's trame process, so the
# proxy path (HTTP + WebSocket) is exercised end to end. The SessionManager's
# allocate_port points every session at that upstream's port.
SECRET = "viewer-shared-secret"


def _upstream_app():
    async def index(_r):
        return web.Response(text="index")

    async def ping(_r):
        return web.Response(text="pong")

    async def echo(request):
        return web.Response(text=request.query.get("x", ""))

    async def ws(request):
        server = web.WebSocketResponse()
        await server.prepare(request)
        async for msg in server:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await server.send_str("echo:" + msg.data)
        return server

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ping", ping),
                    web.get("/echo", echo), web.get("/ws", ws)])
    return app


def _run_with_upstream(coro_factory, *, secret=SECRET):
    """Run *coro_factory(client)* with a viewer app whose single session port
    points at a live upstream server (the fake trame process)."""
    async def go():
        upstream = TestServer(_upstream_app())
        await upstream.start_server()
        mgr = SessionManager(launch=lambda scene, port: FakeProc(),
                             allocate_port=lambda: upstream.port)
        app = make_app(mgr, authorize=jwt_authorizer(secret))
        # unsafe=True so the session cookie is kept for the 127.0.0.1 test host.
        jar = aiohttp.CookieJar(unsafe=True)
        async with TestClient(TestServer(app), cookie_jar=jar) as client:
            await coro_factory(client)
        await upstream.close()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(go())
    finally:
        loop.close()


def _token(scene="s3://bucket/scene.json", ttl=60, secret=SECRET):
    return sign_hs256({"scene": scene, "exp": time.time() + ttl}, secret)


def test_view_creates_session_then_reverse_proxies_http():
    async def steps(client):
        # /view follows its 302 to "/", which the cookie now routes to the session.
        r = await client.get(f"/view?token={_token()}")
        assert r.status == 200
        assert await r.text() == "index"
        # subsequent root-relative requests proxy to the same session
        assert await (await client.get("/ping")).text() == "pong"
        assert await (await client.get("/echo?x=hi")).text() == "hi"

    _run_with_upstream(steps)


def test_websocket_is_proxied_after_view():
    async def steps(client):
        await client.get(f"/view?token={_token()}")
        ws = await client.ws_connect("/ws")
        await ws.send_str("frame")
        msg = await ws.receive()
        assert msg.data == "echo:frame"
        await ws.close()

    _run_with_upstream(steps)


def test_view_rejects_bad_token():
    app = make_app(_manager(), authorize=jwt_authorizer(SECRET))

    async def steps(client):
        bad = sign_hs256({"scene": "x", "exp": time.time() + 60}, "wrong-secret")
        assert (await client.get(f"/view?token={bad}")).status == 401
        assert (await client.get("/view")).status == 401          # no token

    _run(app, steps)


def test_proxy_without_cookie_returns_404():
    app = make_app(_manager(), authorize=jwt_authorizer(SECRET))

    async def steps(client):
        # No /view first -> no session cookie -> nothing to proxy to.
        assert (await client.get("/some/asset.js")).status == 404

    _run(app, steps)


def test_management_api_requires_admin_scope():
    app = make_app(_manager(), authorize=jwt_authorizer(SECRET))

    async def steps(client):
        view_tok = sign_hs256({"scene": "x", "exp": time.time() + 60}, SECRET)
        r = await client.post("/sessions", json={"scene": "x"},
                              headers={"Authorization": f"Bearer {view_tok}"})
        assert r.status == 403                             # a view token is not admin
        admin_tok = sign_hs256({"scope": "admin", "exp": time.time() + 60}, SECRET)
        r = await client.post("/sessions", json={"scene": "x"},
                              headers={"Authorization": f"Bearer {admin_tok}"})
        assert r.status == 201

    _run(app, steps)
