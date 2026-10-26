"""The viewer-service HTTP app: session create/get/touch/close + auth hook.

Driven without pytest-asyncio by running each coroutine on a fresh event loop,
with a fake process launcher so no trame server starts.
"""
import asyncio
import itertools

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from seisvtk.viz.sessions import SessionManager
from seisvtk.viz.viewer_service import make_app


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
