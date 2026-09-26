"""Regression coverage for the loop used by the Linux serve runner."""

import pytest
import uvicorn._compat as uvicorn_compat

from hermes_cli import web_server


@pytest.mark.linux_only
def test_linux_serve_runner_receives_uvicorn_loop_factory(monkeypatch):
    loop_factory = object()
    config_calls = []

    class Config:
        def get_loop_factory(self):
            config_calls.append(True)
            return loop_factory

    runner_calls = []

    def fake_uvicorn_runner(coro, *, loop_factory=None):
        runner_calls.append(loop_factory)
        coro.close()

    monkeypatch.setattr(uvicorn_compat, "asyncio_run", fake_uvicorn_runner, raising=False)

    bare_runner_calls = []

    def fake_asyncio_run(coro):
        bare_runner_calls.append(True)
        coro.close()

    monkeypatch.setattr(web_server.asyncio, "run", fake_asyncio_run)

    async def serve():
        pass

    web_server._run_serve(serve, Config(), "127.0.0.1", 0)

    assert config_calls == [True]
    assert runner_calls == [loop_factory]
    assert bare_runner_calls == []


@pytest.mark.linux_only
def test_linux_serve_treats_keyboardinterrupt_as_clean_shutdown(monkeypatch):
    loop_factory = object()
    seen = {}

    def fake_uvicorn_runner(coro, *, loop_factory=None):
        seen["loop_factory"] = loop_factory
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn_compat, "asyncio_run", fake_uvicorn_runner, raising=False)

    def fail_bare_runner(coro):
        coro.close()
        raise AssertionError("Linux serve must use uvicorn's loop-factory runner")

    monkeypatch.setattr(web_server.asyncio, "run", fail_bare_runner)

    class Config:
        def get_loop_factory(self):
            return loop_factory

    async def serve():
        pass

    # _run_serve catches the runner's KeyboardInterrupt after graceful shutdown.
    web_server._run_serve(serve, Config(), "127.0.0.1", 0)
    assert seen["loop_factory"] is loop_factory
