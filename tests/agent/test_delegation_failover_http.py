"""A real API failure on an optional delegated route falls back to configured routing."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import hermes_yaml as yaml


def test_optional_route_http_503_uses_configured_responses_fallback_and_cools_route(
    tmp_path, monkeypatch,
):
    from agent import delegation_route_recovery
    from agent.delegation_route_health import RouteHealthRegistry
    from hermes_cli import plugins
    from hermes_cli.config import load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB
    from tools import delegate_tool
    from tools.delegate_tool_routes import resolve_delegation_route

    requests = []
    metadata_probes = []

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in {
                "/deepseek/v1/chat/completions", "/configured/v1/responses",
            }:
                metadata_probes.append(self.path)
                self._json(404, {"error": {"message": "unexpected metadata probe"}})
                return
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            requests.append({
                "path": self.path,
                "model": payload.get("model"),
                "authorization": self.headers.get("Authorization"),
                "route": self.headers.get("X-Hermes-Route"),
            })
            if self.path == "/deepseek/v1/chat/completions":
                self._json(503, {"error": {"message": "route temporarily unavailable", "type": "server_error"}})
                return
            if self.path == "/configured/v1/responses":
                events = [
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": "msg-fallback",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "configured route completed"}],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {"id": "resp-fallback", "status": "completed", "output": []},
                    },
                ]
                body = "".join(
                    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._json(404, {"error": {"message": "unexpected route"}})

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    server_thread.start()

    home = tmp_path / ".hermes"
    home.mkdir()
    bundled_plugins = tmp_path / "bundled-plugins"
    bundled_plugins.mkdir()
    plugin_dir = home / "plugins" / "route-http-probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: route-http-probe\nversion: 1.0.0\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "from hermes_cli.delegation_routes import DelegationRouteSelection\n"
        "def select(_settings):\n"
        "    return DelegationRouteSelection(\n"
        "        'deepseek-flash', provider='custom:deepseek-route', api_mode='chat_completions',\n"
        "        fallback_to_configured=True, failure_cooldown_seconds=60,\n"
        "    )\n"
        "def register(ctx):\n"
        "    ctx.register_delegation_route_selector('http-probe', select)\n",
        encoding="utf-8",
    )

    base_url = f"http://127.0.0.1:{server.server_port}"
    config = {
        "plugins": {"enabled": ["route-http-probe"], "isolation": "in_process"},
        "delegation": {
            "provider": "custom:configured-route",
            "model": "gpt-6-luna",
            "api_mode": "codex_responses",
            "max_iterations": 3,
            "fallback_providers": [],
            "request_overrides": {"extra_headers": {"X-Hermes-Route": "configured"}},
        },
        "agent": {
            "api_max_retries": 1,
            "api_outage_recovery": {"enabled": True, "probe_command": "unused-probe"},
        },
        "custom_providers": [
            {
                "name": "deepseek-route",
                "base_url": f"{base_url}/deepseek/v1",
                "api_key": "deepseek-route-key",
                "api_mode": "chat_completions",
                "extra_headers": {"X-Hermes-Route": "optional"},
            },
            {
                "name": "configured-route",
                "base_url": f"{base_url}/configured/v1",
                "api_key": "configured-route-key",
                "api_mode": "codex_responses",
                "extra_headers": {"X-Hermes-Route": "configured"},
            },
        ],
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled_plugins))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")

    clock = [0.0]
    route_health = RouteHealthRegistry(clock=lambda: clock[0])
    monkeypatch.setattr(delegation_route_recovery, "ROUTE_HEALTH", route_health)
    plugins._reset_plugin_managers_for_tests()
    home_token = set_hermes_home_override(str(home))
    parent_db = SessionDB(db_path=home / "state.db")
    parent_db.create_session("parent-http", source="cli", profile_name="http", model_config={})
    parent = SimpleNamespace(
        _session_db=parent_db,
        session_id="parent-http",
        _delegate_depth=0,
        model="parent-model",
        provider="custom",
        requested_provider="custom:configured-route",
        base_url=f"{base_url}/configured/v1",
        api_key="configured-route-key",
        api_mode="codex_responses",
        request_overrides={},
        reasoning_config=None,
        _fallback_chain=None,
        enabled_toolsets=[],
        disabled_toolsets=[],
        capabilities=None,
        prefill_messages=None,
        tool_progress_callback=None,
    )
    child = None
    outage_waits = []

    class NeverWait:
        def wait(self, *_args, **_kwargs):
            outage_waits.append(True)
            raise AssertionError("configured-route fallback should avoid API outage parking")

    try:
        monkeypatch.setattr("model_tools.get_tool_definitions", lambda *_a, **_kw: [])
        monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
        cfg = load_config_readonly()["delegation"]
        route, creds = resolve_delegation_route(cfg, parent)
        assert creds["model"] == "deepseek-flash"
        assert creds["provider"] == "custom:deepseek-route"
        assert creds["api_mode"] == "chat_completions"
        assert route["fallback_providers"][0]["model"] == "gpt-6-luna"
        assert route["fallback_providers"][0]["api_mode"] == "codex_responses"

        child = delegate_tool._build_child_agent(
            task_index=0,
            goal="Complete this using the available delegated API route.",
            context=None,
            toolsets=[],
            model=creds["model"],
            max_iterations=3,
            task_count=1,
            parent_agent=parent,
            **delegate_tool._credential_overrides(creds, route),
        )
        assert child._selected_route_recovery is not None
        assert child._selected_route_recovery["identity"] == (
            child.requested_provider.lower(), child.base_url.rstrip("/"), child.model, child.api_mode,
        )
        assert child._api_outage_recovery_config.enabled is True
        child._api_outage_recovery_waiter = NeverWait()

        result = child.run_conversation("Return the configured route's completion.")
        assert result["completed"] is True
        assert result["final_response"] == "configured route completed"
        assert outage_waits == []
        assert child._selected_route_recovery is None
        assert set(metadata_probes) <= {"/deepseek/api/show"}
        optional_requests = [request for request in requests if request["path"] == "/deepseek/v1/chat/completions"]
        configured_requests = [request for request in requests if request["path"] == "/configured/v1/responses"]
        assert optional_requests
        assert all(
            request["model"] == "deepseek-flash"
            and request["authorization"] == "Bearer deepseek-route-key"
            and request["route"] == "optional"
            for request in optional_requests
        )
        assert configured_requests
        assert all(
            request["model"] == "gpt-6-luna"
            and request["authorization"] == "Bearer configured-route-key"
            and request["route"] == "configured"
            for request in configured_requests
        )
        assert requests[-1] == configured_requests[-1]

        cooled_route, cooled_creds = resolve_delegation_route(cfg, parent)
        assert cooled_route is cfg
        assert cooled_creds["model"] == "gpt-6-luna"
        assert cooled_creds["provider"] == "custom:configured-route"
    finally:
        if child is not None:
            child.close()
        parent_db.close()
        plugins._reset_plugin_managers_for_tests()
        reset_hermes_home_override(home_token)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
