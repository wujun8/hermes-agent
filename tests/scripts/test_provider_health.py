"""Black-box regression coverage for the provider health probe's SSE handling."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-provider.sh"
_FAKE_API_KEY = "hermes-probe-test-fake-credential"


def _sse_event(event_type: str, response_status: str) -> bytes:
    data = {
        "type": event_type,
        "response": {
            "id": "resp-probe-test",
            "status": response_status,
            "model": "probe-test",
            "output": [],
        },
    }
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _error_event() -> bytes:
    data = {"type": "error", "error": {"message": "fixture error"}}
    return f"event: error\ndata: {json.dumps(data)}\n\n".encode()


def _done_event() -> bytes:
    return b"data: [DONE]\n\n"


_COMPLETED = _sse_event("response.completed", "completed")
_SSE_CASES = [
    pytest.param(_COMPLETED, True, id="completed-without-done"),
    pytest.param(_COMPLETED + _done_event(), True, id="completed-trailing-done"),
    pytest.param(_done_event(), False, id="done-only"),
    pytest.param(
        _done_event() + _COMPLETED,
        False,
        id="done-before-completed",
    ),
    pytest.param(
        _COMPLETED + _done_event() + _done_event(),
        False,
        id="repeated-trailing-done",
    ),
    pytest.param(
        _error_event() + _COMPLETED,
        False,
        id="error-before-completed",
    ),
    pytest.param(
        _sse_event("response.failed", "failed"),
        False,
        id="response-failed",
    ),
    pytest.param(
        _sse_event("response.incomplete", "incomplete"),
        False,
        id="response-incomplete",
    ),
    pytest.param(
        _sse_event("response.completed", "queued"),
        False,
        id="completed-with-wrong-status",
    ),
]


class _ProbeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.requests: list[tuple[str, dict | None]] = []


class _SSEHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        request_bytes = self.rfile.read(content_length)
        try:
            request_body = json.loads(request_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = None
        self.server.requests.append((self.path, request_body))

        if self.path != "/v1/responses":
            self.send_error(404)
            return

        payload = self.server.sse_payload
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.server.requests.append((self.path, None))
        self.send_error(404)

    def log_message(self, _format, *_args):
        pass


def _start_sse_server(payload: bytes):
    server = _ProbeServer(("127.0.0.1", 0), _SSEHandler)
    server.sse_payload = payload
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _minimal_path() -> str:
    curl_path = shutil.which("curl")
    jq_path = shutil.which("jq")
    assert curl_path, "curl is required by scripts/test-provider.sh"
    assert jq_path, "jq is required by scripts/test-provider.sh"

    entries = [
        Path(sys.executable).parent,
        Path(curl_path).parent,
        Path(jq_path).parent,
        *(Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry),
    ]
    return os.pathsep.join(dict.fromkeys(str(entry) for entry in entries))


def _write_test_config(hermes_home: Path, base_url: str) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  provider: custom",
                f'  base_url: "{base_url}"',
                "  default: probe-test",
                "  api_mode: codex_responses",
                f'  api_key: "{_FAKE_API_KEY}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(("sse_payload", "should_pass"), _SSE_CASES)
def test_provider_health_probe_handles_responses_sse_black_box(
    sse_payload, should_pass, tmp_path
):
    """Exercise the shell probe against a local Responses SSE endpoint."""
    assert _SCRIPT.is_file(), f"probe script is missing: {_SCRIPT}"
    case_dir = tmp_path
    hermes_home = case_dir / "hermes-home"
    fake_home = case_dir / "user-home"
    fake_home.mkdir()

    server, server_thread = _start_sse_server(sse_payload)
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        _write_test_config(hermes_home, base_url)
        child_env = {
            "HOME": str(fake_home),
            "PATH": _minimal_path(),
            "TMPDIR": str(case_dir),
            "HERMES_DIR": str(hermes_home),
            "HERMES_HOME": str(hermes_home),
            "TEST_MODEL": "probe-test",
            "LIST_MODELS": "0",
            "CURL_TIMEOUT": "3",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        result = subprocess.run(
            ["bash", str(_SCRIPT)],
            cwd=_SCRIPT.parents[1],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)

    assert not server_thread.is_alive(), "local SSE server thread did not stop"
    output = f"{result.stdout}\n{result.stderr}"
    if should_pass:
        assert result.returncode == 0, f"probe exit code was {result.returncode}\n{output}"
    else:
        assert result.returncode != 0, f"probe unexpectedly exited successfully\n{output}"
    marker = "[PASS]" if should_pass else "[FAIL]"
    assert marker in output.upper(), f"probe output did not contain {marker}\n{output}"

    assert server.requests, "probe did not make a request to the local server"
    assert all(path == "/v1/responses" for path, _body in server.requests), (
        f"probe made unexpected requests: {server.requests!r}"
    )
    assert all(
        isinstance(body, dict) and body.get("model") == "probe-test"
        for _path, body in server.requests
    ), f"probe sent an unexpected model request: {server.requests!r}"
