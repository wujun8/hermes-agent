"""Regression: Codex Responses terminal ``status=failed`` + transient error
codes must honor ``agent.api_max_retries`` before fallback.

Before the fix, conversation_loop treated every ``failed``/``cancelled``
Responses terminal as an immediate eager-fallback trigger, so a single
``upstream_error`` (or ``overloaded`` / ``server_error``) bypassed
``api_max_retries`` even when the primary might recover.

Rate-limit / quota / billing codes must still fall back immediately.
Unknown codes and ``cancelled`` keep the prior eager-fallback behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_codex_agent_with_fallback(fb_chain, *, api_max_retries=3):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url="https://codex.example.com/v1",
            provider="custom",
            model="primary-model",
            api_mode="codex_responses",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fb_chain,
        )
        agent.client = MagicMock()
        agent._api_max_retries = api_max_retries
        return agent


def _mock_chat_response(content: str, *, model: str = "fallback-model"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model=model, usage=None)


def _mock_codex_failed(*, code: str, message: str = "Upstream request failed"):
    """Responses API terminal failure as returned by the stream consumer."""
    return SimpleNamespace(
        status="failed",
        error=SimpleNamespace(code=code, message=message),
        output=None,
        output_text=None,
        model="primary-model",
        usage=None,
        id="resp_failed_1",
    )


def _mock_codex_ok(content: str = "ok via primary"):
    return SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=content)],
            )
        ],
        output_text=content,
        model="primary-model",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        id="resp_ok_1",
        error=None,
    )


def _fallback_chain():
    return [
        {
            "provider": "custom",
            "model": "fallback-model",
            "base_url": "https://fallback.example.com/v1",
        }
    ]


def _run_with_mocked_transport(agent, fake_api_call):
    mock_fb_client = MagicMock()
    mock_fb_client.api_key = "primary-key-abcdef12"
    mock_fb_client.base_url = "https://fallback.example.com/v1"
    mock_fb_client._custom_headers = None
    mock_fb_client.default_headers = None

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop.time.sleep"),
        patch("agent.conversation_loop.jittered_backoff", return_value=0.0),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_fb_client, "fallback-model"),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda m, p: m,
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        return agent.run_conversation("hello")


class TestCodexTerminalTransientFallbackRespectsApiMaxRetries:
    def test_upstream_error_uses_all_api_max_retries_before_fallback(self):
        """With api_max_retries=3, three primary upstream_error terminals precede fallback."""
        agent = _make_codex_agent_with_fallback(_fallback_chain(), api_max_retries=3)
        assert agent.api_mode == "codex_responses"

        calls: list[tuple[str, str]] = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if agent._fallback_activated:
                return _mock_chat_response("ok via fallback", model="fallback-model")
            return _mock_codex_failed(code="upstream_error")

        result = _run_with_mocked_transport(agent, fake_api_call)

        assert result["completed"] is True
        assert result["final_response"] == "ok via fallback"
        assert calls == [
            ("custom", "primary-model"),
            ("custom", "primary-model"),
            ("custom", "primary-model"),
            ("custom", "fallback-model"),
        ]
        assert agent._fallback_activated is True

    def test_overloaded_and_server_error_also_defer_eager_fallback(self):
        """``overloaded`` / ``server_error`` share the transient retry path."""
        for code in ("overloaded", "server_error"):
            agent = _make_codex_agent_with_fallback(_fallback_chain(), api_max_retries=3)
            calls: list[tuple[str, str]] = []

            def fake_api_call(api_kwargs, _code=code):
                calls.append((agent.provider, agent.model))
                if agent._fallback_activated:
                    return _mock_chat_response("ok via fallback", model="fallback-model")
                return _mock_codex_failed(code=_code, message=f"{_code} boom")

            result = _run_with_mocked_transport(agent, fake_api_call)

            assert result["completed"] is True, code
            assert calls == [
                ("custom", "primary-model"),
                ("custom", "primary-model"),
                ("custom", "primary-model"),
                ("custom", "fallback-model"),
            ], code

    def test_rate_limit_exceeded_still_falls_back_immediately(self):
        """Quota/rate-limit terminal codes must not wait for api_max_retries."""
        agent = _make_codex_agent_with_fallback(_fallback_chain(), api_max_retries=3)
        calls: list[tuple[str, str]] = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if agent._fallback_activated:
                return _mock_chat_response("ok via fallback", model="fallback-model")
            return _mock_codex_failed(
                code="rate_limit_exceeded",
                message="Slow down",
            )

        result = _run_with_mocked_transport(agent, fake_api_call)

        assert result["completed"] is True
        assert calls == [
            ("custom", "primary-model"),
            ("custom", "fallback-model"),
        ]

    def test_insufficient_quota_still_falls_back_immediately(self):
        agent = _make_codex_agent_with_fallback(_fallback_chain(), api_max_retries=3)
        calls: list[tuple[str, str]] = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if agent._fallback_activated:
                return _mock_chat_response("ok via fallback", model="fallback-model")
            return _mock_codex_failed(
                code="insufficient_quota",
                message="You exceeded your current quota",
            )

        result = _run_with_mocked_transport(agent, fake_api_call)

        assert result["completed"] is True
        assert calls == [
            ("custom", "primary-model"),
            ("custom", "fallback-model"),
        ]
