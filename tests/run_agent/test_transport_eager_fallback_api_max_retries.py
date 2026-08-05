"""Regression: transport eager-fallback must honor agent.api_max_retries.

Before the fix, conversation_loop.py used a hardcoded ``retry_count >= 2``
gate for timeout/overloaded errors, so fallback engaged after only two
primary attempts even when ``api_max_retries`` was 3 (or higher).

Rate-limit / billing / upstream_rate_limit must still fall back immediately.
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


def _make_agent_with_fallback(fb_chain, *, api_max_retries=3):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            model="primary-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fb_chain,
        )
        agent.client = MagicMock()
        agent._api_max_retries = api_max_retries
        return agent


def _mock_response(content: str, *, model: str = "primary-model"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model=model, usage=None)


class ReadTimeout(Exception):
    pass


class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


class TestTransportEagerFallbackRespectsApiMaxRetries:
    def test_transport_timeout_uses_all_api_max_retries_before_fallback(self):
        """With api_max_retries=3, three primary timeouts must run before fallback."""
        fb_chain = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example.com/v1",
            }
        ]
        agent = _make_agent_with_fallback(fb_chain, api_max_retries=3)

        calls: list[tuple[str, str]] = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if agent._fallback_activated:
                return _mock_response("ok via fallback", model="fallback-model")
            raise ReadTimeout("read timed out")

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
            patch.object(agent, "_try_recover_primary_transport", return_value=False),
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
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "ok via fallback"
        assert calls == [
            ("custom", "primary-model"),
            ("custom", "primary-model"),
            ("custom", "primary-model"),
            ("custom", "fallback-model"),
        ]
        assert agent._fallback_activated is True

    def test_rate_limit_still_falls_back_immediately(self):
        """429 on first attempt must not wait for api_max_retries to exhaust."""
        fb_chain = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example.com/v1",
            }
        ]
        agent = _make_agent_with_fallback(fb_chain, api_max_retries=3)

        calls: list[tuple[str, str]] = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if agent._fallback_activated:
                return _mock_response("ok via fallback", model="fallback-model")
            raise RateLimitError()

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
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert calls == [
            ("custom", "primary-model"),
            ("custom", "fallback-model"),
        ]
