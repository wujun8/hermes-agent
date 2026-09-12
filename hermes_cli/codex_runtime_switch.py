"""Shared logic for the /codex-runtime slash command (CLI and gateway call into this module)."""

from __future__ import annotations

import copy
import functools
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


VALID_RUNTIMES = ("auto", "codex_app_server")

# Human-friendly synonyms accepted by parse_args.
_ARG_SYNONYMS = {
    "on": "codex_app_server", "codex": "codex_app_server", "enable": "codex_app_server",
    "off": "auto", "default": "auto", "disable": "auto", "hermes": "auto"}

_HERMES_TOOLS_CALLBACK_NOTE = (
    "Hermes tool callback registered: codex can now use "
    "web_search, web_extract, browser_*, vision_analyze, "
    "image_generate, skill_view, skills_list, text_to_speech, "
    "kanban_* (worker + orchestrator) via MCP.",
    "  (delegate_task, memory, session_search, todo run "
    "only on the default Hermes runtime — they need the "
    "agent loop context.)")


@dataclass
class CodexRuntimeStatus:
    """Result of a /codex-runtime invocation; callers render it per surface (Rich panel / text)."""

    success: bool
    new_value: Optional[str] = None
    old_value: Optional[str] = None
    message: str = ""
    requires_new_session: bool = False


def parse_args(arg_string: str) -> tuple[Optional[str], list[str]]:
    """Parse the slash-command argument string into ``(value, errors)``.

    No args → ``(None, [])`` (show current state); a runtime name or synonym → that runtime.
    """
    raw = (arg_string or "").strip().lower()
    if not raw:
        return None, []
    value = _ARG_SYNONYMS.get(raw, raw)
    if value in VALID_RUNTIMES:
        return value, []
    return None, [f"Unknown runtime {raw!r}. Use one of: auto, codex_app_server, on, off"]


def get_current_runtime(config: dict) -> str:
    """Current ``model.openai_runtime``; 'auto' for unset / empty / unrecognized values."""
    if not isinstance(config, dict):
        return "auto"
    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        return "auto"
    value = str(model_cfg.get("openai_runtime") or "").strip().lower()
    return value if value in VALID_RUNTIMES else "auto"


def set_runtime(config: dict, new_value: str) -> str:
    """Persist *new_value* into the config dict in place; returns the previous value."""
    if new_value not in VALID_RUNTIMES:
        raise ValueError(f"invalid runtime {new_value!r}; must be one of {VALID_RUNTIMES}")
    old = get_current_runtime(config)
    if not isinstance(config.get("model"), dict):
        config["model"] = {}
    config["model"]["openai_runtime"] = new_value
    return old


def check_codex_binary_ok() -> tuple[bool, Optional[str]]:
    """Best-effort codex CLI install/version check → ``(ok, version_or_message)``."""
    try:
        from agent.transports.codex_app_server import check_codex_binary

        return check_codex_binary()
    except Exception as exc:  # pragma: no cover
        return False, f"codex check failed: {exc}"


def resolve_effective_runtime(config: dict) -> dict:
    """Resolve the runtime represented by a config snapshot without requiring it to be saved first."""
    model_cfg = config.get("model") if isinstance(config, dict) else None
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    requested = model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else None
    target_model = model_cfg.get("default") if isinstance(model_cfg.get("default"), str) else None

    from hermes_cli.runtime_provider import (
        _maybe_apply_codex_app_server_runtime,
        _resolve_named_custom_runtime,
        resolve_runtime_provider,
    )

    # The normal resolver reads provider definitions from disk. Resolve a named custom entry from
    # the pending snapshot first so validation works before save_config() commits the toggle.
    custom_runtime = _resolve_named_custom_runtime(
        requested_provider=requested or "", target_model=target_model,
        config=config, model_cfg=model_cfg,
    )
    if custom_runtime is not None:
        return custom_runtime

    runtime = dict(resolve_runtime_provider(requested=requested, target_model=target_model))
    runtime["api_mode"] = _maybe_apply_codex_app_server_runtime(
        provider=str(runtime.get("provider") or ""), api_mode=str(runtime.get("api_mode") or ""),
        model_cfg=model_cfg,
    )
    return runtime


def _effective_runtime_details(config: dict, resolver) -> tuple[str, str]:
    runtime = resolver(config)
    if not isinstance(runtime, dict):
        raise TypeError("runtime resolver returned a non-dict result")
    api_mode = str(runtime.get("api_mode") or "").strip()
    if not api_mode:
        raise ValueError("runtime resolver returned no api_mode")
    return api_mode, str(runtime.get("provider") or "auto").strip() or "auto"


def _format_runtime_resolution_error(exc: Exception) -> str:
    try:
        from hermes_cli.runtime_provider import format_runtime_provider_error

        return format_runtime_provider_error(exc)
    except Exception:
        return str(exc) or exc.__class__.__name__


def _restore_model_snapshot(config: dict, had_model: bool, original_model) -> None:
    if had_model:
        config["model"] = original_model
    else:
        config.pop("model", None)


def _migration_lines(config: dict) -> list[str]:
    """Run the ~/.codex/config.toml migration and describe it; failures are non-fatal."""
    lines: list[str] = []
    try:
        from hermes_cli.codex_runtime_plugin_migration import migrate
        mig_report = migrate(config)
        # The hermes-tools callback is internal plumbing — surfaced separately below.
        user_servers = [s for s in mig_report.migrated if s != "hermes-tools"]
        if user_servers:
            lines.append(f"Migrated {len(user_servers)} MCP server(s): {', '.join(user_servers)}")
        if mig_report.migrated_plugins:
            lines.append(
                f"Migrated {len(mig_report.migrated_plugins)} native "
                f"Codex plugin(s): {', '.join(mig_report.migrated_plugins)}")
        elif mig_report.plugin_query_error:
            lines.append(f"Codex plugin discovery skipped: {mig_report.plugin_query_error}")
        if mig_report.wrote_permissions_default:
            lines.append(
                f"Default sandbox: {mig_report.wrote_permissions_default} "
                f"(no approval prompt on every write)")
        if "hermes-tools" in mig_report.migrated:
            lines.extend(_HERMES_TOOLS_CALLBACK_NOTE)
        lines.append(f"  (config: {mig_report.target_path})")
        for err in mig_report.errors:
            lines.append(f"⚠ MCP migration: {err}")
    except Exception as exc:
        lines.append(f"⚠ MCP migration skipped: {exc}")
    return lines


def apply(
    config: dict, new_value: Optional[str], *, persist_callback=None, effective_runtime_resolver=None,
) -> CodexRuntimeStatus:
    """Entry point for CLI and gateway. ``config`` is mutated in place when ``new_value`` is set
    (None = show current state); ``persist_callback(config)`` writes it, skipped when None. Enabling
    is accepted only when the effective resolver confirms the app-server route."""
    current = get_current_runtime(config)
    effective_runtime_resolver = effective_runtime_resolver or resolve_effective_runtime

    # Cached per apply() call: the enable path would otherwise spawn `codex --version` up to 3x.
    _check_binary_cached = functools.cache(check_codex_binary_ok)

    if new_value is None:
        ok, ver = _check_binary_cached()
        msg_lines = [
            f"openai_runtime: {current}",
            f"codex CLI: {'OK ' + ver if ok else 'not available — ' + (ver or 'install with `npm i -g @openai/codex`')}",
        ]
        if current == "codex_app_server":
            try:
                api_mode, provider = _effective_runtime_details(config, effective_runtime_resolver)
                suffix = "" if api_mode == "codex_app_server" else " (configured app-server is inactive)"
                msg_lines.append(f"effective runtime: {api_mode} (provider: {provider}){suffix}")
            except Exception as exc:
                msg_lines.append(f"effective runtime: unresolved ({_format_runtime_resolution_error(exc)})")
        return CodexRuntimeStatus(
            success=True, new_value=current, old_value=current, message="\n".join(msg_lines),
        )

    # Re-enabling codex_app_server falls through to the migration: the config value is already
    # correct but the world state (managed block in ~/.codex/config.toml, hermes-tools MCP
    # callback, plugin discovery) may be stale — a common footgun when users pre-set
    # `openai_runtime: codex_app_server` by hand. The migration is idempotent so re-running is
    # cheap and safe. Re-setting `auto` returns immediately (disabling never touches ~/.codex/).
    reapplying_enable = new_value == current == "codex_app_server"
    if new_value == current and not reapplying_enable:
        return CodexRuntimeStatus(
            success=True, new_value=current, old_value=current,
            message=f"openai_runtime already set to {current}")

    # Switching ON: verify codex CLI before persisting — an opt-in toggle that silently fails on
    # the first turn is the worst possible UX.
    if new_value == "codex_app_server":
        ok, ver_or_msg = _check_binary_cached()
        if not ok:
            return CodexRuntimeStatus(
                success=False, new_value=None, old_value=current,
                message=(
                    "Cannot enable codex_app_server runtime: "
                    f"{ver_or_msg or 'codex CLI not available'}\n"
                    "Install with: npm i -g @openai/codex"))

    had_model = "model" in config
    original_model = copy.deepcopy(config.get("model"))
    if not reapplying_enable:
        set_runtime(config, new_value)

    effective_api_mode = None
    effective_provider = None
    if new_value == "codex_app_server":
        try:
            effective_api_mode, effective_provider = _effective_runtime_details(config, effective_runtime_resolver)
        except Exception as exc:
            if not reapplying_enable:
                _restore_model_snapshot(config, had_model, original_model)
            return CodexRuntimeStatus(
                success=False, new_value=None, old_value=current,
                message=("Cannot enable codex_app_server runtime: effective runtime resolution failed: "
                         f"{_format_runtime_resolution_error(exc)}\nopenai_runtime was not changed."),
            )
        if effective_api_mode != "codex_app_server":
            if not reapplying_enable:
                _restore_model_snapshot(config, had_model, original_model)
            return CodexRuntimeStatus(
                success=False, new_value=None, old_value=current,
                message=("Cannot enable codex_app_server runtime: effective runtime resolved to "
                         f"{effective_api_mode} (provider: {effective_provider}).\n"
                         "openai_runtime was not changed."),
            )

    if not reapplying_enable:
        if persist_callback is not None:
            try:
                persist_callback(config)
            except Exception as exc:
                _restore_model_snapshot(config, had_model, original_model)
                logger.exception("failed to persist openai_runtime change")
                return CodexRuntimeStatus(
                    success=False, new_value=new_value, old_value=current,
                    message=f"could not persist openai_runtime change: {exc}")

    msg_lines = [
        f"openai_runtime already set to {current} — re-applying migration"
        if reapplying_enable
        else f"openai_runtime: {current} → {new_value}"]
    if new_value == "codex_app_server":
        ok, ver = _check_binary_cached()
        if ok:
            msg_lines.append(f"codex CLI: {ver}")
        msg_lines.append(
            f"effective runtime: {effective_api_mode} (provider: {effective_provider})"
        )
        # Migrate Hermes' MCP servers + Codex's curated plugins into ~/.codex/config.toml so the
        # spawned codex subprocess sees the same tool surface AND can call back into Hermes.
        msg_lines.extend(_migration_lines(config))
        msg_lines.append(
            "OpenAI/Codex turns now run through `codex app-server` "
            "(terminal/file ops/patching inside Codex; "
            "Hermes tools available via MCP callback).")
        msg_lines.append(
            "Effective on next session — current cached agent keeps "
            "the prior runtime to preserve prompt cache.")
    else:
        msg_lines.append("OpenAI/Codex turns will use the default Hermes runtime.")
        msg_lines.append("Effective on next session.")
    return CodexRuntimeStatus(
        success=True, new_value=new_value, old_value=current,
        message="\n".join(msg_lines), requires_new_session=True)
