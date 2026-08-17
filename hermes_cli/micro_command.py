"""DB-first execution for the session-scoped ``/micro`` command.

The parser and status formatter live in :mod:`hermes_cli.micro_compaction`.
This module owns the side-effect ordering shared by the classic CLI and the
later command surfaces: read global config, ensure the exact session row for a
mutation, write the durable override, and only then apply the live policy.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from hermes_cli.config import load_config_readonly
from hermes_cli.micro_compaction import (
    MICRO_COMPACT_OVERRIDE_KEY,
    MICRO_COMPACT_USAGE,
    effective_micro_compact,
    format_micro_compact_status,
    parse_global_micro_compact,
    parse_micro_compact_command,
)


_CONFIG_ERROR = "Micro-compaction error: could not read global configuration"
_DB_ERROR = "Micro-compaction error: session database operation failed"
logger = logging.getLogger(__name__)


def _command_args(raw_args: Any) -> Any:
    """Accept either command arguments or a full ``/micro ...`` command."""
    if not isinstance(raw_args, str):
        return raw_args
    text = raw_args.strip()
    parts = text.split(None, 1)
    if parts and parts[0].lstrip("/").lower() == "micro":
        return parts[1] if len(parts) == 2 else ""
    return text


def _global_config_value(config_loader: Callable[[], Any]) -> Any:
    """Read only ``compression.micro_compact`` from the supplied loader."""
    config = config_loader()
    if not isinstance(config, dict):
        return False
    compression = config.get("compression", {})
    if not isinstance(compression, dict):
        return False
    return compression.get("micro_compact", False)


def load_global_micro_compact_value(
    config_loader: Callable[[], Any] | None = None,
) -> Any:
    """Load the raw global value using the same readonly path as the service."""
    return _global_config_value(config_loader or load_config_readonly)


def hydrate_micro_compact_policy_for_session(
    *,
    agent: Any,
    session_db: Any,
    session_id: str,
    config_loader: Callable[[], Any] | None = None,
) -> bool:
    """Hydrate one exact resumed row through the committed core bridge."""
    try:
        global_value = _global_config_value(config_loader or load_config_readonly)
    except Exception as exc:
        # A conversation boundary must still hydrate the exact target row.
        # Retaining the last known global value avoids making an old session's
        # explicit override look like the new session's policy.
        global_value = getattr(agent, "_micro_compact_global_value", False)
        logger.warning(
            "Micro-compaction policy hydration could not load global config; "
            "using last known value for session %s: %s",
            session_id,
            exc,
        )
    from agent.agent_init import hydrate_micro_compact_policy

    return bool(
        hydrate_micro_compact_policy(
            agent,
            session_db=session_db,
            session_id=session_id,
            global_value=global_value,
        )
    )


def _strict_session_override(session_db: Any, session_meta: Any) -> bool | None:
    """Read the canonical strict tri-state value from a session row."""
    getter = getattr(session_db, "session_micro_compact_override", None)
    if callable(getter):
        value = getter(session_meta)
    else:
        # Keep lightweight test/session-store adapters usable while retaining
        # SessionDB's strict parser as the single fallback interpretation.
        from hermes_state import SessionDB

        value = SessionDB.session_micro_compact_override(session_meta)
    return value if value is True or value is False else None


def _read_session(session_db: Any, session_id: str) -> Any:
    getter = getattr(session_db, "get_session", None)
    if not callable(getter):
        raise AttributeError("session database has no get_session()")
    return getter(session_id)


def _error(prefix: str, exc: BaseException) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"{prefix}: {detail}"


def _status(global_value: Any, session_override: bool | None) -> str:
    return format_micro_compact_status(global_value, session_override)


def _unsupported_warning() -> str:
    return (
        "Warning: the policy was saved, but the current engine does not "
        "support switching micro-compaction at runtime."
    )


def _apply_live_policy(
    agent: Any,
    *,
    session_override: bool | None,
    global_value: Any,
) -> bool:
    """Call the committed core runtime bridge lazily to avoid import cycles."""
    from agent.agent_init import apply_micro_compact_policy

    return bool(
        apply_micro_compact_policy(
            agent,
            session_override=session_override,
            global_value=global_value,
        )
    )


def reset_micro_compact_policy_for_new_session(
    *,
    agent: Any,
    config_loader: Callable[[], Any] | None = None,
) -> bool:
    """Clear the live session override at the classic ``/new`` boundary.

    This intentionally changes only the in-memory policy.  The old session row
    remains untouched and the caller creates the new row afterward without an
    override key.  If readonly global config is temporarily unavailable, the
    agent's cached global value is the safest boundary fallback.
    """
    try:
        global_value = _global_config_value(config_loader or load_config_readonly)
    except Exception as exc:
        global_value = getattr(agent, "_micro_compact_global_value", False)
        logger.warning(
            "Micro-compaction policy reset at /new could not load global config; "
            "using last known value: %s",
            exc,
        )
    return _apply_live_policy(
        agent,
        session_override=None,
        global_value=global_value,
    )


def execute_micro_command(
    *,
    agent: Any | None,
    session_db: Any,
    session_id: str,
    raw_args: str,
    ensure_session: Callable[[], Any] | None = None,
    config_loader: Callable[[], Any] | None = None,
) -> str:
    """Execute one session-scoped micro policy command without printing.

    ``status`` is strictly read-only.  Mutating commands perform all validation
    and config loading first, call the canonical lazy-row callback, verify the
    exact row, persist via ``SessionDB.set_session_micro_compact_override``,
    and only then apply the live policy to ``agent``.  When ``agent`` is
    ``None`` (a cold, agent-less session), the durable write is the complete
    mutation: the policy is hydrated when the session's agent starts or
    resumes.
    """
    try:
        command = parse_micro_compact_command(_command_args(raw_args))
    except (TypeError, ValueError):
        return MICRO_COMPACT_USAGE

    try:
        global_value = _global_config_value(config_loader or load_config_readonly)
    except Exception as exc:
        return _error(_CONFIG_ERROR, exc)

    if command == "status":
        # A missing route/session or unavailable DB has no possible override.
        # Keep status read-only and report the source profile's global value
        # without opening, creating, or querying a database.
        if (
            not isinstance(session_id, str)
            or not session_id.strip()
            or session_db is None
        ):
            return _status(global_value, None)
        try:
            session_meta = _read_session(session_db, session_id)
            session_override = (
                None
                if session_meta is None
                else _strict_session_override(session_db, session_meta)
            )
        except Exception as exc:
            return _error(_DB_ERROR, exc)
        return _status(global_value, session_override)

    if not isinstance(session_id, str) or not session_id.strip():
        return "Micro-compaction error: a non-empty exact session ID is required"
    if session_db is None:
        return "Micro-compaction error: session database is not available"

    if ensure_session is None and agent is not None:
        ensure_session = getattr(agent, "_ensure_db_session", None)
    if not callable(ensure_session):
        return "Micro-compaction error: canonical session ensure callback is not available"

    # The callback is intentionally the first mutation step.  It is the
    # foreground agent's canonical lazy-session path, not a second row creator.
    try:
        ensure_session()
        session_meta = _read_session(session_db, session_id)
    except Exception as exc:
        return _error(_DB_ERROR, exc)
    if session_meta is None:
        return f"Micro-compaction error: session row not found: {session_id}"

    session_override: bool | None
    if command == "on":
        session_override = True
    elif command == "off":
        session_override = False
    else:
        session_override = None

    setter = getattr(session_db, "set_session_micro_compact_override", None)
    if not callable(setter):
        return "Micro-compaction error: session database has no override setter"
    try:
        # Durable state is committed before the engine or agent is touched.
        setter(session_id, session_override)
    except Exception as exc:
        return _error(_DB_ERROR, exc)

    if agent is None:
        state = "ON" if effective_micro_compact(global_value, session_override)[0] else "OFF"
        result = f"Micro-compaction override saved: {state}\n{_status(global_value, session_override)}"
        return (
            f"{result}\n"
            "Policy saved; it will apply when this session's agent starts or resumes."
        )

    try:
        supported = _apply_live_policy(
            agent,
            session_override=session_override,
            global_value=global_value,
        )
    except Exception as exc:
        return _error(
            "Micro-compaction saved but live policy application failed",
            exc,
        )

    state = "ON" if effective_micro_compact(global_value, session_override)[0] else "OFF"
    result = f"Micro-compaction override saved: {state}\n{_status(global_value, session_override)}"
    if not supported:
        result = f"{result}\n{_unsupported_warning()}"
    return result


# A descriptive alias for callers that prefer a verb matching other command
# modules.  Both names resolve to the one implementation above.
run_micro_command = execute_micro_command


__all__ = [
    "execute_micro_command",
    "hydrate_micro_compact_policy_for_session",
    "load_global_micro_compact_value",
    "reset_micro_compact_policy_for_new_session",
    "run_micro_command",
    "MICRO_COMPACT_OVERRIDE_KEY",
]
