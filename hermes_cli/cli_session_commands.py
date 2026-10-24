"""Classic CLI adapter for native plugin session commands."""

from __future__ import annotations

from contextlib import suppress


def run_cli_session_command(cli, handler, raw_args: str):
    """Parse before touching the foreground session and use its canonical writer."""
    from hermes_cli.session_commands import (
        make_session_context, prepare_session_command, run_session_command,
    )
    from hermes_constants import get_hermes_home

    try:
        request = prepare_session_command(handler, raw_args)
    except ValueError as exc:
        return str(exc)

    agent = getattr(cli, "agent", None)
    if agent is None:
        raise ValueError("the foreground agent is not initialized")
    session_id = str(getattr(agent, "session_id", "") or "")
    if not session_id.strip():
        raise ValueError("a non-empty exact session ID is required")
    if str(getattr(cli, "session_id", session_id) or "") != session_id:
        raise ValueError("foreground session identity mismatch; refusing to mutate")
    session_db = getattr(agent, "_session_db", None)
    context = make_session_context(
        agent=agent,
        session_db=session_db,
        session_id=session_id,
        ensure_session=getattr(agent, "_ensure_db_session", None),
        profile_home=get_hermes_home(),
    )
    return run_session_command(handler, request, context)
