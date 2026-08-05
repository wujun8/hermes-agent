"""Pure semantic helpers for the session micro-compaction policy.

This module deliberately knows nothing about agents, gateways, or persistence.
It is the shared parser/evaluator/formatter used by the later command surfaces
and by the runtime policy bridge.
"""

from __future__ import annotations

from typing import Any, Literal

from utils import is_truthy_value


MICRO_COMPACT_OVERRIDE_KEY = "micro_compact_override"
MICRO_COMPACT_COMMANDS = frozenset({"on", "off", "inherit", "status"})
MICRO_COMPACT_USAGE = "Usage: /micro on|off|inherit|status"
MicroCompactionCommand = Literal["on", "off", "inherit", "status"]


class MicroCompactionUsageError(ValueError):
    """Safe, non-mutating error raised for invalid micro policy input."""

    usage = MICRO_COMPACT_USAGE

    def __init__(self) -> None:
        super().__init__(f"invalid micro-compaction command; {self.usage}")


def parse_micro_compact_command(raw: str) -> MicroCompactionCommand:
    """Parse exactly one normalized ``on|off|inherit|status`` token.

    The parser intentionally accepts only a string containing one token after
    outer whitespace is removed.  It never performs persistence or runtime
    mutation, so callers can validate before changing any state.
    """
    if not isinstance(raw, str):
        raise MicroCompactionUsageError()
    tokens = raw.strip().lower().split()
    if len(tokens) != 1 or tokens[0] not in MICRO_COMPACT_COMMANDS:
        raise MicroCompactionUsageError()
    return tokens[0]  # type: ignore[return-value]


def parse_global_micro_compact(value: Any) -> bool:
    """Parse the global value using the project's shared truthy semantics."""
    return is_truthy_value(value, default=False)


def effective_micro_compact(
    global_value: Any,
    session_override: bool | None,
) -> tuple[bool, Literal["session", "global"]]:
    """Return the effective flag and whether it came from the session.

    Only literal booleans are explicit overrides.  Treating any other value as
    inheritance is fail-closed and prevents arbitrary strings from becoming a
    durable session decision.
    """
    global_enabled = parse_global_micro_compact(global_value)
    if session_override is True:
        return True, "session"
    if session_override is False:
        return False, "session"
    return global_enabled, "global"


def format_micro_compact_status(
    global_value: Any,
    session_override: bool | None,
) -> str:
    """Format a stable status report for a future CLI/gateway/TUI surface."""
    enabled, source = effective_micro_compact(global_value, session_override)
    effective_label = "ON" if enabled else "OFF"
    global_label = "ON" if parse_global_micro_compact(global_value) else "OFF"
    source_label = "session" if source == "session" else "global (inherited)"
    return (
        f"Micro-compaction: {effective_label}\n"
        f"Source: {source_label}\n"
        f"Global: {global_label}"
    )


# Small aliases keep the semantic API easy to discover without introducing
# alternate implementations.
parse_micro_compact = parse_micro_compact_command
evaluate_micro_compact = effective_micro_compact
format_micro_status = format_micro_compact_status
