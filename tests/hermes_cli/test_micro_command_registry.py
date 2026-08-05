"""Central registry coverage for the session-scoped /micro command."""

from __future__ import annotations

from hermes_cli.commands import (
    COMMANDS,
    GATEWAY_KNOWN_COMMANDS,
    SUBCOMMANDS,
    gateway_help_lines,
    is_gateway_known_command,
    resolve_command,
)


def test_micro_is_canonical_registry_command_with_exact_subcommands():
    command = resolve_command("/micro")

    assert command is not None
    assert command.name == "micro"
    assert command.args_hint == "[on|off|inherit|status]"
    assert command.subcommands == ("on", "off", "inherit", "status")
    assert command.busy_policy == "reject"
    assert command.cli_only is False
    assert command.advertise_in_gateway_menu is False
    assert COMMANDS["/micro"]
    assert SUBCOMMANDS["/micro"] == ["on", "off", "inherit", "status"]
    assert "micro" in GATEWAY_KNOWN_COMMANDS
    assert is_gateway_known_command("micro")
    assert any(line.startswith("`/micro ") for line in gateway_help_lines())


def test_micro_resolves_case_insensitively_for_tui_command_resolution():
    plain = resolve_command("micro")
    upper = resolve_command("/MICRO")
    assert plain is not None and plain.name == "micro"
    assert upper is not None and upper.name == "micro"
