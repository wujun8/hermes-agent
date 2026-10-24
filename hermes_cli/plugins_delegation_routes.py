"""Ledger-backed registration for delegation route selectors."""

import inspect
import logging
from copy import deepcopy

from hermes_cli.delegation_routes import DelegationRouteSelection

logger = logging.getLogger("tools.delegate_tool")
import re


def register_delegation_route_selector(ctx, name, select):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", name):
        raise ValueError("invalid delegation route selector name")
    if not callable(select) or inspect.iscoroutinefunction(select):
        raise TypeError("delegation route selector must be a synchronous callable")
    if name in ctx._manager._delegation_route_selectors:
        raise ValueError(f"delegation route selector already registered: {name}")
    return ctx._register_entry(
        "delegation_route_selector", name, ctx._manager._delegation_route_selectors,
        {"select": select, "plugin_key": ctx.plugin_id},
        "Plugin %s registered delegation route selector: %s", name,
    )


_ROUTE_KEYS = frozenset({
    "model", "provider", "base_url", "api_key", "api_mode", "request_overrides",
    "command", "args", "acp_command", "acp_args", "fallback_providers",
})


def resolve_delegation_route(cfg, parent_agent, *, allow_selectors=True):
    """Compatibility for existing selector plugins; new plugins use DelegationProvider."""
    from tools import delegate_tool
    from hermes_cli.plugins import _ensure_plugins_discovered
    from tools.delegate_tool_route_policy import resolve_policy_route

    if allow_selectors:
        selectors = list(_ensure_plugins_discovered()._delegation_route_selectors.items())
        for name, entry in selectors:
            settings = deepcopy({key: value for key, value in cfg.items() if key not in _ROUTE_KEYS})
            try:
                selection = entry["select"](settings)
                if selection is None:
                    continue
                if not isinstance(selection, DelegationRouteSelection):
                    raise ValueError("selector must return DelegationRouteSelection or None")
                if not isinstance(selection.provider, str) or not selection.provider.strip():
                    raise ValueError("route provider must be non-empty")
                route = deepcopy(cfg)
                if selection.provider.strip().lower() != "inherit":
                    for key in _ROUTE_KEYS - {"fallback_providers"}:
                        route.pop(key, None)
                    route["provider"] = selection.provider.strip()
                route["model"] = selection.model
                if selection.api_mode is not None:
                    route["api_mode"] = selection.api_mode
                route["route_policy"] = {
                    "required_endpoint_host": selection.required_endpoint_host,
                    "fallback_to_configured": selection.fallback_to_configured,
                    "failure_cooldown_seconds": selection.failure_cooldown_seconds,
                }
                return resolve_policy_route(route, parent_agent, cfg, name)
            except Exception:  # health: allow BLE001 -- legacy selectors fail open without logging secret-bearing exception details
                logger.warning("Delegation route selector %s unavailable; using the configured route", name)
                break
    return cfg, delegate_tool._resolve_delegation_credentials(cfg, parent_agent)
