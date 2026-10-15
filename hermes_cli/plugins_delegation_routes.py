"""Ledger-backed registration for delegation route selectors."""

import inspect
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
