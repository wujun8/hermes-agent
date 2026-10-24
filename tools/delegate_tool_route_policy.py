"""Host-owned validation and recovery for opportunistic provider routes.

``route_policy`` is a local, JSON-only extension to DelegationProvider. Both the
official provider path and the legacy selector adapter use this resolver.
"""

from copy import deepcopy
import logging
from collections.abc import Mapping

from utils import base_url_hostname

logger = logging.getLogger("tools.delegate_tool")


def _validated_policy(route):
    policy = route.get("route_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("route_policy must be a mapping")
    fallback = policy.get("fallback_to_configured", False)
    if not isinstance(fallback, bool):
        raise ValueError("invalid configured-route recovery policy")
    cooldown = policy.get("failure_cooldown_seconds", 300)
    if type(cooldown) is not int or not 1 <= cooldown <= 86400:
        raise ValueError("invalid route failure cooldown")
    host = policy.get("required_endpoint_host")
    if host is not None and (not isinstance(host, str) or not host.strip()
                             or base_url_hostname(f"https://{host}") != host.lower()):
        raise ValueError("invalid endpoint hostname requirement")
    if not isinstance(route.get("model"), str) or not route["model"].strip():
        raise ValueError("route model must be non-empty")
    if route.get("provider") is not None and not isinstance(route["provider"], str):
        raise ValueError("route provider must be a string")
    return fallback, cooldown, host.lower() if host else None, _validated_mode(route.get("api_mode"))


def _validated_mode(raw_mode):
    from hermes_cli.runtime_provider import _parse_api_mode

    if raw_mode is None:
        return None
    if not isinstance(raw_mode, str):
        raise ValueError("invalid route api_mode")
    if not raw_mode.strip():
        return None
    mode = _parse_api_mode(raw_mode)
    if mode is None:
        raise ValueError("invalid route api_mode")
    return mode


def _derive_inherited_mode(parent_agent, model, creds):
    """Keep the parent's complete bundle while deriving transport for a new model."""
    from tools.delegate_tool_config import _inherit_parent_endpoint
    from hermes_cli.runtime_provider import resolve_runtime_provider

    provider = (getattr(parent_agent, "requested_provider", None)
                or getattr(parent_agent, "provider", None))
    if not provider:
        return
    key = getattr(parent_agent, "api_key", None)
    kwargs = getattr(parent_agent, "_client_kwargs", None)
    if not key and isinstance(kwargs, dict):
        key = kwargs.get("api_key")
    endpoint, key = _inherit_parent_endpoint(
        parent_agent, getattr(parent_agent, "base_url", None), key,
    )
    try:
        runtime = resolve_runtime_provider(
            requested=provider, target_model=model,
            explicit_base_url=endpoint, explicit_api_key=key,
        )
    except Exception:  # health: allow BLE001 -- optional derivation; resolver exceptions can contain credentials
        logger.debug("Could not derive delegated transport from the inherited provider")
    else:
        if runtime.get("api_mode"):
            creds["api_mode"] = runtime["api_mode"]


def resolve_policy_route(route, parent_agent, config, backend):
    """Resolve a backend's route; only explicit recovery opt-in may retain config."""
    from tools.delegate_tool import _resolve_delegation_credentials
    from tools.delegate_tool_config import _inherit_parent_endpoint
    from agent.delegation_route_recovery import prepare_selected_route

    policy = route.get("route_policy")
    recover = isinstance(policy, Mapping) and policy.get("fallback_to_configured") is True
    try:
        _fallback, cooldown, host, mode = _validated_policy(route)
        selected = deepcopy(route)
        selected["model"] = selected["model"].strip()
        creds = _resolve_delegation_credentials(selected, parent_agent)
        if mode is not None:
            creds["api_mode"] = selected["api_mode"] = mode
        elif not selected.get("provider") and not selected.get("base_url"):
            _derive_inherited_mode(parent_agent, selected["model"], creds)
        endpoint = creds.get("base_url")
        if host and not endpoint and not creds.get("provider"):
            endpoint, _key = _inherit_parent_endpoint(
                parent_agent, getattr(parent_agent, "base_url", None), getattr(parent_agent, "api_key", None),
            )
        if host and base_url_hostname(endpoint or "") != host:
            raise ValueError("resolved endpoint does not meet the route requirement")
        if recover:
            return prepare_selected_route(
                config, parent_agent, selected, creds, cooldown_seconds=cooldown, backend=backend,
            )
        return selected, creds
    except Exception:  # health: allow BLE001 -- optional-route failures retain config without secret-bearing exception details
        if not recover:
            raise
        logger.warning("Delegation backend %s optional route unavailable; using the configured route", backend)
    return config, _resolve_delegation_credentials(config, parent_agent)
