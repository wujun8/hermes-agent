"""Host-owned resolution of credential-free plugin delegation route requests."""

from copy import deepcopy
import logging

from hermes_cli.delegation_routes import DelegationRouteSelection
from utils import base_url_hostname

logger = logging.getLogger("tools.delegate_tool")
_ROUTE_KEYS = frozenset({
    "model", "provider", "base_url", "api_key", "api_mode", "request_overrides",
    "command", "args", "acp_command", "acp_args", "fallback_providers",
})


def _validated_selection(selection):
    from hermes_cli.runtime_provider import _parse_api_mode

    if not isinstance(selection, DelegationRouteSelection):
        raise ValueError("selector must return DelegationRouteSelection or None")
    if not isinstance(selection.model, str) or not selection.model.strip():
        raise ValueError("route model must be non-empty")
    if not isinstance(selection.provider, str) or not selection.provider.strip():
        raise ValueError("route provider must be non-empty")
    if not isinstance(selection.fallback_to_configured, bool):
        raise ValueError("invalid configured-route recovery policy")
    if (type(selection.failure_cooldown_seconds) is not int
            or not 1 <= selection.failure_cooldown_seconds <= 86400):
        raise ValueError("invalid route failure cooldown")
    mode = None
    if selection.api_mode is not None:
        if not isinstance(selection.api_mode, str):
            raise ValueError("invalid route api_mode")
        mode = _parse_api_mode(selection.api_mode)
        if mode is None:
            raise ValueError("invalid route api_mode")
    host = selection.required_endpoint_host
    if host is not None and (not isinstance(host, str) or not host.strip()
                             or base_url_hostname(f"https://{host}") != host.lower()):
        raise ValueError("invalid endpoint hostname requirement")
    return DelegationRouteSelection(
        model=selection.model.strip(), provider=selection.provider.strip(),
        api_mode=mode, required_endpoint_host=host.lower() if host else None,
        fallback_to_configured=selection.fallback_to_configured,
        failure_cooldown_seconds=selection.failure_cooldown_seconds,
    )


def _derive_inherited_mode(parent_agent, model, creds):
    """Preserve the inherited bundle while deriving transport for its new model."""
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
    except Exception:  # health: allow BLE001 -- transport derivation is optional and resolver exception details can contain credentials
        logger.debug("Could not derive delegated transport from the inherited provider")
    else:
        if runtime.get("api_mode"):
            creds["api_mode"] = runtime["api_mode"]


def _resolve_selection(cfg, parent_agent, selection):
    from tools import delegate_tool
    from tools.delegate_tool_config import _runtime_provider_credentials

    selected = _validated_selection(selection)
    route = deepcopy(cfg)
    if selected.provider.lower() == "inherit":
        route["model"] = selected.model
        if selected.api_mode is not None:
            route["api_mode"] = selected.api_mode
        creds = delegate_tool._resolve_delegation_credentials(route, parent_agent)
        if selected.api_mode is None and not route.get("provider") and not route.get("base_url"):
            _derive_inherited_mode(parent_agent, selected.model, creds)
    else:
        creds = _runtime_provider_credentials(
            {"provider": selected.provider, "model": selected.model}, None,
        )
        # Retain the dispatch owner's fallback and reasoning policies, but no
        # endpoint/key/request/transport fields from a different provider.
        for key in _ROUTE_KEYS - {"fallback_providers"}:
            route.pop(key, None)
        route.update(model=creds["model"], provider=creds["provider"], api_mode=creds["api_mode"])
    if selected.api_mode is not None:
        creds["api_mode"] = selected.api_mode
        route["api_mode"] = selected.api_mode
    if (selected.required_endpoint_host is not None
            and base_url_hostname(creds.get("base_url") or "") != selected.required_endpoint_host):
        raise ValueError("resolved endpoint does not meet the route requirement")
    return route, creds


def resolve_delegation_route(cfg, parent_agent, *, allow_selectors=True):
    """Select only new ordinary dispatches; internal credential owners bypass plugins."""
    from tools import delegate_tool
    from hermes_cli.plugins import _ensure_plugins_discovered

    if allow_selectors:
        selectors = list(_ensure_plugins_discovered()._delegation_route_selectors.items())
        for name, entry in selectors:
            settings = deepcopy({key: value for key, value in cfg.items() if key not in _ROUTE_KEYS})
            try:
                selection = entry["select"](settings)
                if selection is None:
                    continue
                route, creds = _resolve_selection(cfg, parent_agent, selection)
                if selection.fallback_to_configured:
                    from agent.delegation_route_recovery import prepare_selected_route
                    return prepare_selected_route(cfg, parent_agent, route, creds, selection, name)
                return route, creds
            except Exception:  # health: allow BLE001 -- plugin/resolver failures must fail open without logging secret-bearing exception details
                logger.warning("Delegation route selector %s unavailable; using the configured route", name)
                break
    return cfg, delegate_tool._resolve_delegation_credentials(cfg, parent_agent)
