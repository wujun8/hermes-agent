"""Recovery policy for optional delegated routes; credentials remain host-owned."""

from copy import deepcopy
import hashlib
import logging
import math
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from agent.delegation_route_health import ROUTE_HEALTH

logger = logging.getLogger(__name__)
POLICY_KEY = "_selected_route_recovery"
_UNAVAILABLE_REASONS = frozenset({
    "server_error", "overloaded", "timeout", "rate_limit", "upstream_rate_limit",
    "auth", "auth_permanent", "billing", "model_not_found", "model_entitlement",
    "upstream_blocked", "ssl_cert_verification", "incomplete_response",
})


def _route_identity(creds):
    provider = creds.get("requested_provider") or creds.get("provider")
    url = urlsplit(str(creds.get("base_url") or ""))
    query = {key: values[0] for key, values in parse_qs(url.query).items()}
    query.update(creds.get("default_query") or {})
    endpoint = urlunsplit((url.scheme, url.netloc, url.path.rstrip("/"), urlencode(sorted(query.items())), ""))
    return (str(provider or "").lower(), endpoint,
            str(creds.get("model") or ""), str(creds.get("api_mode") or ""))


def _agent_route_identity(agent):
    # Runtime endpoints can be properties backed by the client's state.
    route = {name: getattr(agent, name, None) for name in (
        "requested_provider", "provider", "base_url", "model", "api_mode",
    )}
    route["default_query"] = (getattr(agent, "_client_kwargs", None) or {}).get("default_query")
    return _route_identity(route)


def prepare_selected_route(cfg, parent, route, creds, selection, selector_name):
    """Attach the original route as first backup, or bypass a cooling-down selection."""
    from tools.delegate_tool import _parent_live_home, _resolve_delegation_credentials
    from tools.delegate_tool_config import _resolve_child_runtime, _resolve_child_fallback_chain

    normal = _resolve_delegation_credentials(cfg, parent)
    parent_key = getattr(parent, "api_key", None) or (getattr(parent, "_client_kwargs", None) or {}).get("api_key")
    def runtime_for(owner, bundle):
        return _resolve_child_runtime(
            parent, cfg, parent_key, model=bundle.get("model"), override_provider=bundle.get("provider"),
            override_base_url=bundle.get("base_url"), override_api_key=bundle.get("api_key"),
            override_api_mode=bundle.get("api_mode"), override_acp_command=bundle.get("acp_command"),
            override_acp_args=bundle.get("acp_args"), routing_cfg=owner,
        )
    runtime = runtime_for(cfg, normal)
    selected_runtime = runtime_for(route, creds)
    backup = {key: runtime[key] for key in ("model", "provider", "base_url", "api_key", "api_mode")}
    backup["request_overrides"] = deepcopy(normal.get("request_overrides") or {})
    if runtime.get("acp_command") or _route_identity(runtime) == _route_identity(selected_runtime):
        return cfg, normal
    home = _parent_live_home(parent)
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    fingerprint = hashlib.sha256(str(selected_runtime.get("api_key") or "").encode()).hexdigest()
    key = (str(home), selector_name, *_route_identity(selected_runtime), fingerprint)
    permit = ROUTE_HEALTH.acquire(
        key, cooldown_seconds=selection.failure_cooldown_seconds,
        probe_lease_seconds=max(60.0, _child_wait_budget(cfg)),
    )
    if permit is None:
        logger.info("Delegated route %s cooling down; retaining configured route", selection.model)
        return cfg, normal
    chain = _resolve_child_fallback_chain(parent, cfg, pinned=bool(
        normal.get("provider") or normal.get("base_url") or normal.get("model"))) or []
    route["fallback_providers"] = [backup, *deepcopy(chain)]
    route[POLICY_KEY] = {"permit": permit, "identity": _route_identity(selected_runtime)}
    return route, creds


def _child_wait_budget(delegation_cfg):
    try:
        limit = float(delegation_cfg.get("api_outage_max_wait_seconds", 600))
    except (ValueError, TypeError):
        limit = 600.0
    return limit if math.isfinite(limit) and limit > 0 else 600.0


def configure_child_recovery(child, route, delegation_cfg):
    """Bind route state explicitly across the child-construction thread boundary."""
    child._selected_route_recovery = (route or {}).get(POLICY_KEY)
    child._api_outage_max_wait_seconds = _child_wait_budget(delegation_cfg)


def record_route_failure(agent, reason):
    policy = getattr(agent, "_selected_route_recovery", None)
    value = getattr(reason, "value", reason)
    if policy and value in _UNAVAILABLE_REASONS and _agent_route_identity(agent) == policy["identity"]:
        ROUTE_HEALTH.fail(policy["permit"])
        agent._selected_route_recovery = None


def record_route_success(agent, message):
    """Only a usable response proves availability, never another model's probe."""
    if not (getattr(message, "content", None) or getattr(message, "tool_calls", None)):
        return
    agent._api_outage_wait_elapsed_seconds = 0.0
    policy = getattr(agent, "_selected_route_recovery", None)
    if policy and _agent_route_identity(agent) == policy["identity"]:
        ROUTE_HEALTH.succeed(policy["permit"])
