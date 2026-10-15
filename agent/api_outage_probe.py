"""Private stdin contract for an outage probe of the live route."""

from copy import deepcopy
import json


def active_route_input(agent):
    """Serialize resolved routing only; never send conversation content or tool results."""
    client = getattr(agent, "_client_kwargs", None) or {}
    mode = getattr(agent, "api_mode", None)
    if mode not in {"chat_completions", "codex_responses"}:
        raise ValueError("active route transport cannot be probed by this command")
    key = client.get("api_key") or getattr(agent, "api_key", None)
    if not isinstance(key, str) or not key:
        raise ValueError("active route credentials are unavailable")
    overrides = deepcopy(getattr(agent, "request_overrides", None) or {})
    query = dict(client.get("default_query") or {})
    query.update(overrides.get("extra_query") or {})
    if query:
        overrides["extra_query"] = query
    payload = {
        "model": {
            "provider": getattr(agent, "requested_provider", None) or agent.provider,
            "default": agent.model,
            "base_url": str(agent.base_url),
            "api_mode": mode,
            "api_key": key,
            "default_headers": dict(client.get("default_headers") or {}),
        },
        "request_overrides": overrides,
    }
    return json.dumps(payload, ensure_ascii=True)


def active_probe_environment(agent):
    """Pin subprocess profile ownership without inheriting another profile's keys."""
    from pathlib import Path
    from hermes_constants import get_hermes_home
    from tools.environments.local import served_profile_child_env

    db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
    home = Path(db_path).parent if db_path else get_hermes_home()
    return served_profile_child_env(target_home=home, inherit_credentials=False)
