---
sidebar_position: 21
title: Delegation route plugins
---

Native plugins can select a model/provider for newly dispatched children without
handling credentials or replacing `delegate_task`. Register a synchronous selector:

```python
from hermes_cli.delegation_routes import DelegationRouteSelection

def register(ctx):
    def select(settings):
        if not settings.get("use_fast_worker", False):
            return None
        return DelegationRouteSelection(model="fast-worker", provider="inherit")

    ctx.register_delegation_route_selector("fast-worker", select)
```

The callback receives a fresh dictionary of the current profile's delegation
settings. Host-owned route fields are omitted: `model`, `provider`, `base_url`,
`api_key`, `api_mode`, `request_overrides`, `command`, `args`, `acp_command`,
`acp_args`, and `fallback_providers`. Changing the dictionary changes neither
configuration nor an existing child. This is a convenience API for trusted native
code; in-process plugins are not a security sandbox.

Return `None` to leave the configured route unchanged. The first non-`None`
proposal owns selection. A callback error, invalid proposal, unavailable provider
or failed endpoint requirement keeps the configured route and skips later
selectors. Logs identify the selector without printing exception details that
may contain credentials.

`DelegationRouteSelection` is an immutable request with four fields:

| Field | Behavior |
| --- | --- |
| `model` | Required, non-empty target model. |
| `provider` | Defaults to `inherit`; otherwise a configured provider name. |
| `api_mode` | Optional registered transport override, validated by Hermes. |
| `required_endpoint_host` | Optional hostname that the resolved endpoint must match. |

`inherit` retains the configured provider, endpoint, credentials and request
overrides. For pure parent inheritance, Hermes derives transport for the selected
model from the parent's provider when possible. A separately named provider
resolves its own complete bundle; the previous provider's endpoint/key/transport
overrides are removed. The dispatch owner's reasoning and fallback configuration
continue with the selected route. The host validates the complete selection
before constructing any child.

Selectors affect ordinary new dispatches only. Control actions and trusted
internal calls with their own `credentials_cfg` bypass them. Nothing changes the
parent route, already-running children, tool schemas or cached prompt prefix.

Registrations belong to one profile and use the normal ownership ledger, including
unload and failed-load cleanup. The capability requires `plugins.isolation:
in_process`; host-isolated plugins are explicitly refused until a synchronous
route-selection protocol is available.

The standalone `hermes-delegation-schedule` plugin consumes this API for scheduled
model selection. Its calendar dependency and provider-specific configuration live
in that plugin, outside the Hermes source tree.
