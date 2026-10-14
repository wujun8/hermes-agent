"""Credential-free route selections for new delegated dispatches."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DelegationRouteSelection:
    """Request a target; the host resolves credentials and retains fallback ownership.

    ``provider='inherit'`` retains the configured route. An optional endpoint
    hostname requirement is checked against the resolved bundle before use.
    """

    model: str
    provider: str = "inherit"
    api_mode: str | None = None
    required_endpoint_host: str | None = None
