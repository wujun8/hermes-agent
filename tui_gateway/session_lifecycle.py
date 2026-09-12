"""Session lifecycle: active-session slot leases, finalize/teardown/close, turn interrupt,
WS-orphan reap scheduling, transport-scoped close. Bodies are rebound onto server.py's
globals at install time (method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import logging

import contextlib
from dataclasses import dataclass
from typing import NamedTuple

from .method_ctx import bind_module


def _notify_session_boundary(event_type: str, session_id: str | None, platform: str | None = None) -> None:
    """Fire session lifecycle hooks with CLI parity."""
    with contextlib.suppress(Exception):
        from hermes_cli.lifecycle import finalize_session, invoke_hook
        if event_type == "on_session_finalize":
            finalize_session(session_id=session_id, platform=_resolve_agent_platform(platform))
        else:
            invoke_hook(event_type, session_id=session_id, platform=_resolve_agent_platform(platform))


_SESSION_OWNERSHIP_UNAVAILABLE = "Hermes could not safely reserve this session. Try again."
_AUTOMATIC_SESSION_END_REASONS = frozenset({"ws_orphan_reap", "ws_disconnect", "idle_timeout", "lru_evict", "tui_shutdown"})


class _MicroControlTarget(NamedTuple):
    """Frozen identity/routing snapshot for one live ``/micro`` invocation."""

    sid: str
    session: dict
    session_key: str
    agent: Any
    agent_session_id: str
    profile_home: str | None
    generation: int
    token: object


class _CompressionReanchorIdentity(NamedTuple):
    """Exact identity reserved while a deferred compression re-anchor runs."""

    sid: str
    session: dict
    generation: int
    agent: Any
    new_session_id: str
    control_token: object
    marker: dict
    old_session_key: str
    profile_home: str | None


@dataclass(frozen=True)
class _MicroControlReleaseResult:
    """Structured result for releasing one live ``/micro`` control lease."""

    success: bool
    warning: str = ""
    reanchored: bool = False

    def __bool__(self) -> bool:
        return self.success


_SESSION_CONTROL_BUSY = object()
_MICRO_TURN_BUSY = object()
_MICRO_TURN_BUSY_MESSAGE = (
    "session busy — /micro can't run mid-turn; interrupt the current turn first"
)
_MICRO_CONTROL_BUSY_MESSAGE = "session busy — /micro control command already in progress"
_MANUAL_COMPRESSION_TURN_BUSY_MESSAGE = (
    "session busy — /interrupt the current turn before /compress"
)
_DEFERRED_COMPRESSION_ROTATION_KEY = "_deferred_compression_rotation"
_MANUAL_COMPRESSION_CONTROL_KEY = "_manual_compression_control"
_MICRO_CONTROL_STATE_KEY = "_micro_control_state"
_COMPRESSION_SYNC_UNCHANGED = "unchanged"
_COMPRESSION_SYNC_APPLIED = "applied"
_COMPRESSION_SYNC_DEFERRED = "deferred"
_COMPRESSION_SYNC_FAILED = "failed"
_COMPRESSION_SYNC_IDENTITY_CHANGED = "identity_changed"
_MICRO_OVERRIDE_UNKNOWN = object()


def _claim_active_session_slot(
    session_key: str, *, live_session_id: str, surface: str = "tui", profile_home: str | Path | None = None
) -> tuple[Any, str | None]:
    try:
        from hermes_cli.active_sessions import try_acquire_active_session
        return try_acquire_active_session(
            session_id=session_key, surface=surface, config=_load_cfg(), registry_home=profile_home,
            metadata={"live_session_id": live_session_id, "bot_live_delivery_consumer": True},
            track_liveness=str(surface or "").strip().lower() == "desktop")
    except Exception as exc:
        logger.warning("Failed to claim active session slot: %s", exc)
        # Fail CLOSED: an errored claim has NOT proven the session unowned; lease-less = silent double-writer hole.
        # Fail CLOSED regardless of surface: per-session exclusivity is a correctness guarantee (see
        # PER_SESSION_EXCLUSIVE_SUBMIT), and a claim that errors out has NOT proven the session is unowned.
        # Proceeding without a lease here is the silent double-writer hole flagged in the #94595 review
        # (blocker 2).
        return (None, _SESSION_OWNERSHIP_UNAVAILABLE)


def _active_session_slot_lock(session: dict) -> threading.Lock:
    """Serialize lazy active-session lease acquisition without holding history_lock over I/O."""
    history_lock = session.get("history_lock")
    if history_lock is None:
        return session.setdefault("_active_session_slot_lock", threading.Lock())
    with history_lock:
        lock = session.get("_active_session_slot_lock")
        if lock is None:
            lock = threading.Lock()
            session["_active_session_slot_lock"] = lock
        return lock


def _release_unowned_active_session_lease(lease, *, sid: str) -> bool:
    """Release a lease that lost the race before it could be installed on its session."""
    if lease is None:
        return True
    try:
        lease.release()
    except BaseException as exc:
        logger.warning(
            "active-session lease cleanup failed before prompt ownership was installed: sid=%s error=%s",
            sid,
            exc,
        )
        return False
    return True


def _claim_active_session_slot_for_prompt(
    sid: str, session: dict
) -> tuple[str | None, Any | None]:
    """Claim a prompt's active-session slot and return only a lease installed by this call."""
    with _active_session_slot_lock(session):
        history_lock = session.get("history_lock")
        if history_lock is not None:
            with history_lock:
                if (
                    session.get("active_session_lease") is not None
                    or session.get("_closing")
                    or session.get("_finalized")
                ):
                    return None, None
        elif (
            session.get("active_session_lease") is not None
            or session.get("_closing")
            or session.get("_finalized")
        ):
            return None, None

        lease, limit_message = _claim_active_session_slot(
            str(session.get("session_key") or ""),
            live_session_id=sid,
            surface=_session_source(session),
            profile_home=session.get("profile_home"),
        )
        if limit_message is not None:
            return limit_message, None
        if lease is None:
            return None, None

        installed = False
        # The external claim is intentionally outside _sessions_lock. Revalidate the exact registry identity
        # before attaching so a concurrent pop/close cannot leave a lease on a detached session.
        with _sessions_lock:
            if _sessions.get(sid) is session:
                with (history_lock or contextlib.nullcontext()):
                    if (
                        session.get("active_session_lease") is None
                        and not session.get("_closing")
                        and not session.get("_finalized")
                    ):
                        session["active_session_lease"] = lease
                        installed = True
        if not installed:
            _release_unowned_active_session_lease(lease, sid=sid)
            return None, None
        return None, lease


def _ensure_active_session_slot(sid: str, session: dict) -> str | None:
    """Claim this session's cap slot on its first real turn; None when ok. session.create/resume deliberately
    do NOT claim: tile paints, reconnect-resumes and abandoned drafts would hold invisible slots (no DB row)
    that starve the messaging gateway sharing the cap. Anything holding a slot must be user-visible."""
    limit_message, _lease = _claim_active_session_slot_for_prompt(sid, session)
    return limit_message


def _lease_retry(attempts: int, fn) -> Exception | None:
    """Call ``fn`` up to ``attempts`` times (50ms*n backoff: registry writes contend across processes); last
    exception when every try failed, else None."""
    for attempt in range(attempts):
        try:
            fn()
            return None
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.05 * (attempt + 1))
    return last_error


def _release_active_session_lease(lease) -> bool:
    """Release one lease with the hosted-runtime retry policy."""
    if lease is None:
        return True
    attempts = 3 if getattr(lease, "track_liveness", False) else 1
    for attempt in range(attempts):
        try:
            lease.release()
            break
        except Exception:
            if attempt + 1 >= attempts:
                logger.warning("Failed to release active session slot", exc_info=True)
                return False
            time.sleep(0.05 * (attempt + 1))
    return bool(getattr(lease, "released", True) or not getattr(lease, "enabled", True))


def _release_active_session_slot(
    session: dict | None,
    *,
    sid: str | None = None,
    expected_lease=None,
    expected_lease_token=None,
    reject_active: bool = False,
) -> bool:
    """Release a lease, optionally requiring exact prompt ownership and an idle session."""
    if not session:
        return True
    if expected_lease is None:
        lease = session.get("active_session_lease")
        try:
            released = _release_active_session_lease(lease)
        except BaseException as exc:
            logger.warning("Failed to release active session slot: %s", exc)
            return False
        if released and session.get("active_session_lease") is lease:
            session.pop("active_session_lease", None)
        return released

    history_lock = session.get("history_lock")
    if history_lock is None:
        return False
    try:
        with history_lock:
            current = session.get("active_session_lease")
            token_matches = (
                expected_lease_token is None
                or getattr(current, "lease_id", None) == expected_lease_token
            )
            if current is not expected_lease or not token_matches:
                return False
            if reject_active and session.get("running"):
                return False
            session_key = str(session.get("session_key") or "")

        # Registry and child-run checks stay outside history_lock; cleanup must
        # never deadlock a prompt/control transition.
        if reject_active and _child_run_active(session_key):
            return False
        if sid is not None:
            with _sessions_lock:
                if _sessions.get(sid) is not session:
                    return False

        with history_lock:
            current = session.get("active_session_lease")
            token_matches = (
                expected_lease_token is None
                or getattr(current, "lease_id", None) == expected_lease_token
            )
            if current is not expected_lease or not token_matches:
                return False
            if reject_active and session.get("running"):
                return False
            session.pop("active_session_lease", None)

        try:
            released = _release_active_session_lease(expected_lease)
        except BaseException as exc:
            released = False
            logger.warning(
                "Failed to release prompt active-session lease: sid=%s error=%s", sid, exc
            )
        if not released:
            # Preserve the exact lease for teardown/orphan reconciliation retry.
            with history_lock:
                if session.get("active_session_lease") is None:
                    session["active_session_lease"] = expected_lease
            return False
        return True
    except BaseException as exc:
        logger.warning(
            "Prompt active-session lease validation failed: sid=%s error=%s", sid, exc
        )
        return False


def _release_prompt_active_session_slot(
    sid: str, session: dict, lease, *, lease_token=None
) -> bool:
    """Best-effort exact cleanup for a prompt rejected by live control."""
    if lease is None:
        return True
    try:
        return _release_active_session_slot(
            session,
            sid=sid,
            expected_lease=lease,
            expected_lease_token=lease_token,
            reject_active=True,
        )
    except BaseException as exc:
        logger.warning("Prompt active-session lease cleanup failed: sid=%s error=%s", sid, exc)
        return False


def _own_live_lease_ids(*, exclude=None) -> set[str]:
    """Snapshot leases still backed by this process's live session records."""
    with _sessions_lock:
        return {str(lease.lease_id) for session in _sessions.values()
                if (lease := session.get("active_session_lease")) is not None and lease is not exclude}


@contextlib.contextmanager
def _other_runtime_lease_guard(session_id: str, session: dict):
    """Release this runtime and lock sibling ownership through the DB write. Yields True (another runtime owns
    the lifecycle -> preserve) when the guard can't be loaded/entered in 3 tries: unknown ownership never ends a row."""
    lease = session.get("active_session_lease")
    try:
        from hermes_cli.active_sessions import active_session_liveness_guard, release_active_session_liveness_guard
    except Exception as exc:
        logger.warning("Failed to load active session ownership guard; preserving session %s: %s", session_id, exc)
        yield True
        return
    stack = contextlib.ExitStack()
    active: list = []
    own_live_lease_ids = _own_live_lease_ids(exclude=lease)

    def _enter() -> None:
        stack.close()  # drop anything a half-failed previous attempt left behind
        if lease is not None and getattr(lease, "enabled", False):
            guard = release_active_session_liveness_guard(lease, session_id, own_live_lease_ids=own_live_lease_ids)
        else:
            guard = active_session_liveness_guard(
                session_id, registry_home=session.get("profile_home"), own_live_lease_ids=own_live_lease_ids)
        active[:] = [stack.enter_context(guard)]

    if (last_error := _lease_retry(3, _enter)) is not None:
        stack.close()
        logger.warning("Failed to inspect active session leases; preserving session %s: %s", session_id, last_error)
        yield True
        return
    try:
        yield active[0]
    finally:
        stack.close()
        if lease is not None and getattr(lease, "released", False) and session.get("active_session_lease") is lease:
            session.pop("active_session_lease", None)


def _transfer_active_session_slot(sid: str, session: dict, *, new_session_id: str) -> bool:
    if not new_session_id:
        return False
    lease = session.get("active_session_lease")
    if lease is None:
        return True
    try:
        from hermes_cli.active_sessions import transfer_active_session
        if transfer_active_session(lease, session_id=new_session_id, metadata={
                "live_session_id": sid, "bot_live_delivery_consumer": True}):
            return True
    except Exception:
        logger.debug("Failed to transfer active session slot", exc_info=True)
    if getattr(lease, "track_liveness", False):
        return False
    # Fallback (entry pruned / pid-check transiently failed): reserve the new slot BEFORE releasing the old one so
    # a gateway at the cap can't grab the freed slot and leave this session lease-less; on failure KEEP the old lease.
    # See #49041.
    new_lease, limit_message = _claim_active_session_slot(
        new_session_id, live_session_id=sid, surface=_session_source(session), profile_home=session.get("profile_home"))
    if new_lease is None:
        if limit_message:
            logger.warning("Compression session lease re-anchor failed (kept old lease): sid=%s new_session_id=%s reason=%s",
                           sid, new_session_id, limit_message)
        return False
    if (old := session.pop("active_session_lease", None)) is not None and (err := _lease_retry(1, old.release)):
        logger.debug("Failed to release stale active session slot", exc_info=err)
    session["active_session_lease"] = new_lease
    return True


def _session_micro_config_loader_for_profile(profile_home: str | None):
    """Return a readonly micro config loader bound to one profile path."""
    from hermes_cli.config import load_config_readonly

    profile_home = str(profile_home or "").strip()
    if not profile_home:
        return load_config_readonly

    def load_profile_config():
        token = set_hermes_home_override(profile_home)
        try:
            return load_config_readonly()
        finally:
            reset_hermes_home_override(token)

    return load_profile_config


def _session_micro_config_loader(session: dict):
    """Return the readonly micro policy config loader for this session."""
    return _session_micro_config_loader_for_profile(session.get("profile_home"))


def _claim_live_micro_control(sid: str, session: dict | None):
    """Atomically claim an idle live session for one direct ``/micro`` call."""
    if session is None:
        return None, None
    history_lock = session.get("history_lock")
    if history_lock is None:
        return None, None

    with history_lock:
        active = bool(session.get("running"))
        session_key = str(session.get("session_key") or "")
        if not active:
            active = _child_run_active(session_key)
        if active:
            return None, _MICRO_TURN_BUSY
        if session.get("_session_control_inflight") is not None:
            return None, _SESSION_CONTROL_BUSY

        agent = session.get("agent")
        if agent is None:
            return None, None
        agent_key = str(getattr(agent, "session_id", "") or "")
        if not session_key.strip() or not agent_key.strip():
            return (
                None,
                "Micro-compaction error: live session identity is incomplete; refusing to mutate",
            )
        if session_key != agent_key:
            return (
                None,
                "Micro-compaction error: live session identity mismatch; refusing to mutate",
            )

        token = object()
        target = _MicroControlTarget(
            sid=str(sid or ""),
            session=session,
            session_key=session_key,
            agent=agent,
            agent_session_id=agent_key,
            profile_home=(
                str(session.get("profile_home"))
                if session.get("profile_home") is not None
                else None
            ),
            generation=int(session.get("session_generation", 0) or 0),
            token=token,
        )
        session["_session_control_inflight"] = token
        session[_MICRO_CONTROL_STATE_KEY] = {}
        return target, None


def _claim_manual_compression_control(session: dict):
    """Reserve the exact session before any direct manual compression work."""
    history_lock = session.get("history_lock")
    if history_lock is None:
        return None, _SESSION_CONTROL_BUSY
    with history_lock:
        active = bool(session.get("running"))
        session_key = str(session.get("session_key") or "")
        if not active:
            active = _child_run_active(session_key)
        if active:
            return None, _MICRO_TURN_BUSY
        if session.get("_session_control_inflight") is not None:
            return None, _SESSION_CONTROL_BUSY
        token = object()
        session["_session_control_inflight"] = token
        session[_MANUAL_COMPRESSION_CONTROL_KEY] = token
        return token, None


def _manual_compression_busy_text(claim_error) -> str:
    if claim_error is _MICRO_TURN_BUSY:
        return _MANUAL_COMPRESSION_TURN_BUSY_MESSAGE
    return _MICRO_CONTROL_BUSY_MESSAGE


def _marker_matches_micro_target(marker: dict | None, target: _MicroControlTarget) -> bool:
    if not isinstance(marker, dict) or marker.get("control_token") is not target.token:
        return False
    try:
        return int(marker.get("generation", -1)) == target.generation
    except (TypeError, ValueError):
        return False


def _micro_rotation_override_warning(
    target: _MicroControlTarget,
    marker: dict,
    state: dict | None,
) -> str | None:
    """Fail closed if a forced rotation left /micro's exact override on old row."""
    if not isinstance(state, dict) or not state.get("mutation_succeeded"):
        return None
    if "requested_override" not in state:
        return None
    requested = state["requested_override"]
    new_session_id = str(marker.get("agent_session_id") or "")
    if not new_session_id:
        return (
            "Micro-compaction warning: forced compression rotation produced no "
            "current session id; retry /micro on the current session"
        )
    try:
        with _session_db(target.session) as db:
            row = db.get_session(new_session_id) if db is not None else None
            if row is None:
                return (
                    "Micro-compaction warning: /micro changed only the old row "
                    "because compression rotated the session; retry /micro on the current session"
                )
            getter = getattr(db, "session_micro_compact_override", None)
            actual = getter(row) if callable(getter) else None
    except Exception as exc:
        return (
            "Micro-compaction warning: could not verify the current continuation "
            f"override after rotation ({exc}); retry /micro on the current session"
        )
    if actual is not requested:
        return (
            "Micro-compaction warning: /micro changed only the old row because "
            "compression rotated the session; retry /micro on the current session"
        )
    live_override = getattr(
        target.agent, "micro_compact_override", _MICRO_OVERRIDE_UNKNOWN
    )
    if live_override is not _MICRO_OVERRIDE_UNKNOWN and live_override is not requested:
        return (
            "Micro-compaction warning: the current Agent policy did not retain "
            "the requested override after rotation; retry /micro on the current session"
        )
    return None


def _release_live_micro_control(
    target: _MicroControlTarget,
) -> _MicroControlReleaseResult:
    """Release one live ``/micro`` claim, retaining it through re-anchor I/O."""
    session = target.session
    marker = None
    state = None
    reanchor_token = None
    expected_reanchor = None

    def _identity_warning() -> _MicroControlReleaseResult:
        return _MicroControlReleaseResult(
            False,
            "Micro-compaction warning: live session changed (identity changed) during "
            "deferred re-anchor; refusing to mutate the replacement",
        )

    def _canonical_state() -> bool:
        with _sessions_lock:
            return (
                _sessions.get(target.sid) is session
                and session.get("_session_control_inflight") is target.token
                and session.get("_deferred_compression_rotation") is None
                and session.get("session_key") == str(marker.get("agent_session_id") or "")
                and session.get("agent") is target.agent
                and str(getattr(target.agent, "session_id", "") or "")
                == str(marker.get("agent_session_id") or "")
                and int(session.get("session_generation", 0) or 0) >= target.generation + 1
            )

    try:
        with session["history_lock"]:
            lease_matches = (
                session.get("_session_control_inflight") is target.token
                and session.get("agent") is target.agent
                and int(session.get("session_generation", 0) or 0) == target.generation
                and (
                    str(session.get("profile_home"))
                    if session.get("profile_home") is not None
                    else None
                )
                == target.profile_home
            )
            if not lease_matches:
                return _MicroControlReleaseResult(
                    False,
                    "Micro-compaction warning: live session changed before the "
                    "completion could clear its control lease",
                )
            candidate = session.get(_DEFERRED_COMPRESSION_ROTATION_KEY)
            if _marker_matches_micro_target(candidate, target):
                marker = candidate
                reanchor_token = target.token
                new_session_id = str(marker.get("agent_session_id") or "")
                if not new_session_id or str(
                    getattr(target.agent, "session_id", "") or ""
                ) != new_session_id:
                    return _identity_warning()
                expected_reanchor = _CompressionReanchorIdentity(
                    sid=target.sid,
                    session=session,
                    generation=target.generation,
                    agent=target.agent,
                    new_session_id=new_session_id,
                    control_token=target.token,
                    marker=marker,
                    old_session_key=target.session_key,
                    profile_home=target.profile_home,
                )
            state = session.get(_MICRO_CONTROL_STATE_KEY)
            if marker is None:
                session.pop("_session_control_inflight", None)

        with _sessions_lock:
            registry_same = _sessions.get(target.sid) is session
        if not registry_same:
            return _identity_warning()

        if marker is None:
            with session["history_lock"]:
                unchanged = (
                    session.get("session_key") == target.session_key
                    and session.get("agent") is target.agent
                    and str(getattr(target.agent, "session_id", "") or "")
                    == target.agent_session_id
                    and int(session.get("session_generation", 0) or 0)
                    == target.generation
                )
                if session.get(_MICRO_CONTROL_STATE_KEY) is state:
                    session.pop(_MICRO_CONTROL_STATE_KEY, None)
            if unchanged:
                return _MicroControlReleaseResult(True)
            return _MicroControlReleaseResult(
                False,
                "Micro-compaction warning: live session changed while /micro "
                "was running; refusing to report a new target mutation",
            )

        try:
            sync_status = _sync_session_key_after_compress(
                target.sid,
                session,
                control_token=reanchor_token,
                expected_reanchor=expected_reanchor,
            )
        except BaseException as exc:
            return _MicroControlReleaseResult(
                False,
                "Micro-compaction warning: deferred compression rotation remains "
                f"pending after release ({exc}); retry /micro on the current session",
            )
        if sync_status == _COMPRESSION_SYNC_IDENTITY_CHANGED:
            return _identity_warning()
        if sync_status not in {_COMPRESSION_SYNC_APPLIED, _COMPRESSION_SYNC_UNCHANGED}:
            return _MicroControlReleaseResult(
                False,
                "Micro-compaction warning: deferred compression rotation remains "
                "pending after release; retry /micro on the current session",
            )

        if not _canonical_state():
            return _identity_warning()

        warning = _micro_rotation_override_warning(target, marker, state)
        if not _canonical_state():
            return _identity_warning()
        with _sessions_lock:
            if session.get(_MICRO_CONTROL_STATE_KEY) is state:
                session.pop(_MICRO_CONTROL_STATE_KEY, None)
        if warning:
            return _MicroControlReleaseResult(
                False,
                warning,
                reanchored=True,
            )
        return _MicroControlReleaseResult(True, reanchored=True)
    except BaseException as exc:
        return _MicroControlReleaseResult(
            False,
            "Micro-compaction warning: could not complete live-control release "
            f"({exc}); retry /micro on the current session",
        )
    finally:
        if reanchor_token is not None:
            try:
                with session["history_lock"]:
                    if session.get("_session_control_inflight") is reanchor_token:
                        session.pop("_session_control_inflight", None)
            except BaseException:
                pass


def _try_claim_session_control(session: dict):
    """Reserve one session-wide control transition without holding I/O locks."""
    with session["history_lock"]:
        if session.get("_session_control_inflight") is not None:
            return None
        token = object()
        session["_session_control_inflight"] = token
        return token


def _release_session_control(session: dict, token) -> None:
    """Release a non-micro transition reservation, best effort."""
    try:
        with session["history_lock"]:
            if session.get("_session_control_inflight") is token:
                session.pop("_session_control_inflight", None)
            if session.get(_MANUAL_COMPRESSION_CONTROL_KEY) is token:
                session.pop(_MANUAL_COMPRESSION_CONTROL_KEY, None)
    except BaseException:
        pass


# Sources this backend must never end in state.db: the messaging gateway owns those sessions and the TUI is only
# a viewer (ending one causes the Groundhog Day loop, see _finalize_session). Self-created/CLI sources are NOT gateway-owned.
# Sources the TUI backend itself creates ("tui", plus whatever a client passes as its own ``source``) and
# the CLI's own sessions are NOT gateway-owned. See #60609.
_NON_GATEWAY_SOURCES = frozenset({
    "", "tui", "cli", "webui", "desktop", "cron", "kanban", "subagent", "test",
    "local", "acp", "webhook", "api_server", "msgraph_webhook"})


def _is_gateway_owned_source(source: str) -> bool:
    """True when ``source`` resolves to a gateway ``Platform`` (enum member or plugin via ``Platform._missing_``, so
    new platforms are covered automatically); self-owned Platform members (local/webhook/api_server) are excluded."""
    src = (source or "").strip().lower()
    if src in _NON_GATEWAY_SOURCES:
        return False
    try:
        from gateway.config import Platform
        Platform(src)  # raises ValueError for arbitrary non-platform strings
        return True
    except Exception:
        return False


def _lifecycle_own_sid(session: dict, sid_hint: str = "") -> str:
    """Live UI sid for ``session``: hint, stamped ``_sid``, else registry scan."""
    own_sid = str(sid_hint or session.get("_sid") or "")
    if not own_sid:
        with contextlib.suppress(Exception), _sessions_lock:
            own_sid = next((cand_sid for cand_sid, cand in _sessions.items() if cand is session), "")
    return own_sid


def _lock_vault_managers(session: dict) -> None:
    """A per-session unlock ends with the session that made it; siblings in the same profile keep theirs."""
    try:
        from agent.vault_backends import unlock

        if sid := session.get("_sid"):
            unlock.release_session(sid)
    except Exception:
        logging.getLogger(__name__).debug("vault manager lock on session end failed", exc_info=True)


def _finalize_session(session: dict | None, end_reason: str = "tui_close") -> None:
    """Best-effort finalize hook + memory commit; mirrors the CLI exit path so a force-quit mid-turn (double
    Ctrl-C, terminal close, SIGHUP) loses nothing."""
    if not session or session.get("_finalized"):
        return
    session["_finalized"] = True
    _lock_vault_managers(session)
    if (history_ready := session.get("resume_history_ready")) is not None and not history_ready.is_set():
        session["resume_history_error"] = "session resume cancelled"
        history_ready.set()
    _desktop_automatic_cleanup = (
        end_reason in _AUTOMATIC_SESSION_END_REASONS and _session_source(session).strip().lower() == "desktop")
    # Automatic Desktop cleanup releases its lease inside the lifecycle guard below; other paths keep force/end semantics.
    if not _desktop_automatic_cleanup:
        _release_active_session_slot(session)
    if (stop_event := session.get("_notif_stop")) is not None:
        stop_event.set()
    agent = session.get("agent")
    with (session.get("history_lock") or contextlib.nullcontext()):
        history = list(session.get("history", []))
    # Persist via ``_persist_session``'s marker-based dedup (gateway-shutdown flush contract). Do NOT pass
    # ``conversation_history``: ``session["history"]`` and ``_session_messages`` alias the SAME list after a turn, so
    # the flush would treat every message as durable and skip it — data loss when finalize is the sole persist path.
    if hasattr(agent, "_persist_session") and (snapshot := getattr(agent, "_session_messages", None)):
        with contextlib.suppress(Exception):
            agent._persist_session(snapshot)
    # interrupted=True so crash-recovery plugins can flush state (mirrors cli.py atexit).
    if agent is not None:
        with contextlib.suppress(Exception):
            from hermes_cli.lifecycle import invoke_hook
            invoke_hook(
                "on_session_end", completed=False, interrupted=True,
                session_id=getattr(agent, "session_id", None) or session.get("session_key", ""),
                model=getattr(agent, "model", "unknown"), platform=getattr(agent, "platform", None) or "tui")
    if agent is not None and history and hasattr(agent, "commit_memory_session"):
        with contextlib.suppress(Exception):
            agent.commit_memory_session(history)

    session_key = session.get("session_key")
    session_id = getattr(agent, "session_id", None) or session_key
    _notify_session_boundary("on_session_finalize", session_id, _session_source(session))
    # End the state.db row so it doesn't linger as a ghost in /resume. Use session_id (agent.session_id), not
    # session_key: after compression the key may be the stale ended parent while session_id is the live continuation.
    # Fix for #20001.
    if _desktop_automatic_cleanup and not session_id:
        _release_active_session_slot(session)
    _lifecycle_guard = (_other_runtime_lease_guard(session_id, session)
                        if _desktop_automatic_cleanup and session_id else contextlib.nullcontext(False))
    with _lifecycle_guard as _other_runtime_owns_lifecycle:
        _tui_owns_lifecycle = not _other_runtime_owns_lifecycle
        if _other_runtime_owns_lifecycle:
            logger.info("Preserving session %s during %s: another backend owns an active lease", session_id, end_reason)
        if session_id:
            # The *session's* profile state.db (app-global remote mode), not the launch profile's.
            with contextlib.suppress(Exception), _session_db(session) as db:
                if db is not None:
                    # Never end gateway-originated sessions: Groundhog Day loop (gateway self-heals to the parent,
                    # compression splits back to the reaped child, forever).
                    if _is_gateway_owned_source((db.get_session(session_id) or {}).get("source", "")):
                        _tui_owns_lifecycle = False
                    elif _tui_owns_lifecycle:
                        db.end_session(session_id, end_reason)
    # In-flight async delegations end WITH the session (no return address left). Always interrupt by THIS live UI
    # sid; by durable session_key only when the TUI owns the lifecycle — a viewer tab must not kill gateway work.
    with contextlib.suppress(Exception):
        from tools.async_delegation import interrupt_for_session
        interrupt_for_session(
            session_key=str(session_key or "") if _tui_owns_lifecycle else "",
            origin_ui_session_id=_lifecycle_own_sid(session), reason=end_reason)
    # Close the slash-worker in this single ``_finalized``-guarded chokepoint (a direct caller can't leak it); idempotent.
    with contextlib.suppress(Exception):
        if worker := session.get("slash_worker"):
            worker.close()


# End reasons where the BACKEND reclaimed a session the client never asked to close (else its next prompt fails
# against a forgotten id). Client-initiated reasons (``tui_close`` etc.) are deliberately absent.
_RECLAIM_END_REASONS = frozenset({"idle_timeout", "lru_evict", "ws_orphan_reap"})


def _announce_session_reclaimed(session: dict, end_reason: str) -> None:
    """Tell connected clients a session was reclaimed out from under them. Broadcast, not session-targeted: reap
    paths run on timer threads with no contextvar binding and no live transport, so ``_emit`` would hit stdio."""
    if end_reason not in _RECLAIM_END_REASONS:
        return
    try:
        _broadcast_global_event("session.reclaimed", {
            "session_id": str(session.get("_sid") or ""),
            "stored_session_id": str(session.get("session_key") or ""),
            "reason": end_reason})
    except Exception:
        logger.debug("session.reclaimed broadcast failed", exc_info=True)


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close") -> None:
    """Fully tear down a session: finalize, unregister notifier, close agent (``session.close`` + WS reaper). The
    slash-worker is closed in ``_finalize_session`` (the single chokepoint), NOT here. Idempotent via ``_finalized``."""
    if not session:
        return
    _finalize_session(session, end_reason=end_reason)
    _announce_session_reclaimed(session, end_reason)
    with contextlib.suppress(Exception):
        from tools.approval import unregister_gateway_notify
        if key := session.get("session_key"):
            unregister_gateway_notify(key)
    with contextlib.suppress(Exception):
        if hasattr(agent := session.get("agent"), "close"):
            agent.close()


def _attach_worker(sid: str, session: dict, worker) -> None:
    """Store worker on session iff sid still maps to it, else close it (a concurrent teardown popped the session)."""
    with _sessions_lock:
        if _sessions.get(sid) is session:
            session["slash_worker"] = worker
            return
    worker.close()


# Wall-clock timestamps, like session last_active; retained after close/reap.
_closed_session_activity: dict[str, float] = {}


def _pop_session_by_id(sid: str, *, respect_control: bool = True):
    """Atomically detach one live session from the registry — the ownership claim for teardown (a concurrent
    close/reaper no-ops). Separate from ``_teardown_session``: slow finalization must not run under the resume lock."""
    with _sessions_lock:
        session = _sessions.get(sid)
        if session is None:
            return None
        if respect_control:
            history_lock = session.get("history_lock")
            if history_lock is not None:
                with history_lock:
                    if session.get("_session_control_inflight") is not None:
                        return _SESSION_CONTROL_BUSY
                    session["_session_control_inflight"] = object()
        from hermes_constants import get_hermes_home

        home = str(Path(session.get("profile_home") or get_hermes_home()).resolve())
        last_active = time.time() if session.get("running") else float(session.get("last_active") or 0)
        _closed_session_activity[home] = max(_closed_session_activity.get(home, 0), last_active)
        session["_closing"] = True
        session = _sessions.pop(sid, None)
        if session is not None:
            session["_sid"] = sid  # out of _sessions now, so teardown can't recover the live id by scanning
    return session


def _teardown_popped_session(session: dict | None, *, end_reason: str = "tui_close") -> bool:
    """Finish a close after the caller has atomically detached the session."""
    if session is None or session is _SESSION_CONTROL_BUSY:
        return False
    run_thread = session.get("_run_thread")
    if end_reason != "tui_shutdown" and run_thread is not None and run_thread is not threading.current_thread():
        try:
            if run_thread.is_alive():
                run_thread.join(timeout=_TURN_SETTLE_BEFORE_CLOSE_SECONDS)
            if run_thread.is_alive():
                logger.warning(
                    "session turn thread still alive after %.1fs teardown grace", _TURN_SETTLE_BEFORE_CLOSE_SECONDS)
        except Exception:
            logger.debug("failed waiting for session turn thread", exc_info=True)
    _teardown_session(session, end_reason=end_reason)
    return True


def _close_session_by_id(
    sid: str, *, end_reason: str = "tui_close", predicate: Callable[[dict], bool] | None = None) -> bool:
    """Idempotent teardown funnel for callers with no resume race (resume-sensitive callers pop under
    ``_session_resume_lock`` and call ``_teardown_popped_session`` after releasing it). Automatic reapers pass
    ``predicate`` to revalidate under ``_sessions_lock`` right before the claim, so a stale scan can't close a
    session that reattached."""
    with _sessions_lock:  # RLock: predicate + claim in one critical section
        current = _sessions.get(sid)
        if predicate is not None and (current is None or not predicate(current)):
            return False
        session = _pop_session_by_id(sid)
    if session is _SESSION_CONTROL_BUSY:
        return False
    return _teardown_popped_session(session, end_reason=end_reason)


def _ws_session_is_detached(session: dict | None) -> bool:
    """True if a live session is still bound to the disconnected-WS sentinel."""
    return bool(session and not session.get("_finalized") and session.get("transport") is _detached_ws_transport)


def _ws_session_is_orphaned(session: dict | None) -> bool:
    """True if a WS session sits on ``_detached_ws_transport`` (where ``handle_ws`` parks disconnected clients), idle."""
    return bool(_ws_session_is_detached(session) and not session.get("running"))


def _interrupt_session_turn(sid: str, session: dict, *, request_id: str | None = None) -> bool:
    """Apply the shared ``session.interrupt`` contract to one claimed session; returns whether the compute-host control
    channel was used. The WS orphan reaper reuses this so a dead client gets the same partial-history/queue semantics."""
    use_compute_host = _session_uses_compute_host(session)
    should_interrupt = bool(session.get("running"))
    run_thread_alive = False
    if use_compute_host:
        # The host owns the live turn (parent `running` can lag a blocked tool), so let it decide. Gate on
        # `_compute_host_active`: HostSupervisor.interrupt() calls start(), so a lazy session would spawn a child to interrupt.
        if should_interrupt or session.get("_compute_host_active"):
            _get_compute_host_supervisor().interrupt(sid, request_id=request_id)
    else:
        run_thread_alive = (rt := session.get("_run_thread")) is not None and rt.is_alive()
    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
    if not use_compute_host:
        if should_interrupt:
            from agent.interrupt_compat import request_hard_interrupt
            request_hard_interrupt(session.get("agent"))
        if not run_thread_alive:
            with session["history_lock"]:
                if session.get("running"):
                    session["running"] = False
                    _clear_inflight_turn(session)
    _clear_pending(sid)
    with contextlib.suppress(Exception):
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    return use_compute_host


def _session_has_active_delegations(sid: str, session: dict | None = None) -> bool:
    """True when UI session ``sid`` still owns live background work — by live UI sid AND, when the TUI owns the durable
    lifecycle (never for gateway-viewer tabs), by session_key so a delegation from an earlier tab keeps it alive.

    See #60609.
    """
    if session is None:
        with _sessions_lock:
            session = _sessions.get(sid)
    if not session:
        return False
    own_sid = _lifecycle_own_sid(session, sid)
    owned_session_key = session_key = str(session.get("session_key") or "")
    session_id = getattr(session.get("agent"), "session_id", None) or session_key
    if session_id:
        # Only when this session may end its durable row by key — never for gateway-originated sessions (TUI is a
        # viewer there). Unknown DB state -> assume ownership.
        with contextlib.suppress(Exception):
            db = _get_db()
            if db is not None and _is_gateway_owned_source((db.get_session(session_id) or {}).get("source", "")):
                owned_session_key = ""
    if not own_sid and not owned_session_key:
        return False
    try:
        from tools.async_delegation import has_live_for_session
        return has_live_for_session(session_key=owned_session_key, origin_ui_session_id=own_sid)
    except Exception:
        logger.debug("Failed to query active delegations for UI session %s", sid, exc_info=True)
        return True  # a transient registry/import failure must not become destructive cleanup


# One pending WS-orphan reap Timer per live sid; guarded by _sessions_lock. Cancelled by _cancel_ws_orphan_reap from
# every resume/reuse/transport-rebind path — else a reap on a reattached session triggers a reap->broadcast->resume storm.
_pending_ws_reaps: dict[str, threading.Timer] = {}


def _cancel_ws_orphan_reap(sid: str) -> None:
    """Cancel a pending WS-orphan reap for ``sid`` (client came back). Called from every path that re-binds a live
    transport; closes the fired-but-not-run Timer race and stops dead Timers accumulating on flappy clients."""
    with _sessions_lock:
        timer = _pending_ws_reaps.pop(sid, None)
    if timer is not None:
        with contextlib.suppress(Exception):
            timer.cancel()


def _reattach_refusal(rid, sid: str, session: dict) -> dict | None:
    """Under ``_session_resume_lock``: why a reattaching RPC (resume/activate/prompt.submit) must NOT rebind
    ``session`` — it is stale, or a client-gone interrupt is still settling and the reap Timer must keep
    polling. None when the reattach may proceed."""
    if _sessions.get(sid) is not session:
        return _err(rid, 4007, "session no longer live; retry resume")
    if session.get("_client_gone_interrupt_requested"):
        return _err(rid, 4009, "session disconnect interrupt settling")
    return None


def _rebind_live_transport(sid: str, session: dict, transport: Transport) -> None:
    """Attach a live peer without displacing existing subscribers (caller holds ``history_lock``).
    Subagent control authority needs no bookkeeping here: it resolves against ``session["transport"]``
    at RPC time (``tools.delegate_tool_registry._subagent_transport_matches``)."""
    _attach_session_transport(session, transport)
    # Every transport that showed this session (pop-outs resume the same sid); on disconnect the last
    # viewer becomes the transport instead of the drop sentinel.
    session.setdefault("viewers", {})[transport] = time.time()
    # See #83716.
    if transport is not _detached_ws_transport:
        _cancel_ws_orphan_reap(sid)  # the client is back — a pending ws-orphan reap must not fire


def _ws_orphan_turn_activity_is_fresh(session: dict) -> bool:
    """Whether a detached RUNNING turn's activity clock (``_touch_activity``) is still fresh — the reaper must NOT
    interrupt healthy detached work (closed laptop). Conservative: disabled threshold, missing/opaque agent, unreadable
    summary or never-stamped clock all report NOT fresh (eligible for interrupt-at-grace) to keep the wedged-turn net.

    Reuses the agent's existing activity summary (``_touch_activity`` is stamped by API waits, stream
    tokens, and tool heartbeats — the same clock the turn-liveness watchdog samples; see
    agent/turn_liveness.py). See #100325, #98028.
    Isolated turns mirror that clock from the child under a unique dispatch token;
    their monotonic samples keep aging even if the child or its pipe stalls.
    """
    if _WS_ORPHAN_ACTIVITY_STALE_S <= 0:
        return False
    if session.get("_compute_host_turn_id"):
        with session["history_lock"]:
            stamp = session.get("_compute_host_activity_ns")
            return (session.get("running", False) and isinstance(stamp, int)
                    and 0 <= (time.perf_counter_ns() - stamp) / 1_000_000_000 < _WS_ORPHAN_ACTIVITY_STALE_S)
    if not callable(summary_fn := getattr(session.get("agent"), "get_activity_summary", None)):
        return False
    try:
        elapsed = summary_fn().get("seconds_since_activity")
        return elapsed is not None and float(elapsed) < _WS_ORPHAN_ACTIVITY_STALE_S
    except Exception:
        return False


def _schedule_ws_orphan_reap(
    sid: str, *, delay_s: float | None = None, _expected_timer: threading.Timer | None = None,
) -> None:
    """After a grace window, reap session ``sid`` iff it's still orphaned. Called from the WS-disconnect path; a
    reconnect or ``session.resume`` cancels the reap by re-binding a live transport. Disabled when grace is 0."""
    if _WS_ORPHAN_REAP_GRACE_S <= 0:
        return

    def _reap() -> None:
        # Serialize the re-check against session.resume (rebinds under _session_resume_lock). Claim teardown by popping
        # under both locks, then release the resume lock before slow finalization. Order: resume_lock -> sessions_lock.
        reschedule_delay = interrupt_session = session = None
        with _session_resume_lock, _sessions_lock:
            # Keep ownership through interrupt I/O and continuation registration. A cancelled
            # callback may already be dispatched, but cannot act on a later detachment.
            if _pending_ws_reaps.get(sid) is not timer:
                return
            current = _sessions.get(sid)
            if current is None:
                _pending_ws_reaps.pop(sid, None)
                return
            if not _ws_session_is_detached(current):
                # This Timer is abandoning the interrupt claim because another
                # writer moved the live record off the detached transport.
                # Do not leave reattach RPCs fenced with 4009, or let this
                # generation's settlement polls shorten a later detachment.
                current.pop("_client_gone_interrupt_requested", None)
                current.pop("_client_gone_interrupt_polls", None)
                _pending_ws_reaps.pop(sid, None)
                return
            if _session_has_active_delegations(sid, current):
                reschedule_delay = _WS_ORPHAN_REAP_GRACE_S
            elif not current.get("running"):
                session = _pop_session_by_id(sid)
                if session is _SESSION_CONTROL_BUSY:
                    reschedule_delay = _WS_ORPHAN_INTERRUPT_REAP_POLL_S
                    session = None
            elif not current.get("_client_gone_interrupt_requested") and _ws_orphan_turn_activity_is_fresh(current):
                # Client-absent but producing: keep running detached (the sentinel buffers emits), re-check each grace.
                logger.debug("client_gone sid=%s action=defer (turn activity fresh; stale threshold %.0fs)",
                             sid, _WS_ORPHAN_ACTIVITY_STALE_S)
                reschedule_delay = _WS_ORPHAN_REAP_GRACE_S
            else:
                # Mid-turn detached sessions must never drop the single Timer: interrupt once after grace, then poll
                # until turn-finalization settles.
                polls = current["_client_gone_interrupt_polls"] = int(current.get("_client_gone_interrupt_polls") or 0) + 1
                # See #85578.
                if polls > _WS_ORPHAN_INTERRUPT_REAP_MAX_POLLS:
                    # Never settled inside the budget — force-reap rather than park forever.
                    logger.error(
                        "client_gone sid=%s: turn did not settle after %d interrupt polls (%.0fs) — force-reaping detached session",
                        sid, polls - 1, (polls - 1) * _WS_ORPHAN_INTERRUPT_REAP_POLL_S)
                    session = _pop_session_by_id(sid)
                else:
                    if not current.get("_client_gone_interrupt_requested"):
                        current["_client_gone_interrupt_requested"] = True
                        interrupt_session = current
                    reschedule_delay = _WS_ORPHAN_INTERRUPT_REAP_POLL_S
            if reschedule_delay is None:
                _pending_ws_reaps.pop(sid, None)
        if interrupt_session is not None:
            try:
                isolated = _interrupt_session_turn(sid, interrupt_session, request_id=f"client-gone-{sid}")
                logger.info("client_gone sid=%s action=interrupt turn_isolation=%s", sid, isolated)
            except Exception:
                logger.exception("client_gone interrupt failed sid=%s", sid)
                with _sessions_lock:
                    if (_sessions.get(sid) is interrupt_session
                            and _pending_ws_reaps.get(sid) is timer):
                        interrupt_session.pop("_client_gone_interrupt_requested", None)
        if reschedule_delay is not None:
            _schedule_ws_orphan_reap(sid, delay_s=reschedule_delay, _expected_timer=timer)
            return
        if session is not None and session.get("_client_gone_interrupt_requested"):
            logger.info("client_gone sid=%s action=reap", sid)
        _teardown_popped_session(session, end_reason="ws_orphan_reap")

    with _sessions_lock:
        if _expected_timer is not None and _pending_ws_reaps.get(sid) is not _expected_timer:
            return
        timer = threading.Timer(_WS_ORPHAN_REAP_GRACE_S if delay_s is None else max(0.0, delay_s), _reap)
        timer.daemon = True
        prior = _pending_ws_reaps.pop(sid, None)
        _pending_ws_reaps[sid] = timer
    if prior is not None:
        with contextlib.suppress(Exception):
            prior.cancel()
    timer.start()


def _close_sessions_for_transport(transport, *, end_reason: str = "ws_disconnect") -> tuple[int, int]:
    """Single WS-disconnect teardown entry point: reap close_on_disconnect sessions (sidecar/dashboard) immediately;
    re-point the rest at the detached transport (later emits miss the dead socket) for the grace-windowed WS-orphan
    reaper. Returns ``(reaped, detached)`` counts.

    Multi-client fan-out: the departing transport is DETACHED from every session first. A session that still has
    another client attached keeps streaming and is neither parked nor reaped — a watcher leaving must not end the
    turn the remaining client is reading. Only the sessions left clientless take the historical
    close_on_disconnect / park-sentinel path, so a single-client disconnect behaves exactly as it always has."""
    clientless = _detach_transport_from_sessions(transport)
    reaped = detached = 0
    for sid, session in clientless:
        claimed_for_teardown = None
        should_schedule_reap = False
        # session.resume fast-path attaches under _session_resume_lock: take it so a reconnect can't attach
        # between the detach above and the claim.
        with _session_resume_lock, _sessions_lock:
            current = _sessions.get(sid)
            if current is not session:
                continue
            # Prune the departing viewer registration in every branch; it must not affect the new owner.
            (current.get("viewers") or {}).pop(transport, None)
            # Revalidate before claiming (#77129, kept under fan-out): between _detach_transport_from_sessions
            # returning this session as clientless and this claim, a concurrent session.resume can attach a NEW
            # live transport. Tearing the session down, or parking the sentinel over it, would knock an attached
            # client into detached state and arm an orphan reap against a session that has a live owner. Attach
            # and detach both serialize on _session_transport_lock, so this check is race-free against them; that
            # lock is a leaf, so taking it under _sessions_lock is safe and _session_has_live_transport does not
            # re-acquire it.
            with _session_transport_lock:
                if _session_has_live_transport(current, excluding=transport):
                    continue
            if current.get("close_on_disconnect"):
                claimed_for_teardown = _pop_session_by_id(sid)
            else:
                # Point at the drop sentinel (NOT real stdio) so _ws_session_is_orphaned recognizes it; standalone
                # `hermes --tui` keeps real _stdio. UNLESS another window (pop-out viewer) still shows the session:
                # re-bind to the most recent surviving viewer instead.
                viewers = current.get("viewers") or {}
                # See #83716.
                viewers.pop(transport, None)
                live = [vt for vt, ts in sorted(viewers.items(), key=lambda kv: kv[1]) if not _transport_is_dead(vt)]
                if live:
                    _rebind_live_transport(sid, current, live[-1])
                else:
                    current["transport"] = _detached_ws_transport
                    current.pop("_client_gone_interrupt_requested", None)
                    current.pop("_client_gone_interrupt_polls", None)
                    should_schedule_reap = True
                    # Register before releasing the detachment claim: an old disconnect
                    # must not arm its first timer over a reconnect's newer detachment.
                    with contextlib.suppress(Exception):
                        _schedule_ws_orphan_reap(sid)
        if claimed_for_teardown is not None:
            reaped += _teardown_popped_session(claimed_for_teardown, end_reason=end_reason)
        elif should_schedule_reap:
            detached += 1
    return reaped, detached


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    # Generated NamedTuple/dataclass methods close over module-private helpers;
    # keep the classes intact instead of rebinding those generated methods.
    server._MicroControlTarget = _MicroControlTarget
    server._CompressionReanchorIdentity = _CompressionReanchorIdentity
    server._MicroControlReleaseResult = _MicroControlReleaseResult
    bind_module(
        globals(),
        server,
        skip=(
            "_",
            "_MicroControlTarget",
            "_CompressionReanchorIdentity",
            "_MicroControlReleaseResult",
        ),
    )
