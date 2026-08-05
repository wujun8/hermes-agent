"""Pure micro-compaction policy and durable SessionDB tests."""

from __future__ import annotations

import json

import pytest

from hermes_cli.micro_compaction import (
    MICRO_COMPACT_OVERRIDE_KEY,
    MicroCompactionUsageError,
    effective_micro_compact,
    format_micro_compact_status,
    parse_global_micro_compact,
    parse_micro_compact_command,
)
from hermes_state import SessionDB


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("on", "on"),
        (" OFF ", "off"),
        ("InHeRiT", "inherit"),
        (" status ", "status"),
    ],
)
def test_parse_micro_compact_command_is_strict_and_normalized(raw, expected):
    assert parse_micro_compact_command(raw) == expected


@pytest.mark.parametrize("raw", ["", "on off", "on status", "maybe", "on\nstatus", None])
def test_parse_micro_compact_command_rejects_bad_usage_without_mutation(raw):
    with pytest.raises(MicroCompactionUsageError):
        parse_micro_compact_command(raw)


@pytest.mark.parametrize(
    ("global_value", "override", "expected"),
    [
        (False, None, (False, "global")),
        (True, None, (True, "global")),
        (False, True, (True, "session")),
        (True, False, (False, "session")),
        ("yes", None, (True, "global")),
        ("off", None, (False, "global")),
        ("unexpected", None, (False, "global")),
    ],
)
def test_effective_micro_compact_has_explicit_session_precedence(
    global_value, override, expected
):
    assert effective_micro_compact(global_value, override) == expected


def test_format_micro_compact_status_reports_effective_source_and_global_value():
    status = format_micro_compact_status(global_value=False, session_override=True)
    assert "ON" in status
    assert "session" in status
    assert "Global: OFF" in status

    inherited = format_micro_compact_status(global_value=True, session_override=None)
    assert "ON" in inherited
    assert "global (inherited)" in inherited
    assert "Global: ON" in inherited


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("on", True), ("YES", True), ("off", False), (None, False)],
)
def test_parse_global_micro_compact_uses_shared_truthy_semantics(value, expected):
    assert parse_global_micro_compact(value) is expected


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _model_config(db: SessionDB, session_id: str):
    row = db.get_session(session_id)
    assert row is not None
    raw = row["model_config"]
    return {} if raw in (None, "") else json.loads(raw)


def _write_raw_model_config(db: SessionDB, session_id: str, raw: str) -> None:
    def _do(conn):
        conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (raw, session_id),
        )

    db._execute_write(_do)


def test_session_override_true_false_remove_preserves_unrelated_keys(db: SessionDB):
    session_id = "micro-merge"
    db.create_session(
        session_id,
        source="cli",
        model_config={"yolo_mode": True, "_branched_from": "parent", "nested": {"x": 1}},
    )

    db.set_session_micro_compact_override(session_id, True)
    assert db.session_micro_compact_override(db.get_session(session_id)) is True
    assert _model_config(db, session_id)[MICRO_COMPACT_OVERRIDE_KEY] is True
    assert _model_config(db, session_id)["nested"] == {"x": 1}

    db.set_session_micro_compact_override(session_id, False)
    assert db.session_micro_compact_override(db.get_session(session_id)) is False
    assert _model_config(db, session_id)["yolo_mode"] is True

    db.set_session_micro_compact_override(session_id, None)
    assert db.session_micro_compact_override(db.get_session(session_id)) is None
    assert MICRO_COMPACT_OVERRIDE_KEY not in _model_config(db, session_id)
    assert _model_config(db, session_id)["_branched_from"] == "parent"


def test_session_override_rejects_malformed_or_non_object_metadata_fail_closed(db: SessionDB):
    session_id = "micro-malformed"
    db.create_session(session_id, source="cli", model_config={"keep": "me"})

    original = "{not-json"
    _write_raw_model_config(db, session_id, original)
    with pytest.raises(ValueError, match="malformed"):
        db.set_session_micro_compact_override(session_id, True)
    assert db.get_session(session_id)["model_config"] == original

    original = "[\"not\", \"an object\"]"
    _write_raw_model_config(db, session_id, original)
    with pytest.raises(ValueError, match="object"):
        db.set_session_micro_compact_override(session_id, False)
    assert db.get_session(session_id)["model_config"] == original


def test_session_override_missing_row_fails_clearly(db: SessionDB):
    with pytest.raises(ValueError, match="missing-session"):
        db.set_session_micro_compact_override("missing-session", True)


def test_session_override_sequential_field_mutations_do_not_clobber(db: SessionDB):
    session_id = "micro-sequential"
    db.create_session(
        session_id,
        source="cli",
        model_config={"_delegate_from": "parent", "yolo_mode": False},
    )

    db.set_session_micro_compact_override(session_id, True)
    db.set_session_yolo(session_id, True)
    db.set_session_micro_compact_override(session_id, None)

    config = _model_config(db, session_id)
    assert MICRO_COMPACT_OVERRIDE_KEY not in config
    assert config["yolo_mode"] is True
    assert config["_delegate_from"] == "parent"


def test_session_override_getter_is_strict_about_boolean_values(db: SessionDB):
    row = {"model_config": json.dumps({MICRO_COMPACT_OVERRIDE_KEY: "true"})}
    assert db.session_micro_compact_override(row) is None
    assert db.session_micro_compact_override({"model_config": "null"}) is None
    assert db.session_micro_compact_override({"model_config": "[]"}) is None
    assert db.session_micro_compact_override({"model_config": {MICRO_COMPACT_OVERRIDE_KEY: False}}) is False


def test_compression_child_preserves_explicit_override_and_omits_inherit_key(db: SessionDB):
    for suffix, model_config, expected in [
        ("on", {MICRO_COMPACT_OVERRIDE_KEY: True, "other": 1}, True),
        ("off", {MICRO_COMPACT_OVERRIDE_KEY: False, "other": 1}, False),
        ("inherit", {"other": 1}, None),
    ]:
        parent = f"lineage-parent-{suffix}"
        child = f"lineage-child-{suffix}"
        db.create_session(parent, source="cli")
        assert db.try_acquire_compression_lock(parent, f"holder-{suffix}", ttl_seconds=60)
        db.publish_compression_child(
            parent_session_id=parent,
            child_session_id=child,
            source="cli",
            model_config=model_config,
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder=f"holder-{suffix}",
        )
        assert db.session_micro_compact_override(db.get_session(child)) is expected
        child_config = _model_config(db, child)
        assert child_config.get("other") == 1
        if expected is None:
            assert MICRO_COMPACT_OVERRIDE_KEY not in child_config
