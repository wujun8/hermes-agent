# Hermes local release patches

These artifacts are maintained on the ``hermes-release`` branch.

``hermes upgrade`` captures user changes before any cleanup, generates the
maintenance payload from committed Git history (not this file), replays it in
a temporary candidate worktree with ``git apply --3way --index``, validates the
candidate, and only then promotes it.  The generated patch is a portable,
byte-safe snapshot for inspection and recovery; it is never the sole source of
truth for an upgrade.

The payload excludes this directory so the series cannot patch itself.  The
JSON ``.release_base`` file records the human release tag, immutable base SHA,
and patch hash/size.  A missing or stale patch artifact therefore cannot make a
committed local change disappear.

## Regenerate after editing local customizations

Run ``hermes upgrade``.  It atomically refreshes this directory after the
candidate has been committed and validated.  Manual edits to these artifacts
are stashed and restored like every other user edit.

## Keep out of this series

- Mem0 sync_turns / infer_turns / api_url → ``$HERMES_HOME/plugins/mem0-local``
  with ``memory.provider: mem0-local`` in config.yaml
