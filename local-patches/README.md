# Hermes local release patches

These artifacts are maintained on the ``hermes-release`` branch.

``hermes upgrade`` captures user changes before any cleanup, generates the
maintenance payload from committed Git history (not this file), incrementally
merges the new upstream release into the long-lived maintenance history in a
temporary candidate worktree, validates the candidate, and only then promotes
it.  Git rerere is enabled for the isolated merge so a previously recorded
resolution can be reused.  The generated patch is a portable, byte-safe
snapshot for inspection and recovery; it is never the upgrade input or sole
source of truth.

The payload excludes this directory so the series cannot patch itself.  The
JSON ``.release_base`` file records the integration mode, human release tag,
immutable upstream base SHA, and patch hash/size.  A missing or stale patch
artifact therefore cannot make a committed local change disappear.

## Regenerate after editing local customizations

Run ``hermes upgrade``.  It atomically refreshes this directory after the
candidate has been committed and validated.  Manual edits to these artifacts
are stashed and restored like every other user edit.

## Keep out of this series

- Mem0 sync_turns / infer_turns / api_url → ``$HERMES_HOME/plugins/mem0-local``
  with ``memory.provider: mem0-local`` in config.yaml
