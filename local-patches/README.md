# Hermes local release patches

These patches are versioned on the ``hermes-release`` maintenance branch.

``hermes upgrade`` snapshots them **before** ``git reset --hard`` to the
release tag (so in-repo copies survive the reset), reapplies every ``*.patch``
(sorted by name), commits the result, then refreshes
``0001-local-maintenance.patch``.

The generated patch deliberately excludes this directory (``:!(local-patches)``)
so the series does not try to patch itself.

The API-recovery auto-continue change is carried as
``0002-api-recovery-auto-continue.patch`` until the next successful upgrade.
It covers both messaging-gateway ``resume_pending`` sessions and desktop/TUI
turn markers. The in-repo and legacy ``$HERMES_HOME/local-patches`` patch
series must remain byte-identical; the upgrader fails closed if they differ.
After applying and validating an official release, the upgrader folds the
series into a refreshed ``0001-local-maintenance.patch`` and removes stale
numbered patches.

Verify this customization after an upgrade with:

```bash
cd ~/.hermes/hermes-agent
./venv/bin/python -m pytest \
  tests/run_agent/test_api_outage_recovery.py \
  tests/gateway/test_restart_resume_pending.py \
  tests/tui_gateway/test_auto_continue.py -q
```

## Regenerate after editing local customizations

```bash
cd ~/.hermes/hermes-agent
git diff v2026.x.y HEAD -- . ':!(local-patches)' > local-patches/0001-local-maintenance.patch
git add local-patches && git commit -m "local: refresh patches"
```

## Keep out of this series

- Mem0 sync_turns / infer_turns / api_url → ``$HERMES_HOME/plugins/mem0-local``
  with ``memory.provider: mem0-local`` in config.yaml
