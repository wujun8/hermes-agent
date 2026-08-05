# Hermes local release patches

These patches are versioned on the ``hermes-release`` maintenance branch.

``hermes upgrade`` snapshots them **before** ``git reset --hard`` to the
release tag (so in-repo copies survive the reset), reapplies every ``*.patch``
(sorted by name), commits the result, then refreshes
``0001-local-maintenance.patch``.

The generated patch deliberately excludes this directory (``:!(local-patches)``)
so the series does not try to patch itself.

## Regenerate after editing local customizations

```bash
cd ~/.hermes/hermes-agent
git diff v2026.x.y HEAD -- . ':!(local-patches)' > local-patches/0001-local-maintenance.patch
git add local-patches && git commit -m "local: refresh patches"
```

## Keep out of this series

- Mem0 sync_turns / infer_turns / api_url → ``$HERMES_HOME/plugins/mem0-local``
  with ``memory.provider: mem0-local`` in config.yaml
