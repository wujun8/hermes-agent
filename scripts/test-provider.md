# Provider recovery probe

`test-provider.sh` is the versioned copy of the recovery probe deployed at
`/Users/herelink/app/test-provider.sh` on this installation. The existing
`agent.api_outage_recovery.probe_command` continues to use that deployed path.

The probe reads the selected Hermes profile configuration and credentials, sends
a small request, validates the complete response, and exits with status 0 only
when the request succeeds. `HERMES_DIR` selects the profile; otherwise it uses
`HERMES_HOME` and then the default Hermes home. It requires Bash, curl, jq, Python
3, and PyYAML. Set `TEST_MODEL` to override the existing health-check model.

Responses-compatible relays may append one `[DONE]` after `response.completed`.
The probe accepts this trailing marker while still requiring a successful
`response.completed` event. An early or repeated `[DONE]`, an error event,
`response.failed`, `response.incomplete`, or a missing completion remains a
failure.

To update this installation after changing the versioned script:

```sh
install -m 755 scripts/test-provider.sh /Users/herelink/app/test-provider.sh
```

The active recovery waiter launches the deployed script for every probe, so a
script update takes effect on its next probe without restarting Hermes. A small
health request succeeding does not guarantee that a larger conversation request
will also succeed.

Run the black-box SSE regression cases with the repository test runner:

```sh
scripts/run_tests.sh -j 1 tests/scripts/test_provider_health.py -q
```
