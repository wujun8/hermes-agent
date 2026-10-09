"""Source import-integrity checks for update and stash restoration."""

import json
import subprocess
from pathlib import Path
from hermes_cli._launchers import runtime_command

# Modules imported on every startup. Unlike _UPDATE_CRITICAL_FILES (only parsed) these are
# *imported*, catching cross-module breakage (a name pulled from a sibling no longer exists).
_UPDATE_CRITICAL_MODULES = "hermes_cli.main", "run_agent", "model_tools", "toolsets"


def _critical_module_import_failures(
    root, *, report_runtime_errors: bool = False) -> dict[str, tuple[str, str]]:
    """Import each ``_UPDATE_CRITICAL_MODULES`` entry in a subprocess; return failures in probe order.

    Syntax validation only *parses*: a partially-updated tree (Windows ZIP copy loop) parses yet
    dies with ``ImportError: cannot import name``. The boot-selected subprocess
    keeps import side effects out of the updater's ``sys.modules``.
    Generic import-time exceptions are tolerated unless ``report_runtime_errors=True``.
    """
    from hermes_cli.update_cmd import _UPDATE_CRITICAL_MODULES
    from hermes_constants import FIRST_PARTY_MODULE_ROOTS
    import secrets
    try:
        checkout_root = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return _probe_failure("CheckoutRoot", f"updated checkout root is not accessible: {exc}")
    if not checkout_root.is_dir():
        return _probe_failure("CheckoutRoot", "updated checkout root is not a directory")
    marker = f"__HERMES_IMPORT_HEALTH_{secrets.token_hex(16)}__"
    probe = (
        "import importlib, json, sys\n"
        "from pathlib import Path\n"
        "_root = Path(%r).resolve(strict=True)\n"
        "sys.path.insert(0, str(_root))\n"
        # Importing hermes_cli.main runs the startup dotenv load, which pulls external secret
        # sources (op/bws/command helpers, up to 120s each) unless argv says ``update``. The
        # probe only checks importability, so it inherits the updater's own argv contract.
        "sys.argv = ['hermes', 'update']\n"
        "failures = []\n"
        "for name in %r:\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "        origin = getattr(module, '__file__', None)\n"
        "        if not origin:\n"
        "            failures.append((name, 'ModuleOriginError', 'critical module has no real __file__'))\n"
        "            continue\n"
        "        try:\n"
        "            resolved_origin = Path(origin).resolve(strict=True)\n"
        "        except (OSError, RuntimeError, TypeError, ValueError) as exc:\n"
        "            failures.append((name, 'ModuleOriginError', f'critical module origin is not a real file: {exc}'))\n"
        "            continue\n"
        "        if not resolved_origin.is_file():\n"
        "            failures.append((name, 'ModuleOriginError', f'critical module __file__ is not a regular file: {resolved_origin}'))\n"
        "            continue\n"
        "        try:\n"
        "            resolved_origin.relative_to(_root)\n"
        "        except ValueError:\n"
        "            failures.append((name, 'ModuleOriginError', f'critical module imported from outside updated checkout: {resolved_origin}'))\n"
        "    except ModuleNotFoundError as exc:\n"
        # A missing *third-party* module means deps aren't installed, not a skewed checkout;
        # only our own packages count. Roots come from hermes_constants so the user hint can't drift.
        "        missing = (getattr(exc, 'name', '') or '').split('.')[0]\n"
        "        if missing in %r or missing.startswith('hermes_') or %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except ImportError as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except Exception as exc:\n"
        "        if %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except BaseException as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "sys.stdout.write('\\n%s' + json.dumps(failures))\n"
        % (str(checkout_root), _UPDATE_CRITICAL_MODULES, tuple(sorted(FIRST_PARTY_MODULE_ROOTS)), report_runtime_errors,
           report_runtime_errors, marker))
    try:
        result = subprocess.run(
            runtime_command(checkout_root, code=probe), cwd=str(checkout_root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120)
    except subprocess.TimeoutExpired:
        return _probe_failure("TimeoutExpired", "timed out before reporting import health")
    except (OSError, subprocess.SubprocessError):
        # Can't run the probe — don't block the update on our own tooling.
        return {}
    output = result.stdout or ""
    if marker not in output:
        return _probe_failure(
            "ProbeTerminated",
            f"terminated before reporting import health (exit code {result.returncode})")
    try:
        failures = json.loads(output.rsplit(marker, 1)[1])
        if not isinstance(failures, list) or any(
            not isinstance(item, list) or len(item) != 3 or not all(isinstance(v, str) for v in item)
            for item in failures):
            raise ValueError("invalid import-health payload")
        return {str(module): (str(kind), str(detail)) for module, kind, detail in failures}
    except (TypeError, ValueError):
        return _probe_failure("MalformedPayload", "reported malformed import health data")


def _probe_failure(kind: str, detail: str) -> dict[str, tuple[str, str]]:
    """Failure row for the probe itself (as opposed to a module it imported)."""
    return {"critical-module probe": (kind, detail)}


def _validate_critical_modules_import(
    root, *, report_runtime_errors: bool = False) -> tuple[bool, str | None, str | None]:
    """Return the first critical-module import failure, if any."""
    failures = _critical_module_import_failures(root, report_runtime_errors=report_runtime_errors)
    if failures:
        module = next(iter(failures))
        return False, module, failures[module][1]
    return True, None, None

