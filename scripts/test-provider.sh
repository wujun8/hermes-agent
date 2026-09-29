#!/usr/bin/env bash
# test-provider.sh — Minimal-token availability check for Hermes main LLM provider.
# Default provider health check uses gpt-5.6-luna with reasoning.effort=none (override with TEST_MODEL). /v1/models is off by default.
set -euo pipefail

# Prefer explicit HERMES_DIR / HERMES_HOME (Docker sets these) over HOME/.hermes —
# container HOME is often a profile sandbox that does not contain config.yaml.
HERMES_DIR="${HERMES_DIR:-${HERMES_HOME:-${HOME}/.hermes}}"
CONFIG_FILE="${HERMES_DIR}/config.yaml"
ENV_FILE="${HERMES_DIR}/.env"
# Hermes serve may re-exec without the entrypoint PATH; prefer the container venv.
if [[ -x /opt/hermes-venv/bin/python3 ]]; then
  export PATH="/opt/hermes-venv/bin:${PATH}"
fi
CURL_TIMEOUT="${CURL_TIMEOUT:-30}"
# Set LIST_MODELS=1 to call /v1/models (informational only; does not change the test model).
LIST_MODELS="${LIST_MODELS:-0}"
PROVIDER_HEALTH_MODEL="gpt-5.6-luna"
REASONING_EFFORT="none"
# TEST_MODEL may be set via env or later via .env; when unset, the provider health default is used.

_TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/hermes-test.XXXXXX")
# shellcheck disable=SC2317,SC2329
cleanup() {
  rm -rf -- "$_TMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

mk_temp() {
  local prefix="$1"
  mktemp "${_TMP_DIR}/${prefix}.XXXXXX"
}

die() {
  echo "[FAIL] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

load_env_file_line() {
  case "$1" in
    ''|'#'*) return 0 ;;
  esac
  case "$1" in
    *=*)
      set -- "${1%%=*}" "${1#*=}" "$2"
      set -- "${1#"${1%%[![:space:]]*}"}" "$2" "$3"
      set -- "${1%"${1##*[![:space:]]}"}" "$2" "$3"
      if [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        set -- "$1" "${2#"${2%%[![:space:]]*}"}" "$3"
        set -- "$1" "${2%"${2##*[![:space:]]}"}" "$3"
        case "$2" in
          \"*\") set -- "$1" "${2#\"}" "$3"; set -- "$1" "${2%\"}" "$3" ;;
          \'*\') set -- "$1" "${2#\'}" "$3"; set -- "$1" "${2%\'}" "$3" ;;
        esac
        if [[ $'\n'"$3"$'\n' == *$'\n'"$1"$'\n'* ]]; then
          return 0
        fi
        unset "$1"
        export "$1=$2"
      fi
      ;;
  esac
}

# Load KEY=VALUE pairs from ~/.hermes/.env without executing shell code.
load_env_file() {
  local file="$1"
  local line
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    load_env_file_line "$line" "$2"
    local line=
  done <"$file"
}

# Hermes selects the transport from model.api_mode; model names never change it.
prefer_responses_endpoint() {
  [[ "$API_MODE" == "codex_responses" ]]
}

# Retry the alternate endpoint only when the primary returned HTTP 400
# (endpoint-level rejection). Never retry on 401/403/429/5xx.
should_fallback_endpoint() {
  local http_code="$1"
  [[ "$http_code" == "400" ]]
}

need_cmd curl
need_cmd jq
need_cmd python3

[[ -f "$CONFIG_FILE" ]] || die "Config not found: $CONFIG_FILE"

load_env_file "$ENV_FILE" "$(compgen -A variable)"

# Parse model/provider settings and build a mode-0600 curl header file.
# Secrets stay in this file and are never passed as literal curl arguments.
_CFG_FILE=$(mk_temp "hermes-test-cfg")
_REQUEST_HEADERS=$(mk_temp "hermes-test-headers")
chmod 600 "$_CFG_FILE" "$_REQUEST_HEADERS"
python3 - "$CONFIG_FILE" "$_REQUEST_HEADERS" >"$_CFG_FILE" <<'PY' || die "Failed to parse $CONFIG_FILE"
import ipaddress
import os
import re
import sys
from urllib.parse import unquote, urlsplit, urlunsplit

try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML is required (pip install pyyaml)\n")
    sys.exit(1)

config_path, headers_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}
model = cfg.get("model") or {}
provider = str(model.get("provider") or "unknown")
base_url = str(model.get("base_url") or "")
default_model = str(model.get("default") or "")
api_mode = str(model.get("api_mode") or "codex_responses")
context_length = str(model.get("context_length") or "")
env_ref = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def expand(value, label):
    text = str(value)
    missing = sorted({name for name in env_ref.findall(text) if not os.environ.get(name)})
    if missing:
        sys.stderr.write("Missing environment variable(s) for %s: %s\n" % (label, ", ".join(missing)))
        sys.exit(2)
    return env_ref.sub(lambda match: os.environ[match.group(1)], text)

def exact_env_ref(value):
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(value).strip())
    return match.group(1) if match else None

def invalid_base_url():
    sys.stderr.write("Invalid base URL configuration\n")
    sys.exit(2)


def validate_hostname(hostname, authority):
    # Keep authority parsing strict: encoded authority delimiters and path
    # separators must never be reinterpreted as part of a host.
    if any(char in authority for char in ("\\", "/", "%")):
        invalid_base_url()

    if ":" in hostname:
        if not authority.startswith("["):
            invalid_base_url()
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            invalid_base_url()
        if address.version != 6:
            invalid_base_url()
        return

    if "." in hostname and hostname.replace(".", "").isdigit():
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            invalid_base_url()
        if address.version != 4:
            invalid_base_url()
        return

    try:
        idna_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        invalid_base_url()
    if idna_host.endswith("."):
        idna_host = idna_host[:-1]
    if not idna_host or len(idna_host) > 253:
        invalid_base_url()
    labels = idna_host.split(".")
    for label in labels:
        if (
            not label
            or len(label) > 63
            or not re.fullmatch(r"[A-Za-z0-9-]+", label)
            or label.startswith("-")
            or label.endswith("-")
        ):
            invalid_base_url()


def validate_base_url(value):
    raw = str(value or "")
    if not raw or any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7f for char in raw):
        invalid_base_url()
    if "?" in raw or "#" in raw:
        invalid_base_url()
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        if parsed.scheme.lower() not in {"http", "https"} or not hostname:
            invalid_base_url()
        authority = parsed.netloc
        if "@" in authority or "@" in unquote(authority):
            invalid_base_url()
        if authority.startswith("["):
            closing = authority.find("]")
            suffix = authority[closing + 1:]
            if suffix == ":" or (suffix and (not suffix.startswith(":") or not suffix[1:].isdigit())):
                invalid_base_url()
        elif ":" in authority:
            if authority.count(":") != 1 or not authority.rsplit(":", 1)[1].isdigit():
                invalid_base_url()
        validate_hostname(hostname, authority)
        parsed.port
    except (TypeError, ValueError):
        invalid_base_url()
    return raw

def normalize_route(value):
    raw = str(value or "")
    if not raw or any(ord(char) <= 0x20 for char in raw):
        return raw
    had_query_delimiter = "?" in raw.split("#", 1)[0]
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return raw
        scheme = parsed.scheme.lower()
        if "%" in hostname:
            address, zone = hostname.split("%", 1)
            host = f"{address.lower()}%{zone}"
        else:
            host = hostname.lower()
        port = parsed.port
    except (TypeError, ValueError):
        return raw
    if parsed.netloc.rsplit("@", 1)[-1].startswith("[") or ":" in host:
        host = f"[{host}]"
    if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
        host = f"{host}:{port}"
    if "@" in parsed.netloc:
        host = f"{parsed.netloc.rsplit('@', 1)[0]}@{host}"
    path = parsed.path[:-1] if parsed.path.endswith("/") and not had_query_delimiter else parsed.path
    normalized = urlunsplit((scheme, host, path, parsed.query, ""))
    if had_query_delimiter and not parsed.query:
        normalized += "?"
    return normalized

def entries(raw):
    if isinstance(raw, dict):
        return [value for value in raw.values() if isinstance(value, dict)]
    if isinstance(raw, list):
        return [value for value in raw if isinstance(value, dict)]
    return []

def named_entries(raw):
    if isinstance(raw, dict):
        return [(str(name), value) for name, value in raw.items() if isinstance(value, dict)]
    if isinstance(raw, list):
        return [(str(value.get("name") or ""), value) for value in raw if isinstance(value, dict)]
    return []

def missing_key_env(name):
    sys.stderr.write("API key environment variable is missing or empty: %s\n" % name)
    sys.exit(2)

def resolve_api_key(value, label):
    text = "" if value is None else str(value).strip()
    if not text:
        sys.stderr.write("Configured API key is empty: %s\n" % label)
        sys.exit(2)
    ref = exact_env_ref(text)
    if ref:
        resolved = os.environ.get(ref, "")
        if not resolved:
            missing_key_env(ref)
        return resolved
    return str(value)

def provider_route(entry):
    fields = ("base_url", "url", "api")
    if "base_url" not in entry:
        fields = ("baseUrl", "url", "api")
    for field in fields:
        value = entry.get(field)
        if value:
            return value
    return None

if base_url:
    validate_base_url(base_url)
    base_url = base_url.rstrip("/")

def matching_provider_entry(*, require_credentials=True, require_route=False):
    provider_names = {provider}
    if ":" in provider:
        provider_names.add(provider.split(":", 1)[1])
    normalized_base = normalize_route(base_url)
    for raw in (cfg.get("custom_providers"), cfg.get("providers")):
        for configured_name, entry in named_entries(raw):
            entry_names = {
                configured_name,
                str(entry.get("name") or ""),
                str(entry.get("provider") or ""),
            }
            identity_match = bool(provider_names.intersection(entry_names - {""}))
            entry_route = provider_route(entry)
            route_match = bool(base_url and entry_route and normalize_route(entry_route) == normalized_base)
            if base_url and entry_route and not route_match:
                continue
            if require_route and not entry_route:
                continue
            if (identity_match or route_match) and (not require_credentials or "key_env" in entry or "api_key" in entry):
                return entry
    return None

provider_route_entry = matching_provider_entry(require_credentials=False, require_route=True)
if not base_url and provider_route_entry is not None:
    base_url = validate_base_url(provider_route(provider_route_entry))
    base_url = base_url.rstrip("/")

# Explicit current-model/provider configuration always wins. Compatibility
# environment variables are consulted only when neither declares a key.
api_key = ""
if "api_key" in model:
    api_key = resolve_api_key(model.get("api_key"), "model.api_key")
else:
    key_entry = matching_provider_entry()
    if key_entry is not None:
        key_env = "" if key_entry.get("key_env") is None else str(key_entry["key_env"]).strip()
        if key_env:
            ref = exact_env_ref(key_env) or key_env
            api_key = os.environ.get(ref, "")
            if not api_key:
                missing_key_env(ref)
        elif "api_key" in key_entry:
            api_key = resolve_api_key(key_entry.get("api_key"), "provider.api_key")
        else:
            sys.stderr.write("Configured API key environment variable name is empty: provider.key_env\n")
            sys.exit(2)
if not api_key:
    for name in ("CODEX_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GLM_API_KEY", "KIMI_API_KEY", "BAILIAN_API_KEY", "AGNES_API_KEY"):
        if os.environ.get(name):
            api_key = os.environ[name]
            break
if not api_key:
    sys.stderr.write("API key not found in configured sources\n")
    sys.exit(2)

compatible_entries = entries(cfg.get("custom_providers")) + entries(cfg.get("providers"))

# Case-insensitive merge: auth first, model.default_headers next, matching
# provider extra_headers last (the most specific level wins, as in Hermes).
headers = {}
def put(name, value):
    name, value = str(name), str(value)
    if "\r" in name or "\n" in name or ":" in name or "\r" in value or "\n" in value:
        sys.stderr.write("Invalid newline or name in configured header: %s\n" % name)
        sys.exit(2)
    old = next((key for key in headers if key.lower() == name.lower()), None)
    if old is not None:
        del headers[old]
    headers[name] = value

put("Authorization", "Bearer " + api_key)
default_headers = model.get("default_headers") or {}
if isinstance(default_headers, dict):
    for name, value in default_headers.items():
        if value is not None:
            put(name, expand(value, "header " + str(name)))
for entry in compatible_entries:
    entry_route = provider_route(entry)
    if not entry_route or normalize_route(entry_route) != normalize_route(base_url):
        continue
    extra = entry.get("extra_headers") or {}
    if isinstance(extra, dict):
        for name, value in extra.items():
            if value is not None:
                put(name, expand(value, "header " + str(name)))
    break
with open(headers_path, "w", encoding="utf-8", newline="\n") as fh:
    for name, value in headers.items():
        fh.write(f"{name}: {value}\n")
os.chmod(headers_path, 0o600)
for value in (provider, base_url, default_model, api_mode, context_length):
    print(value)
PY

PROVIDER=$(sed -n '1p' "$_CFG_FILE")
BASE_URL=$(sed -n '2p' "$_CFG_FILE")
DEFAULT_MODEL=$(sed -n '3p' "$_CFG_FILE")
API_MODE=$(sed -n '4p' "$_CFG_FILE")
CONTEXT_LENGTH=$(sed -n '5p' "$_CFG_FILE")

# Select the explicit override, or the fixed provider health default.
if [[ -n "${TEST_MODEL:-}" ]]; then
  MODEL_SOURCE="TEST_MODEL override"
else
  TEST_MODEL="$PROVIDER_HEALTH_MODEL"
  MODEL_SOURCE="provider health default"
fi

[[ -n "$BASE_URL" ]] || die "No base URL configured in model.base_url or a matching provider entry for ${PROVIDER} in $CONFIG_FILE"

echo "=== Provider Availability Test ==="
echo "Provider: ${PROVIDER} (${BASE_URL})"
echo "Configured model: ${DEFAULT_MODEL:-"(unset)"}"
echo "API mode: ${API_MODE}"
echo "API key: configured"
[[ -n "$CONTEXT_LENGTH" ]] && echo "Context length: ${CONTEXT_LENGTH}"
echo

is_json_file() {
  jq empty "$1" >/dev/null 2>&1
}

response_content_type_is_sse() {
  python3 - "$1" <<'PY'
import re
import sys

try:
    raw = open(sys.argv[1], "rb").read()
except OSError:
    raise SystemExit(1)
text = raw.decode("iso-8859-1")
headers = {}
saw_status = False
for line in re.split(r"\r\n|\n|\r", text):
    if line.lower().startswith("http/"):
        headers = {}
        saw_status = True
        continue
    if not saw_status or not line or ":" not in line:
        continue
    name, value = line.split(":", 1)
    if name.strip().lower() == "content-type":
        headers["content-type"] = value.strip()
media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
raise SystemExit(0 if media_type == "text/event-stream" else 1)
PY
}

json_error_detail() {
  python3 - "$1" <<'PY'
import json
import sys
import unicodedata

body_path = sys.argv[1]
try:
    with open(body_path, encoding="utf-8") as fh:
        payload = json.load(fh)
except (OSError, UnicodeError, ValueError):
    print("empty JSON error response")
    raise SystemExit(0)


def scalar_detail(value):
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def normalize_detail(value):
    text = scalar_detail(value)
    if text is None:
        return None
    text = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in text
    )
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > 300:
        return text[:299] + "…"
    return text


detail = None
if isinstance(payload, dict):
    error = payload.get("error")
    if isinstance(error, dict):
        detail = normalize_detail(error.get("message"))
        if detail is None:
            detail = normalize_detail(payload.get("message"))
        if detail is None:
            detail = normalize_detail(error.get("code"))
        if detail is None:
            detail = normalize_detail(error.get("type"))
    else:
        detail = normalize_detail(payload.get("message"))
if detail is None:
    detail = "unrecognized JSON error response"
print(detail)
PY
}

diagnose_non_json() {
  local prefix="$1" http_code="$2" headers_file="$3" body_file="$4"
  python3 - "$prefix" "$http_code" "$headers_file" "$body_file" <<'PY'
import re
import sys
prefix, status, headers_path, body_path = sys.argv[1:]
allowed = {
    "content-type": "Content-Type", "server": "Server", "cf-ray": "CF-Ray",
    "cf-mitigated": "CF-Mitigated", "location": "Location", "retry-after": "Retry-After",
}
headers = {}
try:
    for raw in open(headers_path, encoding="iso-8859-1", errors="replace"):
        line = raw.rstrip("\r\n")
        if line.lower().startswith("http/"):
            headers = {}
            continue
        if ":" in line:
            name, value = line.split(":", 1)
            key = name.strip().lower()
            if key in allowed:
                headers[key] = value.strip()
except OSError:
    pass
try:
    body = open(body_path, encoding="utf-8", errors="replace").read(4096)
except OSError:
    body = ""
ctype = headers.get("content-type", "").lower()
looks_html = "text/html" in ctype or bool(re.search(r"<(?:!doctype\s+html|html|head|body)\b", body, re.I))
looks_cf = "cloudflare" in headers.get("server", "").lower() or "cf-ray" in headers or "cf-mitigated" in headers
if looks_html and looks_cf:
    print(f"{prefix}Cloudflare/proxy returned HTML instead of API JSON")
elif looks_cf:
    print(f"{prefix}Cloudflare/proxy returned a non-JSON response")
else:
    print(f"{prefix}Provider returned a non-JSON response")
print(f"  HTTP status: {status}")
for key, label in allowed.items():
    if key in headers:
        print(f"  {label}: {headers[key]}")
print("  Body summary: (omitted)")
PY
}

SELECTED_MODEL="$TEST_MODEL"

if [[ "$LIST_MODELS" == "1" ]]; then
  MODELS_HTTP="000"
  MODELS_ERR=""
  _MODELS_FILE=$(mk_temp "hermes-test-models")
  _MODELS_HEADERS=$(mk_temp "hermes-test-model-headers")
  _MODELS_CURL_ERR=$(mk_temp "hermes-test-model-curl")
  if ! MODELS_HTTP=$(curl -sS --max-time "$CURL_TIMEOUT" -o "$_MODELS_FILE" -D "$_MODELS_HEADERS" -w '%{http_code}' \
    -H @"$_REQUEST_HEADERS" "${BASE_URL}/models" 2>"$_MODELS_CURL_ERR"); then
    MODELS_ERR=$(cat "$_MODELS_CURL_ERR" 2>/dev/null || true)
    MODELS_HTTP="000"
  fi

  if [[ -n "$MODELS_ERR" ]]; then
    echo "Models endpoint: unavailable (HTTP ${MODELS_HTTP}) [LIST_MODELS=1]"
    echo "  curl error: ${MODELS_ERR}"
  elif ! is_json_file "$_MODELS_FILE"; then
    echo "Models endpoint: unavailable (HTTP ${MODELS_HTTP}) [LIST_MODELS=1]"
    diagnose_non_json "  [WARN] Models endpoint: " "$MODELS_HTTP" "$_MODELS_HEADERS" "$_MODELS_FILE"
  elif [[ "$MODELS_HTTP" == "200" ]] && jq -e '.data | type == "array"' "$_MODELS_FILE" >/dev/null 2>&1; then
    MODEL_COUNT=$(jq '.data | length' "$_MODELS_FILE")
    echo "Models endpoint: OK (${MODEL_COUNT} models) [LIST_MODELS=1]"
    jq -r '.data[].id' "$_MODELS_FILE" | sed 's/^/  - /'
  else
    echo "Models endpoint: unavailable (HTTP ${MODELS_HTTP}) [LIST_MODELS=1]"
    printf '  JSON error: %s\n' "$(json_error_detail "$_MODELS_FILE")"
  fi
  echo
else
  echo "Models endpoint: skipped (default; set LIST_MODELS=1 to list)"
fi

[[ -n "$SELECTED_MODEL" ]] || die "No model available to test"

if prefer_responses_endpoint; then
  PRIMARY_ENDPOINT="responses"
else
  PRIMARY_ENDPOINT="chat_completions"
fi
endpoint_suffix() {
  printf '/%s' "${1//_//}"
}

PRIMARY_ENDPOINT_SUFFIX=$(endpoint_suffix "$PRIMARY_ENDPOINT")

echo
echo "Test model: ${SELECTED_MODEL} (${MODEL_SOURCE})"
echo "Primary endpoint: ${PRIMARY_ENDPOINT_SUFFIX}"
echo "Reasoning effort: ${REASONING_EFFORT}"
# Token policy differs per endpoint: /responses omits max_output_tokens
# (provider default limit), /chat/completions uses max_tokens: 1.
if [[ "$PRIMARY_ENDPOINT" == "responses" ]]; then
  echo 'Prompt: Reply OK. | instructions: Reply with OK only. | max output tokens: provider default | stream: true'
else
  echo 'Prompt: Reply OK. | max_tokens: 1 | stream: true'
fi
echo

api_request() {
  local endpoint="$1"
  local payload="$2"
  local curl_err_file http_code

  RESPONSE_FILE=$(mk_temp "hermes-test-resp")
  RESPONSE_HEADERS_FILE=$(mk_temp "hermes-test-response-headers")
  curl_err_file=$(mk_temp "hermes-test-curl")

  if http_code=$(curl -sS --max-time "$CURL_TIMEOUT" \
    -o "$RESPONSE_FILE" \
    -D "$RESPONSE_HEADERS_FILE" \
    -w '%{http_code}' \
    -H @"$_REQUEST_HEADERS" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "${BASE_URL}${endpoint}" 2>"$curl_err_file"); then
    :
  else
    http_code="000"
  fi

  RESPONSE_CURL_ERR=$(cat "$curl_err_file" 2>/dev/null || true)
  ENDPOINT_USED="$endpoint"
  RESPONSE_HTTP="$http_code"
}

require_response_body() {
  if [[ -n "$RESPONSE_CURL_ERR" ]]; then
    echo "Endpoint: ${ENDPOINT_USED} (HTTP ${RESPONSE_HTTP})"
    echo "[FAIL] curl error: ${RESPONSE_CURL_ERR}"
    exit 1
  fi
  if is_json_file "$RESPONSE_FILE"; then
    return 0
  fi
  if [[ "$RESPONSE_HTTP" == "200" ]] && response_content_type_is_sse "$RESPONSE_HEADERS_FILE"; then
    return 0
  fi
  echo "Endpoint: ${ENDPOINT_USED} (HTTP ${RESPONSE_HTTP})"
  diagnose_non_json "[FAIL] " "$RESPONSE_HTTP" "$RESPONSE_HEADERS_FILE" "$RESPONSE_FILE"
  exit 1
}

RESPONSE_HTTP=""
ENDPOINT_USED=""
RESPONSE_CURL_ERR=""
RESPONSE_FILE=""
RESPONSE_HEADERS_FILE=""

validate_sse_response() {
  local endpoint="$1" body_file="$2" result
  if result=$(python3 - "$endpoint" "$body_file" <<'PY'
import json
import re
import sys

endpoint, body_path = sys.argv[1:]


def fail(message):
    print(message[:300])
    raise SystemExit(1)


def usage_tokens(usage):
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool):
        return str(total)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if all(isinstance(value, int) and not isinstance(value, bool)
           for value in (input_tokens, output_tokens)):
        return str(input_tokens + output_tokens)
    return None


def object_usage(obj):
    if not isinstance(obj, dict):
        return None
    tokens = usage_tokens(obj.get("usage"))
    if tokens is not None:
        return tokens
    response = obj.get("response")
    if isinstance(response, dict):
        return usage_tokens(response.get("usage"))
    return None


def decode_data(data):
    if data == "[DONE]":
        return data
    if not data:
        fail("SSE data event is empty")
    try:
        record = json.loads(data)
    except (TypeError, ValueError):
        fail("SSE data is malformed JSON")
    if not isinstance(record, dict):
        fail("SSE data event is not an object")
    return record


try:
    with open(body_path, "rb") as fh:
        raw_body = fh.read()
except OSError:
    fail("SSE body could not be read")
try:
    body = raw_body.decode("utf-8")
except UnicodeDecodeError:
    fail("SSE body is not valid UTF-8")

records = []
data_lines = []


def dispatch():
    if data_lines:
        records.append(decode_data("\n".join(data_lines)))
        data_lines.clear()


def is_complete_data_event(data):
    if data == "[DONE]":
        return True
    if not data:
        return False
    try:
        record = json.loads(data)
    except (TypeError, ValueError):
        return False
    return isinstance(record, dict)


for raw_line in body.splitlines(keepends=True):
    if raw_line.endswith("\r\n"):
        line = raw_line[:-2]
    elif raw_line.endswith(("\n", "\r")):
        line = raw_line[:-1]
    else:
        line = raw_line
    if line == "":
        dispatch()
        continue
    if line.startswith(":"):
        continue
    if ":" in line:
        field, value = line.split(":", 1)
        if value.startswith(" "):
            value = value[1:]
    else:
        field, value = line, ""
    if field not in {"data", "event", "id", "retry"}:
        fail("SSE contains an unknown field")
    if field in {"event", "id"} and "\x00" in value:
        fail("SSE contains an invalid field value")
    if field == "retry" and value and not re.fullmatch(r"[0-9]+", value):
        fail("SSE retry field is invalid")
    if field == "data":
        # Providers commonly emit one JSON object per data line without the
        # blank dispatch line required by the SSE specification. Preserve that
        # compatibility while still assembling genuinely multi-line events.
        if data_lines and is_complete_data_event("\n".join(data_lines)):
            dispatch()
        data_lines.append(value)

dispatch()

if endpoint == "/responses":
    # Some compatible relays append [DONE] after response.completed.
    # Keep the completed/status/error checks below authoritative.
    if records and records[-1] == "[DONE]":
        records = records[:-1]
    for record in records:
        if record == "[DONE]":
            fail("Responses SSE contains an unexpected [DONE]")
        if not isinstance(record, dict):
            fail("Responses SSE event is invalid")
        event_type = record.get("type")
        if event_type == "error":
            fail("Responses SSE error event")
        if record.get("error"):
            fail("Responses SSE error envelope")
        if event_type in {"response.failed", "response.incomplete"}:
            fail(f"Responses SSE terminal {event_type}")
    if not records or not isinstance(records[-1], dict):
        fail("Responses SSE ended without response.completed")
    terminal = records[-1]
    if terminal.get("type") != "response.completed":
        fail("Responses SSE ended without response.completed")
    response = terminal.get("response")
    if not isinstance(response, dict):
        fail("Responses SSE completed event has no response object")
    if response.get("status") != "completed":
        fail("Responses SSE completed event has invalid status")
    tokens = object_usage(terminal)
    if tokens is None:
        for record in reversed(records):
            if isinstance(record, dict):
                tokens = object_usage(record)
                if tokens is not None:
                    break
    print(tokens if tokens is not None else "unknown")
else:
    if not records or records[-1] != "[DONE]":
        fail("Chat SSE ended without [DONE]")
    if any(record == "[DONE]" for record in records[:-1]):
        fail("Chat SSE received [DONE] before the final event")
    for record in records[:-1]:
        if not isinstance(record, dict):
            fail("Chat SSE event is invalid")
        event_type = record.get("type")
        if event_type == "error":
            fail("Chat SSE error event")
        if record.get("error"):
            fail("Chat SSE error envelope")
        if event_type in {"response.failed", "response.incomplete"}:
            fail(f"Chat SSE terminal {event_type}")
    if not any(isinstance(record, dict)
               and isinstance(record.get("choices"), list)
               and len(record["choices"]) > 0
               for record in records[:-1]):
        fail("Chat SSE had no choices-bearing chunk")
    tokens = None
    for record in reversed(records[:-1]):
        tokens = object_usage(record)
        if tokens is not None:
            break
    print(tokens if tokens is not None else "unknown")
PY
  ); then
    TOKENS="$result"
    return 0
  fi
  echo "[FAIL] ${result:-Invalid SSE response}"
  exit 1
}

validate_json_response() {
  local endpoint="$1" body_file="$2" result
  if result=$(python3 - "$endpoint" "$body_file" <<'PY'
import json
import sys

endpoint, body_path = sys.argv[1:]


def fail(message):
    print(message[:300])
    raise SystemExit(1)


def usage_tokens(usage):
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool):
        return str(total)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if all(isinstance(value, int) and not isinstance(value, bool)
           for value in (input_tokens, output_tokens)):
        return str(input_tokens + output_tokens)
    return None


try:
    with open(body_path, encoding="utf-8") as fh:
        payload = json.load(fh)
except (OSError, ValueError):
    fail("body is not valid JSON")

if not isinstance(payload, dict):
    fail("top-level JSON value is not an object")
if payload.get("error") is not None:
    fail("error envelope")

if endpoint == "/responses":
    if payload.get("status") != "completed":
        fail("status is not completed")
    if not isinstance(payload.get("output"), list):
        fail("output is not an array")
    print(usage_tokens(payload.get("usage")) or "unknown")
else:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        fail("choices is not a non-empty array")
    choice = choices[0]
    if not isinstance(choice, dict):
        fail("first choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        fail("first choice has no message object")
    if message.get("role") != "assistant":
        fail("first choice message role is not assistant")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        fail("assistant message content is empty or not a string")
    print(usage_tokens(payload.get("usage")) or "unknown")
PY
  ); then
    TOKENS="$result"
    return 0
  fi
  echo "[FAIL] Unexpected ${endpoint} JSON response: ${result:-invalid JSON response}"
  exit 1
}

try_responses() {
  local payload
  # The upstream rejects /responses requests that carry max_output_tokens:
  # even a value of 1 yields HTTP 400 upstream_error (verified with
  # gpt-5.6-luna and gpt-5.6-sol). Omit the field entirely so the provider
  # default output-token limit applies. /chat/completions keeps max_tokens: 1.
  payload=$(jq -nc --arg model "$SELECTED_MODEL" --arg effort "$REASONING_EFFORT" \
    '{model: $model, instructions: "Reply with OK only.", input: "Reply OK.", store: false, stream: true, reasoning: {effort: $effort}}')
  api_request "/responses" "$payload"
}

try_chat() {
  local payload
  payload=$(jq -nc --arg model "$SELECTED_MODEL" \
    '{model: $model, messages: [{role: "user", content: "Reply OK."}], max_tokens: 1, stream: true}')
  api_request "/chat/completions" "$payload"
}

if [[ "$PRIMARY_ENDPOINT" == "responses" ]]; then
  try_responses
  require_response_body
  if [[ "$RESPONSE_HTTP" != "200" ]] && should_fallback_endpoint "$RESPONSE_HTTP"; then
    echo "Primary ${PRIMARY_ENDPOINT_SUFFIX} returned HTTP ${RESPONSE_HTTP}, trying $(endpoint_suffix chat_completions)..."
    try_chat
    require_response_body
  fi
else
  try_chat
  require_response_body
  if [[ "$RESPONSE_HTTP" != "200" ]] && should_fallback_endpoint "$RESPONSE_HTTP"; then
    echo "Primary ${PRIMARY_ENDPOINT_SUFFIX} returned HTTP ${RESPONSE_HTTP}, trying $(endpoint_suffix responses)..."
    try_responses
    require_response_body
  fi
fi

echo "Endpoint: ${ENDPOINT_USED} (HTTP ${RESPONSE_HTTP})"

if [[ -n "$RESPONSE_CURL_ERR" ]]; then
  echo "[FAIL] curl error: ${RESPONSE_CURL_ERR}"
  exit 1
fi

if [[ "$RESPONSE_HTTP" != "200" ]]; then
  ERR_MSG=$(json_error_detail "$RESPONSE_FILE")
  printf '[FAIL] Provider returned error (HTTP %s): %s\n' "$RESPONSE_HTTP" "$ERR_MSG"
  exit 1
fi

# Validate response shape.
if response_content_type_is_sse "$RESPONSE_HEADERS_FILE"; then
  validate_sse_response "$ENDPOINT_USED" "$RESPONSE_FILE"
else
  validate_json_response "$ENDPOINT_USED" "$RESPONSE_FILE"
fi

echo "[PASS] Provider is available. Tokens used: ${TOKENS}"
exit 0
