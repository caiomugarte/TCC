#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
API_DIR="$ROOT_DIR/api"
WEB_DIR="$ROOT_DIR/web"
API_VENV="$API_DIR/.venv"
API_PYTHON="$API_VENV/bin/python"
API_PORT=8000
WEB_PORT=3000

API_ENABLED=1
WEB_ENABLED=1
RUN_MIGRATIONS=0
API_PID=""
WEB_PID=""
LOG_DIR=""
API_LOG=""
WEB_LOG=""
MIGRATION_LOG=""
WEB_ENV_FILE=""
CLEANED=0
CHILD_PIDS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/run-local.sh [options]

Start the local FastAPI API and Next.js web app from any working directory.
Migrations are skipped by default; use --migrate explicitly when needed.

Options:
  --migrate       Run Alembic upgrade head before starting the API.
  --skip-migrations
                  Keep migrations disabled (the default).
  --api-only      Start only the API.
  --web-only      Start only the web app.
  -h, --help      Show this help.

The script never overwrites .env files. If one is absent, it copies the
corresponding .env.example and stops if required local configuration is
missing. API_PORT is read from api/.env and WEB_PORT from web/.env.local
(or web/.env); defaults are 8000 and 3000. Logs are written to a temporary
directory printed at startup.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || die "Required command not found: $command_name"
}

pid_alive() {
  kill -0 "$1" >/dev/null 2>&1
}

register_pid() {
  local candidate="$1"
  local existing

  [[ -n "$candidate" ]] || return 0
  if [[ "${#CHILD_PIDS[@]}" -gt 0 ]]; then
    for existing in "${CHILD_PIDS[@]}"; do
      [[ "$existing" == "$candidate" ]] && return 0
    done
  fi
  CHILD_PIDS+=("$candidate")
}

register_descendants() {
  local parent="$1"
  local child

  for child in $(pgrep -P "$parent" 2>/dev/null || true); do
    register_pid "$child"
    register_descendants "$child"
  done
}

stop_tree() {
  local pid="$1"
  local child

  pid_alive "$pid" || return 0
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    stop_tree "$child"
  done
  kill -TERM "$pid" >/dev/null 2>&1 || true
}

force_stop_tree() {
  local pid="$1"
  local child

  pid_alive "$pid" || return 0
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    force_stop_tree "$child"
  done
  kill -KILL "$pid" >/dev/null 2>&1 || true
}

cleanup() {
  local pid

  [[ "$CLEANED" -eq 0 ]] || return 0
  CLEANED=1
  trap - INT TERM EXIT

  if [[ "${#CHILD_PIDS[@]}" -gt 0 ]]; then
    for pid in "${CHILD_PIDS[@]}"; do
      stop_tree "$pid"
    done
    sleep 1
    for pid in "${CHILD_PIDS[@]}"; do
      force_stop_tree "$pid"
    done
    for pid in "${CHILD_PIDS[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi

  if [[ -n "$LOG_DIR" ]]; then
    printf 'Logs preserved in: %s\n' "$LOG_DIR"
  fi
}

handle_signal() {
  local signal_name="$1"
  cleanup
  if [[ "$signal_name" == "TERM" ]]; then
    exit 143
  fi
  exit 130
}

env_has_value() {
  local file="$1"
  local key="$2"
  local line name value

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"

    name="${line%%=*}"
    name="${name%"${name##*[![:space:]]}"}"
    [[ "$name" == "$key" ]] || continue

    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ -n "$value" && "$value" != '""' && "$value" != "''" ]] && return 0
  done < "$file"

  return 1
}

env_value() {
  local file="$1"
  local key="$2"
  local line name value

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"

    name="${line%%=*}"
    name="${name%"${name##*[![:space:]]}"}"
    [[ "$name" == "$key" ]] || continue

    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ ${#value} -ge 2 ]]; then
      if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] ||
        [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi
    printf '%s\n' "$value"
    return 0
  done < "$file"

  return 1
}

resolve_port() {
  local file="$1"
  local key="$2"
  local default="$3"
  local label="$4"
  local value original

  if [[ -f "$file" ]] && value="$(env_value "$file" "$key")"; then
    original="$value"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$label port ($key) must be an integer between 1 and 65535 in $file; got '$original'."
    while [[ "$value" == 0* && "$value" != 0 ]]; do
      value="${value#0}"
    done
    (( ${#value} <= 5 )) || die "$label port ($key) must be an integer between 1 and 65535 in $file; got '$original'."
    (( 10#$value >= 1 && 10#$value <= 65535 )) || die "$label port ($key) must be an integer between 1 and 65535 in $file; got '$original'."
    printf '%d\n' "$((10#$value))"
    return 0
  fi

  printf '%s\n' "$default"
}

ensure_env_file() {
  local env_file="$1"
  local example_file="$2"
  local label="$3"

  if [[ -d "$env_file" ]]; then
    die "$label env path is a directory: $env_file"
  fi
  if [[ ! -e "$env_file" ]]; then
    [[ -f "$example_file" ]] || die "Missing env example: $example_file"
    cp "$example_file" "$env_file"
    printf 'Created %s from %s; review it before starting.\n' "$env_file" "$example_file"
  fi
  [[ -r "$env_file" ]] || die "Cannot read $label env file: $env_file"
}

check_port_free() {
  local label="$1"
  local port="$2"

  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    die "$label port $port is already in use; stop that service or choose another workflow. No existing process was killed."
  fi
}

preflight_api() {
  require_command uv
  require_command curl
  require_command lsof
  require_command pgrep

  [[ -x "$API_PYTHON" ]] || die "API Python not found at $API_PYTHON. Create the API environment there; the root .venv is for the legacy pipeline."
  "$API_PYTHON" --version >/dev/null 2>&1 || die "API Python is not executable: $API_PYTHON"
  [[ -f "$API_DIR/requirements.txt" ]] || die "Missing API dependency file: $API_DIR/requirements.txt"
  ensure_env_file "$API_DIR/.env" "$API_DIR/.env.example" "API"

  if ! "$API_PYTHON" -c 'import alembic, fastapi, sqlalchemy, uvicorn' >/dev/null 2>&1; then
    die "API dependencies are incomplete in api/.venv. Install api/requirements.txt with uv before retrying."
  fi

  if env_has_value "$API_DIR/.env" DATABASE_URL; then
    printf 'API database: DATABASE_URL configured (value hidden).\n'
  elif [[ -f "$API_DIR/prumo-dev.db" ]]; then
    printf 'API database: DATABASE_URL unset; using the existing application SQLite default at api/prumo-dev.db.\n'
  elif [[ "$RUN_MIGRATIONS" -eq 1 ]]; then
    printf 'API database: DATABASE_URL unset; explicit migration will create/apply the application SQLite default.\n'
  else
    die "No API database configured. Set DATABASE_URL in api/.env, or rerun with --migrate to apply the documented local SQLite schema."
  fi

  if ! env_has_value "$API_DIR/.env" CLERK_SECRET_KEY || ! env_has_value "$API_DIR/.env" CLERK_JWT_KEY; then
    printf 'Warning: API Clerk keys are incomplete; /health can run, but protected API routes will not authenticate.\n' >&2
  fi

  API_PORT="$(resolve_port "$API_DIR/.env" API_PORT 8000 API)"
  check_port_free API "$API_PORT"
}

preflight_web() {
  require_command npm
  require_command curl
  require_command lsof
  require_command pgrep

  [[ -f "$WEB_DIR/package-lock.json" ]] || die "Missing frontend lockfile: $WEB_DIR/package-lock.json"
  [[ -x "$WEB_DIR/node_modules/.bin/next" ]] || die "Frontend dependencies are missing. Run 'npm ci' in web before retrying."
  if [[ -d "$WEB_DIR/.env.local" || -d "$WEB_DIR/.env" ]]; then
    die "Frontend env path is a directory; expected web/.env.local or web/.env"
  fi
  if [[ -f "$WEB_DIR/.env.local" ]]; then
    WEB_ENV_FILE="$WEB_DIR/.env.local"
  elif [[ -f "$WEB_DIR/.env" ]]; then
    WEB_ENV_FILE="$WEB_DIR/.env"
  else
    [[ -f "$WEB_DIR/.env.example" ]] || die "Missing env example: $WEB_DIR/.env.example"
    WEB_ENV_FILE="$WEB_DIR/.env.local"
    cp "$WEB_DIR/.env.example" "$WEB_ENV_FILE"
    printf 'Created %s from %s; review it before starting.\n' "$WEB_ENV_FILE" "$WEB_DIR/.env.example"
  fi
  [[ -r "$WEB_ENV_FILE" ]] || die "Cannot read frontend env file: $WEB_ENV_FILE"

  env_has_value "$WEB_ENV_FILE" NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || die "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is missing in $WEB_ENV_FILE; set it before starting the frontend."
  if ! env_has_value "$WEB_ENV_FILE" CLERK_SECRET_KEY; then
    printf 'Warning: frontend CLERK_SECRET_KEY is missing; protected web routes will not authenticate.\n' >&2
  fi

  WEB_PORT="$(resolve_port "$WEB_ENV_FILE" WEB_PORT 3000 frontend)"
  check_port_free frontend "$WEB_PORT"
}

run_migrations() {
  printf 'Running explicit Alembic upgrade; no destructive migration command is used.\n'
  if ! (
    set -a
    # shellcheck disable=SC1091
    source "$API_DIR/.env"
    set +a
    cd "$ROOT_DIR"
    PYTHONPATH=api "$API_PYTHON" -m alembic -c api/alembic.ini upgrade head
  ) >"$MIGRATION_LOG" 2>&1; then
    die "Migration failed. No API was started; inspect $MIGRATION_LOG."
  fi
}

start_api() {
  (
    cd "$ROOT_DIR"
    export PYTHONPATH=api
    exec "$API_PYTHON" -m uvicorn app.main:app --reload --env-file api/.env --port "$API_PORT"
  ) >"$API_LOG" 2>&1 &
  API_PID=$!
  register_pid "$API_PID"
  register_descendants "$API_PID"
}

start_web() {
  local configured_api_url="${NEXT_PUBLIC_API_BASE_URL:-}"
  local env_file

  if [[ -z "$configured_api_url" ]]; then
    for env_file in "$WEB_DIR/.env.local" "$WEB_DIR/.env"; do
      if [[ -f "$env_file" ]] && configured_api_url="$(env_value "$env_file" NEXT_PUBLIC_API_BASE_URL)" && [[ -n "$configured_api_url" ]]; then
        break
      fi
    done
  fi
  [[ -n "$configured_api_url" ]] || configured_api_url="http://localhost:${API_PORT}"

  (
    cd "$WEB_DIR"
    export NEXT_PUBLIC_API_BASE_URL="$configured_api_url"
    exec npm run dev -- --port "$WEB_PORT"
  ) >"$WEB_LOG" 2>&1 &
  WEB_PID=$!
  register_pid "$WEB_PID"
  register_descendants "$WEB_PID"
}

wait_for_url() {
  local label="$1"
  local url="$2"
  local pid="$3"
  local log_file="$4"
  local elapsed=0
  local exit_status=0

  while [[ "$elapsed" -lt 30 ]]; do
    if curl -fsS --max-time 2 --output /dev/null "$url" >/dev/null 2>&1; then
      printf '%s healthy: %s\n' "$label" "$url"
      return 0
    fi
    register_descendants "$pid"
    if ! pid_alive "$pid"; then
      wait "$pid" || exit_status=$?
      die "$label stopped before its health check (exit $exit_status). Inspect $log_file."
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  die "$label did not become healthy within 30 seconds. Inspect $log_file."
}

monitor_children() {
  local exit_status=0

  while :; do
    if [[ "$API_ENABLED" -eq 1 ]]; then
      register_descendants "$API_PID"
      if ! pid_alive "$API_PID"; then
        wait "$API_PID" || exit_status=$?
        die "API stopped (exit $exit_status). Inspect $API_LOG."
      fi
    fi
    if [[ "$WEB_ENABLED" -eq 1 ]]; then
      register_descendants "$WEB_PID"
      if ! pid_alive "$WEB_PID"; then
        wait "$WEB_PID" || exit_status=$?
        die "frontend stopped (exit $exit_status). Inspect $WEB_LOG."
      fi
    fi
    sleep 1
  done
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --migrate)
      RUN_MIGRATIONS=1
      ;;
    --skip-migrations)
      RUN_MIGRATIONS=0
      ;;
    --api-only)
      [[ "$WEB_ENABLED" -eq 1 && "$API_ENABLED" -eq 1 ]] || die "Choose only one of --api-only and --web-only."
      WEB_ENABLED=0
      ;;
    --web-only)
      [[ "$WEB_ENABLED" -eq 1 && "$API_ENABLED" -eq 1 ]] || die "Choose only one of --api-only and --web-only."
      API_ENABLED=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1 (use --help for usage)"
      ;;
  esac
  shift
done

[[ "$API_ENABLED" -eq 1 || "$WEB_ENABLED" -eq 1 ]] || die "At least one service must be enabled."
[[ "$RUN_MIGRATIONS" -eq 0 || "$API_ENABLED" -eq 1 ]] || die "--migrate requires the API; do not combine it with --web-only."

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap cleanup EXIT

printf 'Repository root: %s\n' "$ROOT_DIR"
[[ "$API_ENABLED" -eq 1 ]] && preflight_api
if [[ "$API_ENABLED" -eq 0 ]]; then
  API_PORT="$(resolve_port "$API_DIR/.env" API_PORT 8000 API)"
fi
[[ "$WEB_ENABLED" -eq 1 ]] && preflight_web

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tcc-local.XXXXXX")"
API_LOG="$LOG_DIR/api.log"
WEB_LOG="$LOG_DIR/web.log"
MIGRATION_LOG="$LOG_DIR/migrations.log"

if [[ "$RUN_MIGRATIONS" -eq 1 ]]; then
  run_migrations
else
  printf 'Migrations: skipped (use --migrate explicitly).\n'
fi

if [[ "$API_ENABLED" -eq 1 ]]; then
  start_api
  printf 'API log: %s\n' "$API_LOG"
fi
if [[ "$WEB_ENABLED" -eq 1 ]]; then
  start_web
  printf 'Frontend log: %s\n' "$WEB_LOG"
fi

if [[ "$API_ENABLED" -eq 1 ]]; then
  wait_for_url API "http://localhost:${API_PORT}/health" "$API_PID" "$API_LOG"
fi
if [[ "$WEB_ENABLED" -eq 1 ]]; then
  wait_for_url frontend "http://localhost:${WEB_PORT}" "$WEB_PID" "$WEB_LOG"
fi

[[ "$API_ENABLED" -eq 1 ]] && printf 'API: http://localhost:%s\n' "$API_PORT"
[[ "$WEB_ENABLED" -eq 1 ]] && printf 'Frontend: http://localhost:%s\n' "$WEB_PORT"
[[ "$RUN_MIGRATIONS" -eq 1 ]] && printf 'Migration log: %s\n' "$MIGRATION_LOG"
printf 'Press Ctrl-C to stop only the processes started by this script.\n'

monitor_children
