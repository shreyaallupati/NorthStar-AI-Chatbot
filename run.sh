#!/usr/bin/env bash
# Start North Star Support Bot (backend + frontend) with one command.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

RUN_SMOKE=0
[ "${1:-}" = "--smoke" ] && RUN_SMOKE=1

say() { printf '\033[36m==>\033[0m %s\n' "$1"; }
die() { printf '\033[31mError:\033[0m %s\n' "$1" >&2; exit 1; }

# Resolve the python executable inside a venv dir (Windows vs Unix layout).
venv_python() {
  if [ -x "$1/Scripts/python.exe" ]; then echo "$1/Scripts/python.exe"
  elif [ -x "$1/bin/python" ]; then echo "$1/bin/python"
  fi
}

# A venv is only usable if pip works; the repo may contain a broken 3.14 venv.
venv_ok() {
  local py
  py="$(venv_python "$1")"
  [ -n "$py" ] && "$py" -m pip --version >/dev/null 2>&1
}

# Prefer Python 3.10-3.12; 3.13+ lacks prebuilt wheels for our pinned deps.
find_python() {
  # Version-suffixed names on PATH (typical on macOS/Linux).
  for cmd in python3.12 python3.11 python3.10; do
    command -v "$cmd" >/dev/null 2>&1 && { echo "$cmd"; return; }
  done

  # Windows py launcher, but only for versions it actually has registered.
  if command -v py >/dev/null 2>&1; then
    for v in 3.12 3.11 3.10; do
      py -"$v" --version >/dev/null 2>&1 && { echo "py -$v"; return; }
    done
  fi

  # Common Windows install dirs; an interpreter can be installed yet missing
  # from the py launcher's registry.
  for base in "$HOME/AppData/Local/Programs/Python" "/c/Program Files/Python" "/c"; do
    for v in 312 311 310; do
      cand="$base/Python$v/python.exe"
      [ -x "$cand" ] && { echo "$cand"; return; }
    done
  done

  # Last resort: a generic python that happens to be in range.
  for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
      if "$cmd" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)' 2>/dev/null; then
        echo "$cmd"; return
      fi
    fi
  done
}

# Git Bash resolves bare `npm` to a shim it cannot exec, so probe for a
# candidate that actually runs.
find_npm() {
  for cand in npm.cmd npm; do
    "$cand" --version >/dev/null 2>&1 && { echo "$cand"; return; }
  done
}

# --- backend venv ---------------------------------------------------------
VENV=""
for cand in "$BACKEND/.venv311" "$BACKEND/.venv"; do
  if [ -d "$cand" ] && venv_ok "$cand"; then VENV="$cand"; break; fi
done

if [ -z "$VENV" ]; then
  PY_CMD="$(find_python)"
  [ -n "$PY_CMD" ] || die "No Python 3.10-3.12 found. Install one, then re-run."
  VENV="$BACKEND/.venv311"
  [ -d "$VENV" ] && VENV="$BACKEND/.venv-run"
  say "Creating virtualenv at ${VENV#$ROOT/} using $PY_CMD"
  # shellcheck disable=SC2086
  $PY_CMD -m venv "$VENV" || die "Failed to create virtualenv."
  FRESH_VENV=1
fi

PY="$(venv_python "$VENV")"
[ -n "$PY" ] || die "Virtualenv at $VENV looks broken. Delete it and re-run."

[ "${FRESH_VENV:-0}" = "1" ] && "$PY" -m pip install --quiet --upgrade pip

# Always reconcile against requirements.txt so newly added packages are picked
# up in an existing virtualenv. This is a no-op once satisfied.
say "Checking backend dependencies"
"$PY" -m pip install --quiet --disable-pip-version-check -r "$BACKEND/requirements.txt" \
  || die "Dependency install failed."

if [ ! -f "$BACKEND/app/data/northstar.db" ]; then
  say "Seeding mock data"
  (cd "$BACKEND" && "$PY" seed_mock_data.py)
fi

if [ "$RUN_SMOKE" = "1" ]; then
  say "Running smoke tests"
  (cd "$BACKEND" && "$PY" smoke_test.py)
  exit 0
fi

# --- pick a free backend port --------------------------------------------
PORT="$("$PY" - <<'PYEOF'
import socket
for port in (8000, 8080, 8001, 8090):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        continue
    else:
        print(port)
        break
    finally:
        s.close()
else:
    print("")
PYEOF
)"
[ -n "$PORT" ] || die "No free port among 8000/8080/8001/8090."

# Keep the frontend pointed at whichever port we landed on.
printf 'VITE_API_URL=http://127.0.0.1:%s\n' "$PORT" > "$FRONTEND/.env"

# --- frontend deps -------------------------------------------------------
NPM="$(find_npm)"
[ -n "$NPM" ] || die "npm not found on PATH. Install Node.js 18+, then re-run."

if [ ! -d "$FRONTEND/node_modules" ]; then
  say "Installing frontend dependencies (first run, ~1-3 min)"
  (cd "$FRONTEND" && "$NPM" install)
fi

# --- start both ----------------------------------------------------------
BACK_PID=""
FRONT_PID=""
cleanup() {
  say "Shutting down"
  [ -n "$BACK_PID" ] && kill "$BACK_PID" 2>/dev/null || true
  [ -n "$FRONT_PID" ] && kill "$FRONT_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

say "Starting backend on http://127.0.0.1:$PORT"
(cd "$BACKEND" && "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT") &
BACK_PID=$!

# Wait for the API to answer before booting the UI.
for _ in $(seq 1 40); do
  if "$PY" -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    say "Backend healthy"
    break
  fi
  kill -0 "$BACK_PID" 2>/dev/null || die "Backend exited during startup."
  sleep 0.5
done

say "Starting frontend on http://127.0.0.1:5173"
(cd "$FRONTEND" && "$NPM" run dev -- --host 127.0.0.1 --port 5173) &
FRONT_PID=$!

cat <<EOF

  North Star Support Bot is running.

    Chat UI   http://127.0.0.1:5173
    API docs  http://127.0.0.1:$PORT/docs
    Health    http://127.0.0.1:$PORT/health

  Press Ctrl+C to stop both servers.

EOF

wait
