#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${HERMES_KNOWLEDGE_DB:-/root/.hermes/knowledge_db}"
QDRANT_PORT=6333
EMBED_PORT="${EMBED_STUB_PORT:-4000}"
START_EMBED_STUB="${START_EMBED_STUB:-0}"
mkdir -p "$BASE/logs" "$BASE/run"

port_open() {
  python3 - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

stop_pidfile() {
  local pidfile="$1"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      for _ in {1..30}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
      done
    fi
    rm -f "$pidfile"
  fi
}

start_one() {
  local name="$1" script="$2" port="$3"
  local pidfile="$BASE/run/${name}.pid"
  local logfile="$BASE/logs/${name}.log"

  if port_open "$port"; then
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "$name: healthy server already active on port $port; keeping it"
      return 0
    fi
    echo "$name: port $port is occupied by an unexpected/unhealthy process" >&2
    return 1
  fi

  stop_pidfile "$pidfile"
  nohup python3 "$script" >>"$logfile" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$pidfile"

  for _ in {1..40}; do
    if port_open "$port"; then
      echo "$name: started pid=$pid port=$port"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name: exited during startup; see $logfile" >&2
      tail -n 40 "$logfile" >&2 || true
      return 1
    fi
    sleep 0.25
  done

  echo "$name: startup timeout; see $logfile" >&2
  return 1
}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

[[ -f "$BASE/qdrant_stub.py" ]] || { echo "Missing $BASE/qdrant_stub.py" >&2; exit 1; }
start_one qdrant_stub "$BASE/qdrant_stub.py" "$QDRANT_PORT"

if [[ "$START_EMBED_STUB" == "1" ]]; then
  [[ -f "$BASE/embed_stub.py" ]] || { echo "Missing $BASE/embed_stub.py" >&2; exit 1; }
  export EMBED_STUB_VECTOR_SIZE="${EMBED_STUB_VECTOR_SIZE:-${MEMORY_WIKI_EMBED_DIMENSIONS:-4096}}"
  start_one embed_stub "$BASE/embed_stub.py" "$EMBED_PORT"
else
  stop_pidfile "$BASE/run/embed_stub.pid"
  echo "embed_stub: disabled (OpenRouter Qwen is the primary embedding provider)"
fi
