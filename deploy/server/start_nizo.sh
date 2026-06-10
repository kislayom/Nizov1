#!/usr/bin/env bash
# Start the nizo-app UI in the background. Stdin/stdout fully detached so ssh exits cleanly.
# Note: NOT using `set -e` — kill on already-dead PIDs returns non-zero and that's expected.

cd /home/kislay/nizo_v1
LOG=/tmp/nizo-ui.log

# Refuse to race the systemd unit — both binding :7777 and respawning each other is a
# guaranteed footgun. If nizo-app.service is up, the operator should use systemctl.
if systemctl is-active --quiet nizo-app 2>/dev/null; then
  echo "ERROR: nizo-app.service is active — use 'sudo systemctl restart nizo-app' instead." >&2
  echo "       (this script bypasses systemd; running it would race the managed PID.)" >&2
  exit 1
fi

# Source ~/.nizo/.env so secrets (SMARTPROXY_*, BRAVE_API_KEY, etc.) are set before
# the JVM starts. Keep this file mode 600.
if [[ -f "$HOME/.nizo/.env" ]]; then
  set -a; source "$HOME/.nizo/.env"; set +a
fi

# Sync default skills from the repo into ~/.nizo/skills/. We use --update so newer
# repo versions win, but local-only files (user-authored skills) are preserved.
# Previous behavior was --ignore-existing which prevented bug-fix updates from landing.
if [[ -d deploy/server/skills ]]; then
  mkdir -p "$HOME/.nizo/skills"
  rsync -a --update deploy/server/skills/ "$HOME/.nizo/skills/" 2>/dev/null || true
fi

# Kill any prior instance bound to :7777 (lookup by listening socket — no argv pattern self-match).
PID=$(ss -tlpn 2>/dev/null | awk '/:7777 / {match($0, /pid=([0-9]+)/, a); print a[1]; exit}')
if [[ -n "${PID:-}" ]]; then
  echo "stopping prior nizo-app PID=$PID"
  kill "$PID" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8; do sleep 1; kill -0 "$PID" 2>/dev/null || break; done
  kill -9 "$PID" 2>/dev/null || true
  sleep 1
fi

: > "$LOG"
# Bind to all interfaces so the iOS/iPad app, browser on phones, etc. on the LAN
# (or via WireGuard tunnel) can reach :7777. Override with NIZO_WEB_HOST in
# ~/.nizo/.env if you want to lock back to 127.0.0.1.
export NIZO_WEB_HOST="${NIZO_WEB_HOST:-0.0.0.0}"
nohup java -jar nizo-app/target/nizo.jar ui </dev/null >>"$LOG" 2>&1 &
NEW_PID=$!
disown
echo "nizo-app launched, PID=$NEW_PID, log=$LOG"
exit 0
