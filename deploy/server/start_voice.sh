#!/usr/bin/env bash
# Start the Nizo voice sidecar (FastAPI on :7780). Detaches cleanly so ssh exits.
# Mirrors the pattern in start_nizo.sh.
#
# Run on server:
#   bash /home/kislay/nizo_v1/deploy/server/start_voice.sh

VENV=/mnt/ai-models/envs/voice
SCRIPT=/home/kislay/nizo_v1/deploy/server/voice_sidecar.py
LOG=/tmp/voice-sidecar.log

# Refuse to race the systemd unit — same rationale as start_nizo.sh.
if systemctl is-active --quiet nizo-voice 2>/dev/null; then
  echo "ERROR: nizo-voice.service is active — use 'sudo systemctl restart nizo-voice' instead." >&2
  exit 1
fi

if [[ ! -f "$VENV/bin/python" ]]; then
  echo "voice venv missing at $VENV — bootstrap with: $VENV/bin/pip install ..." >&2
  exit 2
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "sidecar script missing at $SCRIPT" >&2
  exit 2
fi

# Kill any prior instance bound to :7780.
PID=$(ss -tlpn 2>/dev/null | awk '/:7780 / {match($0, /pid=([0-9]+)/, a); print a[1]; exit}')
if [[ -n "${PID:-}" ]]; then
  echo "stopping prior voice-sidecar PID=$PID"
  kill "$PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do sleep 1; kill -0 "$PID" 2>/dev/null || break; done
  kill -9 "$PID" 2>/dev/null || true
  sleep 1
fi

: > "$LOG"

# setsid + nohup + full I/O detach so SSH can exit cleanly without dragging the
# process down. Same pattern as start_nizo.sh.
setsid nohup "$VENV/bin/python" "$SCRIPT" </dev/null >>"$LOG" 2>&1 &
NEW_PID=$!
disown
echo "voice-sidecar launched, PID=$NEW_PID, log=$LOG"
exit 0
