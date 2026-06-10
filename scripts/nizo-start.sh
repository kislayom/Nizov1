#!/usr/bin/env bash
# Bring up the full Nizo stack from this laptop.
#
# Wakes the server (Wake-on-LAN if asleep) → re-establishes the :7777 SSH tunnel →
# triggers the server-side `nizo-start.sh` which starts nizo-llama, nizo-voice,
# nizo-app via systemd. Idempotent: already-up services stay up.
#
# Usage: nizo-start.sh
#
# Env:
#   NIZO_HOST       — server hostname/IP (default 192.168.4.200)
#   NIZO_SSH_USER   — SSH user (default kislay)
#   NIZO_LOCAL_PORT — local tunnel port (default 7777)
set -euo pipefail

HOST="${NIZO_HOST:-192.168.4.200}"
SSH_USER="${NIZO_SSH_USER:-kislay}"
LOCAL_PORT="${NIZO_LOCAL_PORT:-7777}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "→ Waking ${HOST} (idempotent)..."
"${SCRIPT_DIR}/wake-server.sh" || { echo "ERROR: server didn't wake"; exit 1; }

echo "→ Starting services on ${HOST}..."
# bash -lc so ~/.local/bin is on PATH (non-interactive SSH sessions skip ~/.bashrc).
ssh -o ConnectTimeout=10 "${SSH_USER}@${HOST}" 'bash -lc nizo-start.sh'

echo "→ Re-establishing :${LOCAL_PORT} SSH tunnel..."
pkill -f "ssh.*-L ${LOCAL_PORT}:" 2>/dev/null || true
sleep 1
ssh -fN -L "${LOCAL_PORT}:localhost:${LOCAL_PORT}" "${SSH_USER}@${HOST}"
sleep 1

if curl -fsS --max-time 5 "http://localhost:${LOCAL_PORT}/api/status" \
        -H "X-Nizo-Token: $(ssh "${SSH_USER}@${HOST}" 'cat ~/.nizo/web-token')" >/dev/null 2>&1; then
    echo "✓ Nizo up at http://localhost:${LOCAL_PORT}/"
else
    echo "⚠ Tunnel established but /api/status didn't respond (200). Service may still be warming."
fi
