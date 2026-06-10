#!/usr/bin/env bash
# Stop the full Nizo stack from this laptop. Does NOT suspend the server itself.
#
# SSH-es to the server and runs the server-side `nizo-stop.sh` which brings down
# nizo-app, nizo-voice, nizo-llama in safe order. Also tears down the local
# :7777 tunnel so the laptop isn't holding a stale port.
#
# Usage: nizo-stop.sh
#
# Env:
#   NIZO_HOST       — server hostname/IP (default 192.168.4.200)
#   NIZO_SSH_USER   — SSH user (default kislay)
#   NIZO_LOCAL_PORT — local tunnel port (default 7777)
set -euo pipefail

HOST="${NIZO_HOST:-192.168.4.200}"
SSH_USER="${NIZO_SSH_USER:-kislay}"
LOCAL_PORT="${NIZO_LOCAL_PORT:-7777}"

if ! ssh -o ConnectTimeout=3 -o BatchMode=yes "${SSH_USER}@${HOST}" 'echo OK' 2>/dev/null | grep -q OK; then
    echo "Server ${HOST} not reachable — nothing to stop."
    pkill -f "ssh.*-L ${LOCAL_PORT}:" 2>/dev/null || true
    exit 0
fi

echo "→ Stopping services on ${HOST}..."
# bash -lc so ~/.local/bin is on PATH (non-interactive SSH sessions skip ~/.bashrc).
ssh "${SSH_USER}@${HOST}" 'bash -lc nizo-stop.sh'

echo "→ Tearing down :${LOCAL_PORT} SSH tunnel..."
pkill -f "ssh.*-L ${LOCAL_PORT}:" 2>/dev/null || true

echo "✓ Nizo stopped."
