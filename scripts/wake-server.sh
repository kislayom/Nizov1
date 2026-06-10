#!/usr/bin/env bash
# Wake the Nizo server (Dell, MAC bc:fc:e7:6b:57:dc) via Wake-on-LAN, then poll
# until SSH is responsive. Safe to run repeatedly — already-awake servers
# return immediately.
#
# Usage: ./scripts/wake-server.sh
#
# Requires: a working LAN route to the broadcast address (typically the same
# subnet as the server). On macOS, install wakeonlan via: brew install wakeonlan
# On Linux: apt install wakeonlan / pacman -S wakeonlan
set -euo pipefail

MAC="bc:fc:e7:6b:57:dc"
HOST="192.168.4.200"
SSH_USER="${NIZO_SSH_USER:-kislay}"
TIMEOUT_SEC="${NIZO_WAKE_TIMEOUT_SEC:-120}"

if ssh -o ConnectTimeout=3 -o BatchMode=yes "$SSH_USER@$HOST" 'echo OK' 2>/dev/null | grep -q OK; then
  echo "Server already up ($HOST)."
  exit 0
fi

# Send the magic packet. Try wakeonlan first (most reliable); fall back to a
# pure-python implementation that doesn't need a `brew install`.
if command -v wakeonlan >/dev/null 2>&1; then
  echo "Sending magic packet via wakeonlan to $MAC..."
  wakeonlan "$MAC"
elif command -v python3 >/dev/null 2>&1; then
  echo "Sending magic packet via python3 (no wakeonlan installed)..."
  python3 - <<EOF
import socket, struct
mac = bytes.fromhex("${MAC//:/}")
pkt = b"\xff" * 6 + mac * 16
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.sendto(pkt, ("255.255.255.255", 9))
s.close()
print("magic packet sent")
EOF
else
  echo "ERROR: need either 'wakeonlan' or 'python3'." >&2
  exit 2
fi

# Poll for SSH. Boot + sshd ready typically takes 30-60s on this Dell.
echo "Waiting for SSH on $HOST (up to ${TIMEOUT_SEC}s)..."
deadline=$(( $(date +%s) + TIMEOUT_SEC ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ssh -o ConnectTimeout=3 -o BatchMode=yes "$SSH_USER@$HOST" 'echo OK' 2>/dev/null | grep -q OK; then
    echo "Server reachable."
    exit 0
  fi
  sleep 3
done
echo "ERROR: server did not respond within ${TIMEOUT_SEC}s." >&2
exit 1
