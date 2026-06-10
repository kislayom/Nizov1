#!/usr/bin/env bash
# Build the Nizo uber-jar on the server (NOT the laptop).
# Usage:
#   ./scripts/build-on-server.sh             # full build (default)
#   ./scripts/build-on-server.sh --quick     # nizo-app and dependencies only
set -euo pipefail

REMOTE=${NIZO_SERVER:-kislay@192.168.4.200}
REMOTE_DIR=${NIZO_REMOTE_DIR:-/home/kislay/nizo_v1}

cd "$(dirname "$0")/.."
./scripts/sync.sh

ARGS=(-q -T 4 -DskipTests -pl 'nizo-app' -am package)
if [[ "${1:-}" == "--full" ]]; then
  ARGS=(-q -T 4 -DskipTests package)
fi

ssh "$REMOTE" "cd $REMOTE_DIR && ./mvnw ${ARGS[*]}"

ssh "$REMOTE" "ls -lh $REMOTE_DIR/nizo-app/target/nizo.jar"
