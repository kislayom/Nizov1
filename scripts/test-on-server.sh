#!/usr/bin/env bash
# Build + run a small battery of smoke tests on the server.
set -euo pipefail

REMOTE=${NIZO_SERVER:-kislay@192.168.4.200}
REMOTE_DIR=${NIZO_REMOTE_DIR:-/home/kislay/nizo_v1}

cd "$(dirname "$0")/.."
./scripts/build-on-server.sh

echo
echo "==== smoke 1: current_time tool ===="
ssh "$REMOTE" "cd $REMOTE_DIR && java -jar nizo-app/target/nizo.jar chat 'What time is it? Use current_time.'"

echo
echo "==== smoke 2: web_search composition ===="
ssh "$REMOTE" "cd $REMOTE_DIR && java -jar nizo-app/target/nizo.jar chat 'Headline of any major tech story today. web_search and synthesize.'"

echo
echo "smoke ok"
