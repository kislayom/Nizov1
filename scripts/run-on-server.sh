#!/usr/bin/env bash
# Run nizo on the server. Defaults to UI mode.
# Usage:
#   ./scripts/run-on-server.sh                     # ui
#   ./scripts/run-on-server.sh chat "hello"        # one-shot chat
#   ./scripts/run-on-server.sh telegram            # bot
set -euo pipefail

REMOTE=${NIZO_SERVER:-kislay@192.168.4.200}
REMOTE_DIR=${NIZO_REMOTE_DIR:-/home/kislay/nizo_v1}

MODE=${1:-ui}
shift || true
PROMPT="$*"

# Forward LLM env (defaults already point to localhost:8080 which is correct on the server).
ssh -t "$REMOTE" "cd $REMOTE_DIR && java -jar nizo-app/target/nizo.jar $MODE ${PROMPT:+\"$PROMPT\"}"
