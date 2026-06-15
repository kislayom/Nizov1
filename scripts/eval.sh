#!/usr/bin/env bash
# Run the Nizo accuracy eval (gold tasks -> live /api/chat -> machine-scored).
#
#   ./scripts/eval.sh            # remote: ssh to Kimaya, run loopback there (recommended)
#   ./scripts/eval.sh local      # local: hit 127.0.0.1:7777 (needs an SSH tunnel + token)
#   ./scripts/eval.sh local /path/to/tasks.json
#
# Exit code is non-zero if any gold task fails — usable as a pre-deploy gate.
set -euo pipefail

CLASS="ai.nizo.agent.eval.EvalRunner"
H="kislay@192.168.4.200"
REMOTE_JAR="/home/kislay/nizo_v1/nizo-app/target/nizo.jar"
LOCAL_JAR="nizo-app/target/nizo.jar"

mode="${1:-remote}"
shift || true

if [ "$mode" = "remote" ]; then
  ssh "$H" "NIZO_WEB_URL=http://127.0.0.1:7777 NIZO_WEB_TOKEN=\$(cat ~/.nizo/web-token) \
            java -cp $REMOTE_JAR $CLASS $*"
else
  : "${NIZO_WEB_TOKEN:?set NIZO_WEB_TOKEN to the server web-token for local mode}"
  NIZO_WEB_URL="${NIZO_WEB_URL:-http://127.0.0.1:7777}" java -cp "$LOCAL_JAR" "$CLASS" "$@"
fi
