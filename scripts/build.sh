#!/usr/bin/env bash
# Build all modules. Skips tests by default (nizo-memory has live-Ollama tests
# that overheat the laptop — do not run them without explicit permission).
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=(-q -T 4 -DskipTests package)
if [[ "${1:-}" == "--quick" ]]; then
  ARGS+=(-pl 'nizo-app' -am)
fi

mvn "${ARGS[@]}"

JAR="nizo-app/target/nizo.jar"
if [[ -f "$JAR" ]]; then
  printf "\nBuilt: %s (%s)\n" "$JAR" "$(du -h "$JAR" | cut -f1)"
else
  echo "Build did not produce $JAR" >&2
  exit 1
fi
