#!/usr/bin/env bash
# Sync source tree from laptop → server. Code lives on laptop; build runs on server.
# Excludes build artifacts, IDE files, OS junk, and the vendored nizo-memory's bench data.
set -euo pipefail

REMOTE=${NIZO_SERVER:-kislay@192.168.4.200}
REMOTE_DIR=${NIZO_REMOTE_DIR:-/home/kislay/nizo_v1}

cd "$(dirname "$0")/.."

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"

rsync -azh --delete \
  --exclude '.git/' \
  --exclude '.idea/' \
  --exclude '.vscode/' \
  --exclude 'target/' \
  --exclude '**/.DS_Store' \
  --exclude '*.iml' \
  --exclude 'nizo-memory/bench/' \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude '*.log' \
  --stats \
  ./ "$REMOTE:$REMOTE_DIR/" 2>&1 | tail -10

echo
echo "Synced to $REMOTE:$REMOTE_DIR"
