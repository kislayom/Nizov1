#!/usr/bin/env bash
# Nightly backup of ~/.nizo (Kimaya) → Dell (192.168.5.90:~/backups/nizo/).
#
# Why this exists: ~/.nizo holds everything Nizo "remembers" — sessions.db,
# memory.db, stock report cache, skills, music jobs — all on one NVMe. This
# ships a consistent snapshot to a second machine nightly.
#
# Safety properties:
#  - SQLite DBs are snapshotted with `.backup` (WAL-safe, no service stop).
#  - Live *.db / -wal / -shm files are EXCLUDED from the tar; only the
#    consistent snapshots under db/ are archived.
#  - cache/ and logs/ are excluded (regenerable / on journald).
#  - Rotation keeps $KEEP_DAYS days on the Dell.
#
# Install: deploy/server/systemd/nizo-backup.{service,timer} (03:30 daily).
# Restore: tar -xzf nizo-YYYYMMDD-*.tar.gz; copy db/* into ~/.nizo/ and the
#          rest of .nizo/ as-is; restart nizo-app.
set -euo pipefail

SRC="$HOME/.nizo"
DEST_HOST="kislay@192.168.5.90"
DEST_DIR="backups/nizo"
KEEP_DAYS=14
STAMP=$(date +%Y%m%d-%H%M%S)
STAGE=$(mktemp -d /tmp/nizo-backup.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT

[ -d "$SRC" ] || { echo "ERROR: $SRC missing"; exit 1; }

# 1) WAL-safe snapshots of every sqlite db
mkdir -p "$STAGE/db"
for db in "$SRC"/*.db; do
  [ -f "$db" ] || continue
  sqlite3 "$db" ".backup '$STAGE/db/$(basename "$db")'"
done

# 2) Archive: .nizo (minus live DBs + regenerables) + the db snapshots
TARBALL="$STAGE/nizo-$STAMP.tar.gz"
tar -C "$HOME" -czf "$TARBALL" \
  --exclude='.nizo/*.db' \
  --exclude='.nizo/*.db-wal' \
  --exclude='.nizo/*.db-shm' \
  --exclude='.nizo/cache' \
  --exclude='.nizo/logs' \
  .nizo \
  -C "$STAGE" db

# 3) Ship to the Dell
ssh -o BatchMode=yes -o ConnectTimeout=10 "$DEST_HOST" "mkdir -p $DEST_DIR"
scp -q -o BatchMode=yes "$TARBALL" "$DEST_HOST:$DEST_DIR/"

# 4) Verify the remote copy is a readable archive before declaring success
ssh -o BatchMode=yes "$DEST_HOST" "tar -tzf $DEST_DIR/nizo-$STAMP.tar.gz > /dev/null" \
  || { echo "ERROR: remote tarball failed verification"; exit 1; }

# 5) Rotate old backups on the Dell
ssh -o BatchMode=yes "$DEST_HOST" \
  "find $DEST_DIR -name 'nizo-*.tar.gz' -mtime +$KEEP_DAYS -delete"

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "[$(date -Iseconds)] backup ok: nizo-$STAMP.tar.gz ($SIZE) → $DEST_HOST:$DEST_DIR"
