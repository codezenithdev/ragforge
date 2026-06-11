#!/usr/bin/env bash
# Briefr backup (C4 / P2.4): a logical Postgres dump + the Chroma persisted
# volume. Run from the repo root with the stack up:
#
#   ./scripts/backup.sh                 # -> ./backups/{pg,chroma}_<ts>.*
#   BACKUP_DIR=/mnt/backups ./scripts/backup.sh
#
# Schedule via cron, e.g.:  0 3 * * *  cd /opt/briefr && ./scripts/backup.sh
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups}"
PROJECT="${COMPOSE_PROJECT_NAME:-briefr}"
PG_USER="${POSTGRES_USER:-briefr}"
PG_DB="${POSTGRES_DB:-briefr}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"
ABS_BACKUP_DIR="$(cd "$BACKUP_DIR" && pwd)"

echo "[backup] Postgres dump (custom format)…"
docker exec "${PROJECT}-postgres" pg_dump -U "$PG_USER" -d "$PG_DB" -Fc \
  > "$BACKUP_DIR/pg_${TS}.dump"

echo "[backup] Chroma data volume…"
docker run --rm \
  -v "${PROJECT}_chromadata:/data:ro" \
  -v "${ABS_BACKUP_DIR}:/backup" \
  alpine tar czf "/backup/chroma_${TS}.tar.gz" -C /data .

echo "[backup] done:"
echo "  $BACKUP_DIR/pg_${TS}.dump"
echo "  $BACKUP_DIR/chroma_${TS}.tar.gz"
