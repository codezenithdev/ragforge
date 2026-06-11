# Briefr restore runbook (C4 / P2.4)

Restores the two stateful stores from artifacts produced by `scripts/backup.sh`:
a Postgres custom-format dump (`pg_<ts>.dump`) and a Chroma volume tarball
(`chroma_<ts>.tar.gz`). Postgres and Chroma must be kept consistent — restore the
**pair** taken at the same timestamp, or the reconcile sweeper (P1.4) will purge
vectors whose documents no longer exist in Postgres.

> Assumes the default compose project name `briefr` (volumes `briefr_pgdata`,
> `briefr_chromadata`). Override with `COMPOSE_PROJECT_NAME` if different.

## 1. Stop the app (keep infra reachable as needed)

```bash
docker compose stop backend worker frontend
```

## 2. Restore Postgres

```bash
# Drop & recreate the database, then load the dump.
docker exec briefr-postgres psql -U briefr -d postgres \
  -c "DROP DATABASE IF EXISTS briefr;" -c "CREATE DATABASE briefr;"
docker exec -i briefr-postgres pg_restore -U briefr -d briefr --clean --if-exists \
  < backups/pg_<ts>.dump
```

## 3. Restore Chroma

```bash
# Stop Chroma, wipe its volume, untar the snapshot back in.
docker compose stop chroma
docker run --rm -v briefr_chromadata:/data alpine sh -c "rm -rf /data/*"
docker run --rm -v briefr_chromadata:/data -v "$(pwd)/backups:/backup" \
  alpine tar xzf /backup/chroma_<ts>.tar.gz -C /data
docker compose start chroma
```

## 4. Bring the app back up

```bash
docker compose up -d            # migrate runs (no-op if schema current), then backend/worker
curl -fsS localhost:8000/ready  # all three deps should report "ok"
```

## Restore drill (verify periodically)

1. `./scripts/backup.sh`
2. Restore the pair into a **throwaway** project: `COMPOSE_PROJECT_NAME=briefr_drill`
   with a copied compose file, or a scratch DB + volume.
3. Confirm `GET /ready` is healthy and `GET /api/v1/documents` returns the
   expected rows. Tear the drill environment down.
