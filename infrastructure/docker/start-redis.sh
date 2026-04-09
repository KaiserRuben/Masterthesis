#!/usr/bin/env bash
set -euo pipefail

SANDISK="/Volumes/SanDisk/redis-data"
COMPOSE="$(dirname "$0")/docker-compose.yml"

if [ -d "/Volumes/SanDisk" ]; then
    mkdir -p "$SANDISK"
    echo "Using SanDisk: $SANDISK"
    REDIS_DATA="$SANDISK" docker compose -f "$COMPOSE" up -d redis-cache
else
    echo "SanDisk not mounted — falling back to Docker volume"
    docker compose -f "$COMPOSE" up -d redis-cache
fi