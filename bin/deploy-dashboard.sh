#!/usr/bin/env bash
set -euo pipefail

cd /srv/eta-slip

git fetch origin main
git reset --hard origin/main

docker compose up -d --build --no-deps dashboard
docker image prune -f
