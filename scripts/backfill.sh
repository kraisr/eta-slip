#!/usr/bin/env bash
set -euo pipefail

# Make src-layout imports work when not using Poetry
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# Use Poetry if available, else fall back to system python.
if command -v poetry >/dev/null 2>&1; then
  PY_CMD="poetry run python"
else
  PY_CMD="python3"
fi

RAW_PATH="$1"

STOP_IDS_FILE="config/stop_ids_manhattan_6_southbound.txt"
MIN_LEAD_SEC=480

find "$RAW_PATH" -type d -path '*/dt=*/hour=*' \
  | sed -n 's|.*dt=\([^/]*\)/hour=\([^/]*\).*|\1 \2|p' \
  | sort -u \
  | while read -r dt hour; do
      echo "=== dt=$dt hour=$hour ==="

      $PY_CMD scripts/parse_to_silver.py \
        --dt "$dt" --hour "$hour" --route 6

      $PY_CMD scripts/build_gold_eta_slip.py \
        --dt "$dt" --hour "$hour" --route 6 \
        --stop-ids-file "$STOP_IDS_FILE" \
        --min-lead-sec "$MIN_LEAD_SEC"

      echo
    done
