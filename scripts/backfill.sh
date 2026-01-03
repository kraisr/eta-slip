#!/usr/bin/env bash
set -euo pipefail

RAW_PATH="$1"

STOP_IDS_FILE="config/stop_ids_manhattan_6_southbound.txt"
MIN_LEAD_SEC=480

find "$RAW_PATH" -type d -path '*/dt=*/hour=*' \
  | sed -n 's|.*dt=\([^/]*\)/hour=\([^/]*\).*|\1 \2|p' \
  | sort -u \
  | while read -r dt hour; do
      echo "=== dt=$dt hour=$hour ==="

      poetry run python scripts/parse_to_silver.py \
        --dt "$dt" --hour "$hour" --route 6

      poetry run python scripts/build_gold_eta_slip.py \
        --dt "$dt" --hour "$hour" --route 6 \
        --stop-ids-file "$STOP_IDS_FILE" \
        --min-lead-sec "$MIN_LEAD_SEC"

      echo
    done
