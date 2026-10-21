#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python scripts/run_dynamic_neighborhood_stat.py --mode score --data-name-list SWAT.csv "$@"
