#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python scripts/run_dynamic_neighborhood_stat.py --mode label --data-name-list SWAT.csv "$@"
