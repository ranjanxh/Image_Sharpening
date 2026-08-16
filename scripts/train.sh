#!/usr/bin/env bash
# Train the KD student model using configs/default.yaml (or --config override).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m src.train "$@"
