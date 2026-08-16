#!/usr/bin/env bash
# Evaluate a trained student checkpoint on the test set.
# Usage: scripts/evaluate.sh --checkpoint model_checkpoints/<name>.pth [--save-images]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m src.evaluate "$@"
