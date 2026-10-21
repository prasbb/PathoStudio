#!/usr/bin/env bash
# Launch Virchow's Suite using local Python and model weights only.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Error: virtual environment not found at $PROJECT_DIR/.venv" >&2
  echo "Create it with: python3 -m venv .venv" >&2
  echo "Then install your project's requirements." >&2
  exit 1
fi

# Prefer an explicitly configured checkpoint. Otherwise, check both common
# local weight-directory layouts without renaming or moving any files.
if [[ -z "${ASHBY_CONCH_WEIGHTS:-}" ]]; then
  for candidate in \
    "$PROJECT_DIR/models/weights/conch/pytorch_model.bin" \
    "$PROJECT_DIR/model_weights/conch/pytorch_model.bin"; do
    if [[ -f "$candidate" ]]; then
      export ASHBY_CONCH_WEIGHTS="$candidate"
      break
    fi
  done
fi

if [[ -n "${ASHBY_CONCH_WEIGHTS:-}" && -f "$ASHBY_CONCH_WEIGHTS" ]]; then
  echo "CONCH checkpoint: $ASHBY_CONCH_WEIGHTS"
else
  echo "Notice: CONCH weights not found; CONCH will remain disabled." >&2
  echo "Set ASHBY_CONCH_WEIGHTS to the local checkpoint path to enable it." >&2
fi

if [[ -d "${ASHBY_FEATURE_MODELS:-$PROJECT_DIR/feature_models}" ]]; then
  echo "Feature-model options use their own local packages; ASHBY_CONCH_WEIGHTS is only for text similarity."
fi

# Do not allow inference libraries to attempt remote model access.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Avoid known VisPy/Qt issues caused by offscreen test configuration on macOS.
unset QT_QPA_PLATFORM PYOPENGL_PLATFORM

exec "$PYTHON" -m ashby_studio "$@"
