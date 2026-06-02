#!/usr/bin/env bash
# Cache logits if needed, then serve the game.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f web/rounds.gpt2.json ] && [ ! -f web/rounds.qwen3.json ]; then
  echo "No web/rounds.<family>.json found -- run cache_family.py on a GPU box first"
  echo "(see README). Serving anyway; the app will error until a family file exists."
fi

PORT="${1:-8000}"
exec python serve.py "$PORT"
