#!/usr/bin/env bash
# Small-models study: "do smaller models fail at numerical CTR optimization?"
# Requires OPENAI_API_KEY + OPENAI_BASE_URL (OpenRouter) set in .env.
#
# Story delivered:
#   - core gradient (small -> large) on full_ctr  => "small models suck" + approach #1 (reasoning)
#   - one verbalized run on the weakest small model => approach #2 preliminary
# gpt-4o-mini full_ctr is reused as the ceiling (already in results/).
# NOTE: no `set -e` — one model failing must not abort the remaining runs.
cd "$(dirname "$0")/.."
PY=./venv/Scripts/python.exe
DOMAIN=ecommerce
METHOD=full_ctr

# Route the OpenAI SDK through OpenRouter using the key stored as 'chatbot-key'
# in .env (non-destructive: your real OPENAI_API_KEY line is left untouched).
OR_KEY=$(sed -nE 's/^chatbot-key[[:space:]]*=[[:space:]]*"?([^"]+)"?.*/\1/p' .env)
if [ -z "$OR_KEY" ]; then echo "ERROR: chatbot-key not found in .env"; exit 1; fi
export OPENAI_API_KEY="$OR_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# small -> reasoning-small -> large open (slugs verified against openrouter.ai)
MODELS=(
  "qwen/qwen3-8b"          # small, non-reasoning
  "openai/gpt-oss-20b"     # small, reasoning (approach #1)
  "moonshotai/kimi-k2.5"   # large open ceiling
  "deepseek/deepseek-chat" # large open ceiling
)

# Fixed 10-topic comparison set (identical to the existing gpt-4o-mini ceiling run)
TOPICS=("Area Rugs" "Coaxial Cables" "Cookware Sets" "DVI" "Kitchen Sinks" \
        "Lighting Cables" "Measuring Tools & Scales" "Spray Bottles" "TV Antennas" "Wall Plates")

for M in "${MODELS[@]}"; do
  echo "############ $M ($METHOD) ############"
  $PY -m src.runner --domain "$DOMAIN" --method "$METHOD" --user-level --gen-model "$M" --resume \
    --topics "${TOPICS[@]}" \
    || echo "[WARN] run failed for $M — continuing to next model"
done

# Approach #2: verbalized feedback on the weakest small model
echo "############ qwen/qwen3-8b ($METHOD, verbalized) ############"
$PY -m src.runner --domain "$DOMAIN" --method "$METHOD" --user-level \
  --gen-model "qwen/qwen3-8b" --verbalize --resume --topics "${TOPICS[@]}" \
  || echo "[WARN] verbalized run failed — continuing"

echo
echo "All runs done. Aggregating..."
$PY scripts/aggregate_ctr.py
