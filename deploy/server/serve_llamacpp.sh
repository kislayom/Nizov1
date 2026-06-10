#!/usr/bin/env bash
# Serve Qwen3.6-27B (Unsloth UD-Q6_K_XL) with vision via llama.cpp.
# OpenAI-compatible HTTP server on :8080.
set -e

LLAMA_BIN=$HOME/llama.cpp/build/bin/llama-server
MODEL=/mnt/ai-models/qwen3.6-27b/gguf-q6/Qwen3.6-27B-UD-Q6_K_XL.gguf
MMPROJ=/mnt/ai-models/qwen3.6-27b/gguf-q6/mmproj-F16.gguf
LOG=/mnt/ai-models/qwen3.6-27b/logs/llama-serve.log
mkdir -p "$(dirname "$LOG")"

exec "$LLAMA_BIN" \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 262144 \
  --n-gpu-layers 999 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --parallel 1 \
  --flash-attn auto \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning-format deepseek \
  --threads 16 \
  --threads-batch 16 \
  --alias "Qwen/Qwen3.6-27B" \
  --log-prefix \
  >>"$LOG" 2>&1
