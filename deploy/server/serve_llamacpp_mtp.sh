#!/usr/bin/env bash
# Serve Qwen3.6-27B MTP (Unsloth UD-Q8_K_XL with multi-token prediction) via llama.cpp.
# OpenAI-compatible HTTP server on :8080 — SAME port as nizo-llama, mutually
# exclusive via systemd Conflicts=. This means nizo-app (which points at :8080)
# doesn't need to know or care which model is active — `llama-switch standard|mtp`
# swaps the model underneath.
#
# MTP = speculative decoding. The MTP-instrumented checkpoint emits draft tokens from
# an internal head; the main model verifies them in batch — ~1.5-2x faster decode with
# no accuracy loss (Unsloth's claim; benchmarks: RTX 3090 38→65 tok/s).
#
# Constraints (upstream llama.cpp May 2026):
#   - --parallel 1   MTP requires a single concurrent request
#   - NO --mmproj    vision not yet compatible with --spec-type draft-mtp
#   - --ctx-size 262144 matches the standard mode for apples-to-apples comparison.
#                      Tight on a 48 GB GPU (~46-47 GB total: 35.8 GB weights + ~9.6 GB
#                      KV @ q8_0 + draft head overhead) — if it OOMs at boot, drop to
#                      131072 or 65536.
set -e

LLAMA_BIN=$HOME/llama.cpp/build/bin/llama-server
MODEL=/mnt/ai-models/qwen3.6-27b/mtp/Qwen3.6-27B-UD-Q8_K_XL.gguf
LOG=/mnt/ai-models/qwen3.6-27b/logs/llama-serve-mtp.log
mkdir -p "$(dirname "$LOG")"

exec "$LLAMA_BIN" \
  --model "$MODEL" \
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
  --spec-type draft-mtp \
  --spec-draft-n-max 6 \
  --jinja \
  --reasoning-format deepseek \
  --threads 16 \
  --threads-batch 16 \
  --alias "Qwen/Qwen3.6-27B-MTP" \
  --log-prefix \
  >>"$LOG" 2>&1
