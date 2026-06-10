#!/usr/bin/env bash
# Restart llama-server cleanly. Find PID by listening port :8080.
LOG=/mnt/ai-models/qwen3.6-27b/logs/llama-serve.log

PID=$(ss -tlpn 2>/dev/null | awk '/:8080 / {match($0, /pid=([0-9]+)/, a); print a[1]; exit}')
if [[ -n "${PID:-}" ]]; then
  echo "Stopping llama-server PID=$PID"
  kill "$PID" 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8; do
    sleep 1
    kill -0 "$PID" 2>/dev/null || break
  done
  kill -9 "$PID" 2>/dev/null
  sleep 2
fi

# Tell ollama to evict any loaded model so GPU is free
curl -s -m 3 http://localhost:11434/api/generate \
  -d '{"model":"qwen3.6:27b-q6","prompt":"","keep_alive":0}' >/dev/null 2>&1 || true
sleep 2

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
: > "$LOG"
nohup /mnt/ai-models/envs/serve_llamacpp.sh </dev/null >>"$LOG" 2>&1 &
echo "llama-server PID=$!"
exit 0
