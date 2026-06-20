#!/usr/bin/env bash
# Download the candidate replacement LLM — HauhauCS abliterated/uncensored "Aggressive" build of our exact
# current base Qwen/Qwen3.6-27B, as GGUF for llama.cpp. Q8_K_P (32 GB) matches our current UD-Q8_K_XL so the
# ONLY variable vs the current model is the uncensoring (no quant/arch/footprint change; vision via the
# included mmproj-f16). This is a CANDIDATE to A/B against the current Qwen and promote only if tool-calling
# + the pipelines hold up — NOT an automatic swap. Pure I/O; safe during the video A/B. Instant rollback by
# pointing serve_llamacpp.sh back at the current gguf. Log: /mnt/ai-models/logs/qwen27b-unc-dl.log
set -uo pipefail
LOG=/mnt/ai-models/logs/qwen27b-unc-dl.log
mkdir -p /mnt/ai-models/logs
exec > >(tee -a "$LOG") 2>&1
echo "================ qwen3.6-27b uncensored download START $(date) ================"

PY=/mnt/ai-models/envs/imagegen/bin/python      # has huggingface_hub
DEST=/mnt/ai-models/qwen3.6-27b/gguf-uncensored
mkdir -p "$DEST"

"$PY" - <<'PY'
from huggingface_hub import hf_hub_download
import os
repo = "HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive"
cache = "/mnt/ai-models/caches/hf"
dest = "/mnt/ai-models/qwen3.6-27b/gguf-uncensored"
files = [
    "Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",          # LM ~32 GB
    "mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-f16.gguf",      # vision ~0.9 GB
]
for f in files:
    dst = os.path.join(dest, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print("HAVE", dst); continue
    print("GET", f, flush=True)
    src = hf_hub_download(repo_id=repo, filename=f, cache_dir=cache)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.realpath(src), dst)
    print("LINKED", dst)
print("QWEN27B UNCENSORED GGUF READY")
PY
ls -lL "$DEST"/*.gguf 2>/dev/null | awk '{print $5, $NF}'
echo "================ qwen3.6-27b uncensored download DONE $(date) ================"
