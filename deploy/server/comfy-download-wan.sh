#!/usr/bin/env bash
# Download Wan2.2 T2V A14B (fp8_scaled, native ComfyUI) into the ComfyUI model dirs.
# A14B is a 2-expert MoE: high-noise expert runs early denoise steps, low-noise expert the later ones
# (ComfyUI swaps them, so only one ~14 GB expert is resident at a time). fp8_scaled = "max quality" tier
# that fits 48 GB with Qwen paused. ~35.5 GB total. Uses the imagegen venv's huggingface_hub so it can run
# in parallel with comfy-setup.sh. Logs to /mnt/ai-models/logs/comfy-wan-dl.log
set -uo pipefail
LOG=/mnt/ai-models/logs/comfy-wan-dl.log
mkdir -p /mnt/ai-models/logs
exec > >(tee -a "$LOG") 2>&1
echo "================ wan2.2 a14b download START $(date) ================"

PY=/mnt/ai-models/envs/imagegen/bin/python      # has huggingface_hub already
MODELS=/mnt/ai-models/comfy/models
mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/vae" "$MODELS/loras"

REPO=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
CACHE=/mnt/ai-models/caches/hf

"$PY" - <<PY
from huggingface_hub import hf_hub_download
import os, shutil
repo = "$REPO"; cache = "$CACHE"; models = "$MODELS"
jobs = [
    ("split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
    ("split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",  "diffusion_models"),
    ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",              "text_encoders"),
    ("split_files/vae/wan_2.1_vae.safetensors",                                        "vae"),
]
for path, sub in jobs:
    name = os.path.basename(path)
    dst = os.path.join(models, sub, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print("HAVE", dst); continue
    print("GET", path, flush=True)
    src = hf_hub_download(repo_id=repo, filename=path, cache_dir=cache)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # symlink to keep one copy in the cache; ComfyUI follows symlinks
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.realpath(src), dst)
    print("LINKED", dst, "->", os.path.realpath(src), flush=True)
print("ALL WAN A14B FILES READY")
PY
echo "--- model dir listing ---"
ls -lL "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/vae" 2>/dev/null | awk '{print $5, $NF}'
echo "================ wan2.2 a14b download DONE $(date) ================"
