#!/usr/bin/env bash
# Download Sulphur-2-base (fp8mixed) — an uncensored LTX-2.3 fine-tune — into the ComfyUI diffusion_models
# dir. It reuses LTX-2.3's text encoder + VAE + nodes (set up alongside the Wan/LTX A/B), so only the DiT
# checkpoint is needed here. NOTE: this model is uncensored and has no stated license; it is added as a
# NON-DEFAULT video option at the owner's explicit request and is never wired into the kids' story path.
# Pure I/O — safe to run while the GPU A/B is in flight. Logs to /mnt/ai-models/logs/comfy-sulphur-dl.log
set -uo pipefail
LOG=/mnt/ai-models/logs/comfy-sulphur-dl.log
mkdir -p /mnt/ai-models/logs
exec > >(tee -a "$LOG") 2>&1
echo "================ sulphur-2-base download START $(date) ================"

PY=/mnt/ai-models/envs/imagegen/bin/python      # has huggingface_hub
MODELS=/mnt/ai-models/comfy/models
mkdir -p "$MODELS/diffusion_models"

"$PY" - <<'PY'
from huggingface_hub import hf_hub_download
import os
repo = "SulphurAI/Sulphur-2-base"
cache = "/mnt/ai-models/caches/hf"
models = "/mnt/ai-models/comfy/models"
fname = "sulphur_dev_fp8mixed.safetensors"
dst = os.path.join(models, "diffusion_models", fname)
if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
    print("HAVE", dst)
else:
    print("GET", fname, flush=True)
    src = hf_hub_download(repo_id=repo, filename=fname, cache_dir=cache)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.realpath(src), dst)
    print("LINKED", dst, "->", os.path.realpath(src))
print("SULPHUR FP8 READY")
PY
ls -lL "$MODELS/diffusion_models"/sulphur* 2>/dev/null | awk '{print $5, $NF}'
echo "================ sulphur-2-base download DONE $(date) ================"
