#!/usr/bin/env bash
# Nizo video backend — ComfyUI standup (Phase 0 de-risk).
# Isolated venv at /mnt/ai-models/envs/comfy so the working imagegen venv (LTX/FLUX/AudioLDM) is untouched.
# Matched torch+torchvision+torchaudio cu128 nightly for the Blackwell sm_120 GPU. Idempotent-ish; logs to
# /mnt/ai-models/logs/comfy-setup.log. Run on the server: bash ~/nizo_v1/deploy/server/comfy-setup.sh
set -uo pipefail
LOG=/mnt/ai-models/logs/comfy-setup.log
mkdir -p /mnt/ai-models/logs
exec > >(tee -a "$LOG") 2>&1
echo "================ comfy setup START $(date) ================"

COMFY=/mnt/ai-models/comfy
VENV=/mnt/ai-models/envs/comfy
PY=$VENV/bin/python

if [ ! -d "$COMFY/.git" ]; then
  echo "ERROR: $COMFY is not a ComfyUI clone — clone it first."; exit 1
fi

# 1. venv (python 3.14 is the only interpreter on this Arch box)
if [ ! -x "$PY" ]; then
  echo "--- creating venv $VENV ---"
  python3.14 -m venv "$VENV"
fi
"$PY" -m pip install --upgrade pip wheel setuptools

# 2. torch trio — matched cu128 nightly set (do NOT pin individually; let the index resolve a consistent trio)
echo "--- installing torch/torchvision/torchaudio (cu128 nightly) ---"
"$PY" -m pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 || { echo "torch trio install FAILED"; exit 1; }

# 3. ComfyUI core requirements (torch already satisfied → not reinstalled)
echo "--- installing ComfyUI requirements ---"
cd "$COMFY"
"$PY" -m pip install -r requirements.txt

# 4. extras for video + quantized loaders
"$PY" -m pip install gguf imageio-ffmpeg

# 5. custom nodes: VideoHelperSuite (mp4 mux), GGUF loader (optional quant path), KJNodes (utils)
mkdir -p "$COMFY/custom_nodes"; cd "$COMFY/custom_nodes"
clone() { [ -d "$2" ] || git clone --depth 1 "$1" "$2"; }
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git ComfyUI-VideoHelperSuite
clone https://github.com/city96/ComfyUI-GGUF.git ComfyUI-GGUF
clone https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI-KJNodes
for d in */; do
  if [ -f "$d/requirements.txt" ]; then
    echo "--- deps for $d ---"
    "$PY" -m pip install -r "$d/requirements.txt" || echo "WARN: deps for $d had issues (continuing)"
  fi
done

# 6. verify
echo "--- verify ---"
"$PY" -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda_ok', torch.cuda.is_available(), 'sm', torch.cuda.get_device_capability())" \
  && echo "VERIFY OK" || echo "VERIFY FAILED"
echo "================ comfy setup DONE $(date) ================"
