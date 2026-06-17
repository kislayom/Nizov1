#!/usr/bin/env python3
"""Nizo text->video generator — LTX-Video via diffusers, run as a subprocess in the isolated
imagegen venv. The voice sidecar invokes this inside `with llama_paused():` (full 48 GB free).
Prints the output .mp4 path on success.

LTX-Video: fast DiT video model, text->video (and image->video). Constraints: width/height must be
multiples of 32; num_frames should be 8*k + 1. Defaults ~4 s at 24 fps, 704x480.
"""
import argparse
import sys
import torch

CACHE = "/mnt/ai-models/caches/hf"
MODEL = "Lightricks/LTX-Video"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default="")                 # optional: image->video conditioning
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--frames", type=int, default=97)      # 8*12 + 1 → ~4 s at 24 fps
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()

    try:
        from diffusers.utils import export_to_video
    except Exception as e:
        print(f"diffusers import failed: {e}", file=sys.stderr)
        return 2

    neg = "worst quality, blurry, jittery, distorted"
    w = (a.width // 32) * 32
    h = (a.height // 32) * 32
    frames_n = ((a.frames - 1) // 8) * 8 + 1

    if a.image:
        from diffusers import LTXImageToVideoPipeline
        from diffusers.utils import load_image
        pipe = LTXImageToVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE)
        pipe.to("cuda")
        img = load_image(a.image)
        out = pipe(image=img, prompt=a.prompt, negative_prompt=neg,
                   width=w, height=h, num_frames=frames_n, num_inference_steps=a.steps)
    else:
        from diffusers import LTXPipeline
        pipe = LTXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE)
        pipe.to("cuda")
        out = pipe(prompt=a.prompt, negative_prompt=neg,
                   width=w, height=h, num_frames=frames_n, num_inference_steps=a.steps)

    export_to_video(out.frames[0], a.out, fps=a.fps)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
