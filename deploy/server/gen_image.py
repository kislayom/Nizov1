#!/usr/bin/env python3
"""Nizo text->image generator — FLUX.1-schnell via diffusers, run as a subprocess in the isolated
imagegen venv (so diffusers' deps never touch the voice/music/YuE venv). The voice sidecar invokes
this inside `with llama_paused():` so the full 48 GB is free. Prints the output path on success.

FLUX.1-schnell: Apache-2.0, guidance-distilled 4-step model — fast, high quality, fits 48 GB.
"""
import argparse
import sys
import torch

CACHE = "/mnt/ai-models/caches/hf"
MODEL = "black-forest-labs/FLUX.1-schnell"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)        # schnell is distilled to ~4 steps
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    try:
        from diffusers import FluxPipeline
    except Exception as e:
        print(f"diffusers import failed: {e}", file=sys.stderr)
        return 2

    # bf16 weights; load straight to GPU (Qwen is paused so 48 GB is free → fastest path).
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE)
    pipe.to("cuda")

    gen = torch.Generator("cuda").manual_seed(a.seed) if a.seed else None
    # schnell is guidance-distilled → guidance_scale must be 0.0.
    image = pipe(
        a.prompt,
        num_inference_steps=max(1, a.steps),
        guidance_scale=0.0,
        width=a.width, height=a.height,
        generator=gen,
    ).images[0]
    image.save(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
