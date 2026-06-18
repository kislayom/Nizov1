#!/usr/bin/env python3
"""Nizo sound-effects generator — AudioLDM (v1: CLAP-conditioned latent diffusion, no autoregressive
LM) via diffusers, run in the imagegen venv. MusicGen made these sound like instruments because it's a
*music* model; AudioLDM is trained on AudioCaps/AudioSet, so it produces actual SFX (a frog, a river,
waves, a lion roar). v1 (not v2) is used on purpose: v2's GPT2 generation loop is broken by the venv's
transformers 5.x. Batches all cues into one invocation (model loads once). Repeatable --prompt / --out
(paired). Prints each output path.
"""
import argparse
import sys
import torch

CACHE = "/mnt/ai-models/caches/hf"
MODEL = "cvssp/audioldm-s-full-v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", required=True, help="repeatable; one clip per prompt")
    ap.add_argument("--out", action="append", required=True, help="repeatable; output wav per prompt (paired)")
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    if len(a.prompt) != len(a.out):
        print(f"prompt/out count mismatch ({len(a.prompt)} vs {len(a.out)})", file=sys.stderr)
        return 2

    try:
        from diffusers import AudioLDMPipeline
    except Exception as e:
        print(f"diffusers AudioLDMPipeline import failed: {e}", file=sys.stderr)
        return 2
    import soundfile as sf

    pipe = AudioLDMPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, cache_dir=CACHE).to("cuda")
    neg = "low quality, music, melody, song, instruments, speech, talking, average"
    for prompt, out in zip(a.prompt, a.out):
        gen = torch.Generator("cuda").manual_seed(0)
        audio = pipe(prompt, negative_prompt=neg, num_inference_steps=max(10, a.steps),
                     audio_length_in_s=a.seconds, generator=gen).audios[0]
        sf.write(out, audio, 16000)   # AudioLDM outputs 16 kHz mono
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
