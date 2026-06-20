# LTX-2.3 video workflows — 4K upscale ladder & long video (parameter map)

Both files are ComfyUI **API-format** dicts: `{ "<node_id>": { "class_type": ..., "inputs": {...} } }`.
Submit via `comfy_client.ComfyClient().queue(wf)` (or `run_with_progress`). ALWAYS wrap queue+wait in the
VRAM choreography: `sudo systemctl stop nizo-llama` → wait until `nvidia-smi` shows ≥40 GB free → queue →
on finish/error `ComfyClient.free()` → `sudo systemctl start nizo-llama` (try/finally) → verify
`curl http://127.0.0.1:8080/health == 200`. A helper that does all of this is `/tmp/run_render.py`
(kept on the server; see its source). Base checkpoint defaults to the clean distilled
`ltx-2.3-22b-distilled-fp8.safetensors`, 8 steps, cfg 1.0 — same as `ltx_sulphur_t2v.json`.

Both reuse the proven base stack: CheckpointLoaderSimple + LTXAVTextEncoderLoader(gemma) +
LoraLoaderModelOnly(distilled speed LoRA) + ModelSamplingLTXV + CLIPTextEncode×2 + LTXVConditioning +
EmptyLTXVLatentVideo + KSamplerSelect(euler) + CFGGuider(cfg 1.0) + RandomNoise.

---

## ltx_4k_upscale.json — 4K spatial upscale ladder (two extra refine stages)

Node graph (base gen → x2 latent upsample + refine → x2 latent upsample + refine → tiled decode):

```
[1 ckpt]+[2 textenc] → [3 lora] → [4 ModelSamplingLTXV] ─┐
[5 pos]/[6 neg] → [7 LTXVConditioning] → [11 CFGGuider] ──┤ (guider reused by all 3 sampler passes)
STAGE 1 (1280x704):  [8 EmptyLTXVLatentVideo] → [9 LTXVScheduler] ┐
                     [10 KSamplerSelect]+[12 RandomNoise] → [13 SamplerCustomAdvanced] → base latent
UPSAMPLE 1 (x2 → 2560x1408): [20 LatentUpscaleModelLoader] → [21 LTXVLatentUpsampler]
REFINE 1:            [22 ManualSigmas "0.4219,0.21,0.0"]+[23 RandomNoise] → [24 SamplerCustomAdvanced]
UPSAMPLE 2 (x2 → 5120x2816): [30 LTXVLatentUpsampler] (reuses model 20)
REFINE 2:            [31 ManualSigmas "0.30,0.15,0.0"]+[32 RandomNoise] → [33 SamplerCustomAdvanced]
DECODE:              [40 LTXVTiledVAEDecode 3x2 tiles] → [41 CreateVideo 24fps] → [42 SaveVideo h264]
```

The two `LTXVLatentUpsampler` nodes each apply the **x2 spatial latent upscaler model**
(`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`); each is followed by a short partial-denoise
`SamplerCustomAdvanced` (low-sigma `ManualSigmas`) that re-injects high-frequency detail at the new
resolution. `LTXVTiledVAEDecode` (spatial tiling) keeps the >4K decode well under VRAM.

| param            | node id | input key       | current / note |
|------------------|---------|-----------------|----------------|
| prompt           | "5"     | text            | koi pond |
| negative         | "6"     | text            | (neg) |
| base width       | "8"     | width           | 1280  (final = ×4) |
| base height      | "8"     | height          | 704   (final = ×4) |
| length (frames)  | "8"     | length          | 49 (÷8 +1; e.g. 49/97/121) |
| base steps       | "9"     | steps           | 8 |
| cfg              | "11"    | cfg             | 1.0 |
| base seed        | "12"    | noise_seed      | 424242 |
| refine-1 sigmas  | "22"    | sigmas          | "0.4219, 0.21, 0.0" (start lower for stronger upscale fidelity, higher for more change) |
| refine-1 seed    | "23"    | noise_seed      | 424243 |
| refine-2 sigmas  | "31"    | sigmas          | "0.30, 0.15, 0.0" |
| refine-2 seed    | "32"    | noise_seed      | 424244 |
| upscale model    | "20"    | model_name      | ltx-2.3-spatial-upscaler-x2-1.1.safetensors |
| decode tiles     | "40"    | horizontal_tiles / vertical_tiles | 3 / 2 (raise if higher res OOMs) |
| frame_rate(cond) | "7"     | frame_rate      | 25.0 |
| fps (out)        | "41"    | fps             | 24.0 |
| checkpoint       | "1"+"2" | ckpt_name       | ltx-2.3-22b-distilled-fp8.safetensors |

**Resolution tiers** (final = base × 4 via two x2 upsamples):
- base 1280×704  → 5120×2816  (delivered proof; well beyond UHD 3840×2160)
- For an exact UHD-ish 4K, set base 960×544 → 3840×2176, or run only ONE upsample stage
  (delete nodes 30–33, wire 40.latents ← ["24",0]) from base 1280×704 → 2560×1408 (1440p).

**Proof** (`/mnt/ai-models/comfy/output/ltx_4k_upscale_proof.mp4`):
- ffprobe: **5120×2816**, h264, 24/1 fps, 49 frames, 2.04 s, 8.86 MB
- wall: **154 s**;  peak VRAM (sampled): **~28 GB**  → huge headroom on 48 GB
- frame inspected: coherent koi-pond scene, genuine high-freq detail (not a soft upscale)

---

## ltx_long_video.json — ≥2-minute video via native temporal tiling (LTXVLoopingSampler)

Node graph (single looping sampler that internally chunks the full-length latent into overlapping
temporal tiles, conditioning each new tile on the previous tile's end frames):

```
[1 ckpt]+[2 textenc] → [3 lora] → [4 ModelSamplingLTXV]
[5 pos]/[6 neg] → [7 LTXVConditioning] → [11 CFGGuider cfg 1.0]
[8 EmptyLTXVLatentVideo (full target length)]
[9 ManualSigmas (8-step distilled schedule)]   ← NOT LTXVScheduler (see CAUTION)
[10 KSamplerSelect euler] + [12 RandomNoise]
        → [13 LTXVLoopingSampler] → [14 LTXVTiledVAEDecode] → [15 CreateVideo 24fps] → [16 SaveVideo]
```

CAUTION (durable, cost us one black render): `LTXVLoopingSampler` must be fed **explicit sigmas via
`ManualSigmas`**, NOT `LTXVScheduler`. LTXVScheduler with `stretch:true` emits NaN-padded sigmas
(`tensor([nan,nan,...,0.])`) that the base `SamplerCustomAdvanced` tolerates but the looping sampler
silently does NOT denoise from → a fully **black** output that still "succeeds" (tell-tale: a 30 s 720p
mp4 only ~160 KB, identical-size frames). Fix in this file: node "9" is `ManualSigmas` with the proven
distilled 8-step schedule `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`.
Also feed the looping sampler a CFGGuider (cfg 1.0) — the node tooltip says "must be STGGuiderAdvanced"
but that is only a quality recommendation; CFGGuider works (confirmed against the official
LTX-2_V2V_Detailer example which also uses CFGGuider here).

| param                 | node id | input key                       | current / note |
|-----------------------|---------|----------------------------------|----------------|
| prompt                | "5"     | text                             | aerial landscape |
| negative              | "6"     | text                             | (neg) |
| width                 | "8"     | width                            | 1280 |
| height                | "8"     | height                           | 704 |
| **length (frames)**   | "8"     | length                           | **2881** (=120 s @24fps; ÷8 +1). 721=30 s, 1441=60 s |
| sigmas (steps)        | "9"     | sigmas                           | 8-step distilled schedule (string) |
| cfg                   | "11"    | cfg                              | 1.0 |
| seed                  | "12"    | noise_seed                       | 424242 |
| temporal_tile_size    | "13"    | temporal_tile_size               | 96 (frames/tile in pixel space; 80–200 for long) |
| temporal_overlap      | "13"    | temporal_overlap                 | 32 (≈⅓ of tile_size; smooth transitions) |
| temporal cond strength| "13"    | temporal_overlap_cond_strength   | 0.5 (↑ = stronger temporal consistency) |
| adain_factor          | "13"    | adain_factor                     | 0.15 (0.1–0.3 prevents long-run oversaturation/drift) |
| spatial tiles         | "13"    | horizontal_tiles / vertical_tiles| 1 / 1 (raise only for >1080p frames) |
| frame_rate(cond)      | "7"     | frame_rate                       | 25.0 |
| fps (out)             | "16"→15 | fps                              | 24.0 |
| checkpoint            | "1"+"2" | ckpt_name                        | ltx-2.3-22b-distilled-fp8.safetensors |

Tile math: first tile = `temporal_tile_size+1` pixel frames; each later tile adds
`temporal_tile_size−temporal_overlap` (=64) frames. VRAM is bounded by ONE tile regardless of total
length — that is the whole point: arbitrarily long videos at constant ~34 GB peak.

**Proof — 30 s validation run** (`/mnt/ai-models/comfy/output/ltx_long_video_proof.mp4`, length=721):
- ffprobe: **1280×704**, h264, 24/1 fps, **721 frames, 30.04 s**, 8.64 MB
- wall: **258 s** (~11 temporal tiles @ ~21 s each + one full-video decode + model load)
- peak VRAM (sampled): **~34.7 GB**; momentary VAE-decode spike ~46 GB (no OOM)
- frames inspected at 1 s / 15 s / 29 s: coherent aerial golden-hour landscape, world stays
  consistent across all tiles, no collapse/drift — temporal tiling holds.

**Full 120 s** (length=2881, the saved default): one full render was attempted. Measured pace was a
clean **~21.5 s per temporal tile** (45 tiles → ~16 min sampling) at a **bounded ~41 GB** VRAM (constant,
independent of total length — temporal tiling never grows). It was cut off at tile ~12/45 by an
**external `systemctl restart nizo-comfy`** issued by a concurrent operator on the box (see FINAL REPORT),
NOT by OOM or a workflow fault — the run was healthy and on-track. Projection for the full 120 s:
~16 min sampling + ~1.5–2 min tiled decode of 2880 frames ≈ **~18 min wall**. To stay inside a 15-min
budget either render **90 s** (length=2161, ~12 min) or raise `temporal_tile_size` to 144 / overlap 48
(fewer, larger tiles → fewer model-reload boundaries).

### Recommended defaults for Nizo
- **4K option** → `ltx_4k_upscale.json` as-is: base 1280×704, 8 steps, two x2 upsamples → 5120×2816,
  ~2.5 min for a short clip, ~28 GB peak. For exact UHD use base 960×544 (→3840×2176) or one upsample
  stage (→2560×1408 / 1440p) if you want it faster.
- **2-min option** → `ltx_long_video.json` as-is: 1280×704, length 2881 (120 s @24fps),
  `temporal_tile_size=96 / overlap=32 / adain_factor=0.15`, tiled decode. ~18 min, ~41 GB peak (bounded).
  Drop length to 1441 (60 s) / 2161 (90 s) for faster turnaround.
- Both must run under the stop-nizo-llama / free / start-nizo-llama choreography (use `/tmp/run_render.py`).
