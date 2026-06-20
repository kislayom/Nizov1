# ComfyUI text-to-video workflows — parameter map (for programmatic patching by Nizo)

Both files are API-format dicts: { "<node_id>": { "class_type": ..., "inputs": {...} } }.
Submit via comfy_client.ComfyClient().queue(workflow_dict). Wrap queue+wait in llama_paused()
(stop nizo-llama, wait >=40GB free, queue, wait, ComfyClient.free(), start nizo-llama in finally).

## wan22_t2v.json  (Wan2.2-A14B two-expert, full-step, highest quality)
Node graph: UNETLoader(high)+UNETLoader(low) -> ModelSamplingSD3 x2 -> CLIPLoader(umt5,wan)
  -> CLIPTextEncode pos/neg -> EmptyHunyuanLatentVideo -> KSamplerAdvanced(high 0..split)
  -> KSamplerAdvanced(low split..steps) -> VAEDecode(wan vae) -> VHS_VideoCombine(h264-mp4)

| param    | node id   | input key             | current  |
|----------|-----------|-----------------------|----------|
| prompt   | "7"       | text                  | koi pond |
| negative | "8"       | text                  | (zh neg) |
| width    | "9"       | width                 | 1280     |
| height   | "9"       | height                | 704      |
| length   | "9"       | length                | 49       |
| steps    | "10"+"11" | steps (both = total)  | 16       |
| split    | "10".end_at_step / "11".start_at_step (= steps/2) | 8 |
| cfg      | "10"+"11" | cfg                   | 3.5      |
| seed     | "10"+"11" | noise_seed            | 424242   |
| fps      | "13"      | frame_rate            | 16       |
| shift    | "5"+"6"   | shift                 | 5.0      |
NOTE: when changing steps, also set "10".end_at_step and "11".start_at_step to steps/2,
and "11".end_at_step to steps.

## ltx_sulphur_t2v.json  (LTX-2.3 / Sulphur full checkpoint + distilled LoRA, few-step, fast)
Node graph: CheckpointLoaderSimple(sulphur -> MODEL,CLIP,VAE) + LTXAVTextEncoderLoader(gemma+sulphur -> CLIP)
  -> LoraLoaderModelOnly(distilled) -> ModelSamplingLTXV -> CLIPTextEncode pos/neg -> LTXVConditioning
  -> EmptyLTXVLatentVideo -> LTXVScheduler(SIGMAS) + KSamplerSelect(euler) + CFGGuider + RandomNoise
  -> SamplerCustomAdvanced -> VAEDecode(ckpt vae) -> CreateVideo -> SaveVideo(h264 mp4)
Requires: sulphur_dev_fp8mixed.safetensors symlinked into models/checkpoints/ (done).
To use the official LTX-2.3 distilled checkpoint instead, set node "1" + "2" ckpt_name to
  "ltx-2.3-22b-distilled-fp8.safetensors".

| param      | node id | input key      | current  |
|------------|---------|----------------|----------|
| prompt     | "5"     | text           | koi pond |
| negative   | "6"     | text           | (neg)    |
| width      | "8"     | width          | 1280     |
| height     | "8"     | height         | 704      |
| length     | "8"     | length         | 97       |
| steps      | "9"     | steps          | 8        |
| cfg        | "11"    | cfg            | 1.0      |
| seed       | "12"    | noise_seed     | 424242   |
| frame_rate | "7"     | frame_rate     | 25.0     |
| fps(out)   | "15"    | fps            | 24.0     |
| lora_str   | "3"     | strength_model | 1.0      |
| checkpoint | "1"+"2" | ckpt_name      | sulphur  |
Constraint: width/height divisible by 64; length divisible by 8 + 1 (e.g. 49, 97, 121).
