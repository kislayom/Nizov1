"""
Nizo voice sidecar.

FastAPI service exposing two endpoints:
- POST /transcribe — multipart audio file → {text, language, ...} via faster-whisper
- POST /speak      — JSON {text, language?, speaker?} → audio/wav bytes via XTTS-v2

Models are lazy-loaded on first call (so process startup is fast and we don't pin
~5 GB VRAM permanently if the user never speaks). All models live on CUDA.

Run:
  /mnt/ai-models/envs/voice/bin/python /home/kislay/nizo_v1/deploy/server/voice_sidecar.py

Env vars:
- VOICE_HOST (default 127.0.0.1)
- VOICE_PORT (default 7780)
- WHISPER_MODEL (default "large-v3-turbo")
- WHISPER_COMPUTE_TYPE (default "int8_float16")
- XTTS_MODEL (default "tts_models/multilingual/multi-dataset/xtts_v2")
- XTTS_DEFAULT_SPEAKER (default "Ana Florence")
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, Body, HTTPException
from fastapi.responses import Response, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("nizo.voice")

# ─────────────────────────────────────────────────────────────────────────────
# RAM-park / VRAM-wake helper
# ─────────────────────────────────────────────────────────────────────────────
# Box has 186 GB RAM and 48 GB VRAM. Qwen3.6 alone eats 24 GB VRAM, leaving
# only ~24 GB. Loading every TTS/music model directly to GPU runs out fast
# (Kokoro + Whisper + XTTS + MMS + MusicGen-large + YuE-s1 ≈ 35 GB).
#
# Solution: load models to CPU (RAM is plentiful), copy to GPU only for the
# duration of one inference call, then move back. PCIe 4.0 x16 ≈ 25 GB/s, so
# a 14 GB model swaps in ~600 ms. For YuE (3-5 min per song) that's
# negligible. For MusicGen (10 s per gen) it's ~10% overhead.
#
# Hot tier — kept on GPU at all times: none currently. Could promote Whisper
# (1.5 GB, called every voice cycle) if the swap latency starts mattering.

import contextlib
import json
import uuid as _uuid
import threading as _threading
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Async music job store
# ─────────────────────────────────────────────────────────────────────────────
# YuE songs take 5-15 minutes. Synchronous HTTP requests time out client-side.
# Solution: queue compose jobs server-side, return a job ID immediately, let
# the client poll. Results survive sidecar restart (persisted to disk).
JOBS_DIR = Path("/home/kislay/.nizo/music-jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)
_job_locks: dict = {}
_job_locks_lock = _threading.Lock()

def _job_path(job_id: str, ext: str = "json") -> Path:
    return JOBS_DIR / f"{job_id}.{ext}"

def _job_lock(job_id: str) -> _threading.Lock:
    with _job_locks_lock:
        if job_id not in _job_locks:
            _job_locks[job_id] = _threading.Lock()
        return _job_locks[job_id]

def _write_job(job_id: str, **kwargs):
    """Atomic JSON write keyed by job ID."""
    p = _job_path(job_id)
    tmp = p.with_suffix(".json.tmp")
    with _job_lock(job_id):
        existing = {}
        if p.exists():
            try: existing = json.loads(p.read_text())
            except Exception: pass
        existing.update(kwargs)
        existing["jobId"] = job_id
        existing["updatedAt"] = int(__import__("time").time())
        tmp.write_text(json.dumps(existing))
        tmp.replace(p)

def _read_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    if not p.exists(): return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _list_jobs(limit: int = 50) -> list:
    rows = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            d = json.loads(p.read_text())
            rows.append({
                "jobId": d.get("jobId"),
                "status": d.get("status"),
                "prompt": d.get("prompt"),
                "engine": d.get("engine"),
                "createdAt": d.get("createdAt"),
                "updatedAt": d.get("updatedAt"),
                "elapsedSec": d.get("elapsedSec"),
                "errorMessage": d.get("errorMessage"),
                "expandedPrompt": d.get("expandedPrompt"),
                "lyrics": d.get("lyrics"),
            })
        except Exception:
            continue
    return rows



@contextlib.contextmanager
def gpu_swap(model, *, also_move_back=True):
    """Move a torch nn.Module to CUDA for the duration of the with-block,
    then back to CPU + empty the cache. Idempotent: if the model is
    already on CUDA, leaves it there and skips the move-back too."""
    import torch
    was_already_gpu = (next(model.parameters()).device.type == "cuda")
    if not was_already_gpu:
        model.to("cuda")
    try:
        yield model
    finally:
        if also_move_back and not was_already_gpu:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# llama-server pause / resume — frees ~34 GB VRAM that Qwen3.6 holds (24 GB
# weights + ~10 GB KV cache that grows with chat). YuE-7B for >2-min songs
# needs ~14 GB; without pausing, we OOM mid-generation. Trade-off: chat is
# down for the duration of YuE inference (~3-30 min). The iOS Music tab
# already shows "Generating…" so this is invisible to the user.
# ─────────────────────────────────────────────────────────────────────────────
import subprocess as _subprocess

_llama_pause_lock = threading.Lock()
_llama_paused_count = 0  # reentrant: only resume on last release

def _gpu_free_mib() -> int:
    """Return free GPU MiB (GPU 0). Returns 0 on any failure."""
    try:
        out = _subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0

def _pause_llama(min_free_mib: int = 30000, wait_sec: int = 20) -> bool:
    """Stop nizo-llama systemd unit if running, then wait until VRAM frees.
    Reentrant: if already paused (counter > 0), just bumps the counter.
    Returns True if we made a state change worth resuming, False if llama
    wasn't running anyway. Caller MUST pair with _resume_llama() in finally.
    """
    global _llama_paused_count
    with _llama_pause_lock:
        if _llama_paused_count > 0:
            _llama_paused_count += 1
            log.info("llama: already paused (count=%d)", _llama_paused_count)
            return True
        # Check active state
        rc = _subprocess.run(
            ["systemctl", "is-active", "--quiet", "nizo-llama"],
        ).returncode
        if rc != 0:
            log.info("llama: not running, skipping pause")
            return False
        try:
            log.info("llama: stopping nizo-llama to free VRAM for YuE")
            _subprocess.run(
                ["sudo", "-n", "systemctl", "stop", "nizo-llama"],
                check=True, timeout=15,
            )
        except _subprocess.CalledProcessError as e:
            log.error("llama: stop failed: %s", e)
            return False
        # Wait for VRAM to drop
        deadline = __import__("time").time() + wait_sec
        free = 0
        while __import__("time").time() < deadline:
            free = _gpu_free_mib()
            if free >= min_free_mib:
                break
            __import__("time").sleep(0.5)
        log.info("llama: stopped, %d MiB free", free)
        _llama_paused_count = 1
        return True

def _resume_llama() -> None:
    """Restart nizo-llama. Best-effort — logs but doesn't raise. Reentrant
    counterpart to _pause_llama: only restarts on last release."""
    global _llama_paused_count
    with _llama_pause_lock:
        if _llama_paused_count == 0:
            return
        _llama_paused_count -= 1
        if _llama_paused_count > 0:
            log.info("llama: still paused (count=%d)", _llama_paused_count)
            return
        try:
            log.info("llama: restarting nizo-llama")
            _subprocess.run(
                ["sudo", "-n", "systemctl", "start", "nizo-llama"],
                check=True, timeout=15,
            )
        except Exception as e:
            log.error("llama: restart failed (manual restart needed): %s", e)


@contextlib.contextmanager
def llama_paused():
    """Context manager: pause llama for VRAM-heavy work, restart after.
    Safe under exceptions and concurrent callers (reentrant counter)."""
    paused = _pause_llama()
    try:
        yield
    finally:
        if paused:
            _resume_llama()


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loaders (thread-safe singletons)
# ─────────────────────────────────────────────────────────────────────────────

_whisper = None
_whisper_lock = threading.Lock()
_xtts = None
_xtts_lock = threading.Lock()
# Kokoro-82M — tiny (~80M param) VITS-derivative TTS that runs at >5x realtime
# even on CPU. We use it as the "fast" English path because it has ~50ms TTFT
# vs XTTS's ~600ms. Apache 2.0 license. 8 voices: af_alloy, af_aoede, af_bella,
# af_jessica, af_kore, af_nicole, af_nova, af_sarah, af_sky, af_river, am_adam,
# am_eric, am_fenrir, am_michael, am_onyx, am_puck (and bf_*/bm_* for British).
_kokoro = None
_kokoro_lock = threading.Lock()
# MusicGen — Meta's text-to-music transformer. Lazy-loaded; pulls
# `facebook/musicgen-medium` (~6GB VRAM) for instrumental music gen.
# Transformers ships native support; no audiocraft dep needed.
_musicgen = None
_musicgen_processor = None
_musicgen_size = None  # track loaded variant so we can swap
_musicgen_lock = threading.Lock()
# MMS-TTS — Meta's multilingual VITS, per-language models. Used for Indian languages
# where XTTS-v2's quality is mediocre (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati,
# Punjabi, Malayalam, Kannada, Urdu). Each language is a separate ~150 MB model that
# loads in ~20s. Standard transformers — no dependency conflicts with coqui-tts.
_mms = {}                       # lang_code → (model, tokenizer)
_mms_lock = threading.Lock()
# Map our language code → Meta's MMS-TTS model name. Coverage focuses on Indic.
MMS_LANGS = {
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "bn": "facebook/mms-tts-ben",
    "mr": "facebook/mms-tts-mar",
    "gu": "facebook/mms-tts-guj",
    "pa": "facebook/mms-tts-pan",
    "ml": "facebook/mms-tts-mal",
    "kn": "facebook/mms-tts-kan",
    "ur": "facebook/mms-tts-urd-script_arabic",
}


def get_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    with _whisper_lock:
        if _whisper is not None:
            return _whisper
        from faster_whisper import WhisperModel
        model_name = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
        download_root = os.environ.get("WHISPER_DOWNLOAD_ROOT", "/mnt/ai-models/caches/whisper")
        log.info("loading Whisper %s (compute=%s)", model_name, compute_type)
        _whisper = WhisperModel(model_name, device="cuda", compute_type=compute_type,
                                download_root=download_root)
        log.info("Whisper loaded")
        return _whisper


def get_mms(lang: str):
    """Lazy-load Meta MMS-TTS for a specific language. Stays on CPU (~150 MB
    per language); gpu_swap moves it to CUDA during inference."""
    if lang not in MMS_LANGS:
        raise ValueError(f"MMS-TTS doesn't support language '{lang}'")
    if lang in _mms:
        return _mms[lang]
    with _mms_lock:
        if lang in _mms:
            return _mms[lang]
        from transformers import VitsModel, AutoTokenizer
        model_name = MMS_LANGS[lang]
        log.info("loading MMS-TTS for '%s' (%s) → CPU", lang, model_name)
        m  = VitsModel.from_pretrained(model_name)  # CPU
        tk = AutoTokenizer.from_pretrained(model_name)
        _mms[lang] = (m, tk)
        log.info("MMS-TTS '%s' loaded to RAM", lang)
        return _mms[lang]


def get_kokoro():
    """Lazy-load Kokoro-82M (English fast TTS). Apache 2.0, ~250MB on disk.
       Held in CPU RAM since the swap cost is ~50ms — lower than its
       generation latency (~50-200ms per sentence)."""
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    with _kokoro_lock:
        if _kokoro is not None:
            return _kokoro
        from kokoro import KPipeline
        log.info("loading Kokoro-82M → CPU")
        _kokoro = KPipeline(lang_code='a', device='cpu')
        log.info("Kokoro loaded to RAM")
        return _kokoro


def get_musicgen(size: str = "medium"):
    """Lazy-load MusicGen. `size` ∈ {"small","medium","large"}.
       small≈2GB / medium≈6GB / large≈12GB VRAM. Swaps in-place when the
       requested size differs from what's loaded — only one variant lives
       in VRAM at a time (Qwen3.6-27B already eats ~24GB so we can't keep
       both medium AND large resident)."""
    global _musicgen, _musicgen_processor, _musicgen_size
    if _musicgen is not None and _musicgen_size == size:
        return _musicgen, _musicgen_processor
    with _musicgen_lock:
        if _musicgen is not None and _musicgen_size == size:
            return _musicgen, _musicgen_processor
        import torch, gc
        if _musicgen is not None and _musicgen_size != size:
            log.info("unloading MusicGen-%s → reloading -%s", _musicgen_size, size)
            del _musicgen
            del _musicgen_processor
            _musicgen = None
            _musicgen_processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        model_name = f"facebook/musicgen-{size}"
        log.info("loading MusicGen %s → CPU (gpu_swap will move it for inference)", model_name)
        proc = AutoProcessor.from_pretrained(model_name)
        # Stay in fp16 on CPU — that's what most modern x86 with bfloat16
        # support handles fine. Saves half the RAM and matches GPU dtype.
        m = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )  # NOT .to("cuda") — stays on CPU
        _musicgen, _musicgen_processor = m, proc
        _musicgen_size = size
        log.info("MusicGen %s loaded to RAM", size)
        return _musicgen, _musicgen_processor


def get_xtts():
    global _xtts
    if _xtts is not None:
        return _xtts
    with _xtts_lock:
        if _xtts is not None:
            return _xtts
        # Coqui-TTS doesn't accept --no-prompt for XTTS-v2 model agreement check.
        # Setting the env var makes it auto-accept the CPML license.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS
        model_name = os.environ.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
        log.info("loading XTTS %s", model_name)
        _xtts = TTS(model_name=model_name, progress_bar=False).to("cuda")
        log.info("XTTS loaded")
        return _xtts


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Nizo Voice Sidecar", version="1.0")


@app.get("/health")
def health():
    return {
        "ok": True,
        "whisperLoaded": _whisper is not None,
        "xttsLoaded": _xtts is not None,
    }


KOKORO_VOICES = [
    # American female
    "af_bella", "af_nicole", "af_sarah", "af_alloy", "af_aoede",
    "af_jessica", "af_kore", "af_nova", "af_sky", "af_river",
    # American male
    "am_adam", "am_michael", "am_eric", "am_fenrir", "am_onyx", "am_puck",
    # British female / male
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


@app.get("/voices")
def voices():
    """List voices grouped by mode/engine, plus supported languages."""
    out = {
        "modes": ["fast", "natural"],
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                      "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
                      "ta", "te", "bn", "mr", "gu", "pa", "ml", "kn", "ur"],
        "fast": {
            "speakers": KOKORO_VOICES,
            "engine": "kokoro-82m",
            "languages": ["en"],
            "ttft_ms": 50,
        },
        "natural": {
            "speakers": [],
            "engine": "xtts-v2",
            "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                          "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"],
            "ttft_ms": 600,
        },
        "indic": {
            "speakers": ["mms-default"],
            "engine": "mms-tts",
            "languages": list(MMS_LANGS.keys()),
            "ttft_ms": 80,
        },
    }
    # Try to surface XTTS speakers if it's loaded — don't force a load on
    # this endpoint (would hang for 30s on first call).
    if _xtts is not None:
        try:
            sm = getattr(_xtts.synthesizer.tts_model, "speaker_manager", None)
            if sm and sm.speakers:
                out["natural"]["speakers"] = sorted(list(sm.speakers.keys()))
        except Exception:
            pass
    # Backwards-compat keys for older clients
    out["speakers"] = out["fast"]["speakers"] + out["natural"]["speakers"]
    return out


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    vad: bool = Form(True),
    hotwords: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    allowed_languages: Optional[str] = Form(None),
):
    """Transcribe a single audio file.

    Optional fields:
      - language: ISO code; auto-detect if not given. EXACT — overrides allowed_languages.
      - allowed_languages: comma-separated whitelist. When provided with 2+ codes, Whisper's
                  detected language MUST be in this set; if not, we re-transcribe with the
                  first allowed language as fallback. When 1 code, force it directly.
                  Solves the "Whisper hears Hindi as Urdu" misdetection problem.
      - vad: enable VAD filter (default true)
      - hotwords: comma- or newline-separated proper nouns to bias toward
      - initial_prompt: free-form prior context (use sparingly)
    """
    if not audio:
        raise HTTPException(status_code=400, detail="audio file required")
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")

    # Debug save — set NIZO_VOICE_DEBUG_DIR=/tmp/nizo-voice and every transcribe call
    # writes audio + metadata for offline analysis. Useful for "Whisper got language wrong"
    # or "transcription is junk" debugging.
    debug_dir = os.environ.get("NIZO_VOICE_DEBUG_DIR")
    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            ts = int(time.time() * 1000) if 'time' in globals() else 0
            import time as _t
            ts = int(_t.time() * 1000)
            base = os.path.join(debug_dir, f"audio-{ts}")
            with open(base + ".webm", "wb") as f: f.write(data)
            meta = {
                "ts_ms": ts,
                "size": len(data),
                "language_param": language,
                "allowed_languages": allowed_languages,
                "hotwords": hotwords,
                "vad": vad,
            }
            with open(base + ".json", "w") as f:
                import json; json.dump(meta, f)
            log.info("debug-saved %s.webm (%d bytes)", base, len(data))
        except Exception as e:
            log.warning("debug save failed: %s", e)

    # Normalize hot-words
    hot = None
    if hotwords:
        parts = [p.strip() for p in hotwords.replace("\n", ",").split(",") if p.strip()]
        hot = " ".join(parts) if parts else None

    # Normalize allowed_languages
    allowed = None
    if allowed_languages:
        allowed = [c.strip().lower() for c in allowed_languages.split(",") if c.strip()]
        if not allowed: allowed = None

    m = get_whisper()
    forced_lang = language  # explicit `language` wins over allowed_languages

    # If user gave an allowed-set and didn't force a specific language, decide here.
    # faster-whisper's detect_language() requires a decoded numpy array — NOT a path
    # and NOT a raw BytesIO. Use the library's own decode_audio() helper which uses
    # PyAV (ffmpeg) internally to handle webm/opus/etc.
    if not forced_lang and allowed:
        if len(allowed) == 1:
            forced_lang = allowed[0]
        else:
            try:
                from faster_whisper.audio import decode_audio
                audio_arr = decode_audio(io.BytesIO(data))   # → np.ndarray, 16kHz mono
                # faster-whisper 1.2 returns (lang, prob, all_probs) where all_probs is a
                # list of (lang, prob) tuples sorted desc. We need this because Whisper
                # often misdetects Hindi as Urdu (high phonetic overlap) — we want to
                # pick the BEST allowed-language match, not blindly fall back to allowed[0].
                _result = m.detect_language(audio_arr)
                detected = _result[0]
                prob = _result[1] if len(_result) > 1 else 0.0
                all_probs = _result[2] if len(_result) > 2 else []

                # Build a probability lookup for allowed languages.
                allowed_probs = {l: 0.0 for l in allowed}
                for entry in all_probs:
                    try:
                        l, p = entry
                    except (TypeError, ValueError):
                        l = getattr(entry, "language", None)
                        p = getattr(entry, "probability", 0.0)
                    if l in allowed_probs:
                        allowed_probs[l] = max(allowed_probs[l], p)

                # Code-switching bias: when a NON-ENGLISH allowed language has even modest
                # probability (>=0.04), prefer it over English. The user's stated rule:
                # "if i use any hindi word move to hindi". Single Hindi words in mostly
                # English speech typically produce hi probability of 0.05-0.20. Setting
                # threshold to 0.04 catches that. Pure English → hi=~0.001, well below.
                BIAS_THRESHOLD = 0.04
                non_en_candidates = [(l, p) for l, p in allowed_probs.items() if l != 'en' and p >= BIAS_THRESHOLD]
                non_en_candidates.sort(key=lambda x: -x[1])

                if non_en_candidates:
                    forced_lang = non_en_candidates[0][0]
                    log.info("code-switch bias: top-detected '%s' (%.2f), but '%s' (%.2f) is significant — picking non-English",
                             detected, prob, forced_lang, non_en_candidates[0][1])
                elif detected in allowed:
                    forced_lang = detected
                    log.info("detected '%s' (prob=%.2f) ∈ allowed %s", detected, prob, allowed)
                else:
                    # Pick highest-prob allowed lang (handles "Hindi misdetected as Urdu")
                    best_in_allowed = max(allowed_probs.items(), key=lambda kv: kv[1], default=(allowed[0], 0))
                    forced_lang = best_in_allowed[0]
                    log.info("top-detected '%s' (%.2f) not allowed; best allowed: '%s' (%.2f)",
                             detected, prob, best_in_allowed[0], best_in_allowed[1])
            except Exception as e:
                log.warning("detect_language failed: %s — using primary allowed '%s'", e, allowed[0])
                forced_lang = allowed[0]

    buf = io.BytesIO(data)
    try:
        kwargs = {"language": forced_lang, "vad_filter": vad, "beam_size": 5}
        if hot: kwargs["hotwords"] = hot
        if initial_prompt: kwargs["initial_prompt"] = initial_prompt
        segments, info = m.transcribe(buf, **kwargs)
        # segments is a generator; materialize it
        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list).strip()
        return {
            "text": text,
            "language": info.language,
            "languageProbability": float(info.language_probability),
            "duration": float(info.duration),
            "segments": [
                {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                for s in seg_list
            ],
        }
    except Exception as e:
        log.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak")
async def speak(body: dict = Body(...)):
    """Synthesize speech.

    JSON body:
      text:       required, the text to speak
      language:   default 'en' (XTTS supports en/es/fr/de/it/pt/pl/tr/ru/nl/cs/ar/zh-cn/ja/hu/ko/hi)
      speaker:    optional, an XTTS built-in speaker name (e.g. "Ana Florence")
      speaker_wav: optional, server-side path to a reference voice WAV (for cloning)

    Returns audio/wav bytes.
    """
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    language = (body.get("language") or "en").lower()
    # Strip XTTS's "zh-cn"-style suffix for MMS lookup
    base_lang = language.split("-")[0]
    speaker = body.get("speaker") or os.environ.get("XTTS_DEFAULT_SPEAKER", "Ana Florence")
    speaker_wav = body.get("speakerWav") or body.get("speaker_wav")
    # Mode: "fast" (Kokoro for English, sub-100ms TTFT) — the new default —
    # vs "natural" (XTTS, current/legacy). Indic always uses MMS regardless
    # because native quality beats both engines for Hindi/Tamil/etc.
    mode = (body.get("mode") or "fast").lower()
    # Legacy explicit-engine override: "engine":"xtts"|"mms"|"kokoro" wins
    # over `mode`. Kept for ad-hoc testing via curl.
    engine_pref = (body.get("engine") or "").lower()

    is_indic = base_lang in MMS_LANGS
    use_mms = False
    use_kokoro = False
    if engine_pref == "mms":
        use_mms = True
    elif engine_pref == "kokoro":
        use_kokoro = True
    elif engine_pref == "xtts":
        pass  # will fall through to XTTS branch
    else:
        # Auto-route: Indic → MMS (native quality). English/other → mode-driven.
        if is_indic:
            use_mms = True
        elif mode == "fast" and base_lang in ("en",):
            use_kokoro = True
        # else: default to XTTS (multilingual coverage)

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    try:
        if use_mms:
            try:
                model, tokenizer = get_mms(base_lang)
            except ValueError:
                # Fallback to XTTS if MMS doesn't support this language
                log.warning("MMS lacks '%s' — falling back to XTTS", base_lang)
                use_mms = False

        if use_mms:
            import torch, soundfile as sf
            with gpu_swap(model):
                ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
                with torch.no_grad():
                    wave = model(input_ids=ids).waveform
                audio = wave.squeeze().cpu().numpy()
            sf.write(out_path, audio, model.config.sampling_rate)
            engine_label = f"mms:{base_lang}"
        elif use_kokoro:
            try:
                pipe = get_kokoro()
                voice = body.get("speaker") if body.get("speaker", "").startswith(("af_", "am_", "bf_", "bm_")) else "af_bella"
                # Kokoro is small (~80M) — runs fast on CPU directly.
                # No GPU swap needed; keeps things simple.
                import numpy as np, soundfile as sf
                chunks = []
                for _, _, audio in pipe(text, voice=voice, speed=1.0):
                    if hasattr(audio, "cpu"): audio = audio.cpu().numpy()
                    chunks.append(audio)
                wave = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
                sf.write(out_path, wave, 24000)
                engine_label = f"kokoro:{voice}"
            except Exception as e:
                log.warning("Kokoro failed (%s) — falling back to XTTS", e)
                use_kokoro = False
                m = get_xtts()
                kwargs = {"text": text, "file_path": out_path, "language": language}
                if speaker_wav: kwargs["speaker_wav"] = speaker_wav
                else:           kwargs["speaker"] = speaker
                m.tts_to_file(**kwargs)
                engine_label = "xtts-v2-fallback"
        else:
            m = get_xtts()
            kwargs = {"text": text, "file_path": out_path, "language": language}
            if speaker_wav: kwargs["speaker_wav"] = speaker_wav
            else:           kwargs["speaker"] = speaker
            m.tts_to_file(**kwargs)
            engine_label = "xtts-v2"

        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        return Response(content=wav_bytes, media_type="audio/wav",
                        headers={"X-Speaker": speaker, "X-Language": language, "X-Engine": engine_label})
    except Exception as e:
        log.exception("speak failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def web_research(query: str, max_results: int = 3) -> str:
    """Quick web research for music inspiration. Returns short snippet
    text the LLM can use as fresh context. Used to make song prompts +
    lyrics feel current rather than generic. Falls back silently on
    failure — research is opt-in.

    Routes through Nizo's SmartProxy first (best quality), then DuckDuckGo
    HTML as a fallback.
    """
    import requests, html as _html, re
    snippets = []
    # 1) SmartProxy via Nizo's tool endpoint (best, paid)
    try:
        r = requests.post(
            os.environ.get("NIZO_API_URL", "http://localhost:7777") + "/api/tools/web_search",
            timeout=10,
            json={"query": query, "limit": max_results},
        )
        if r.status_code == 200:
            data = r.json()
            for hit in (data.get("results") or [])[:max_results]:
                title = (hit.get("title") or "").strip()
                snippet = (hit.get("snippet") or "").strip()
                if title or snippet:
                    snippets.append(f"- {title}: {snippet}")
            if snippets:
                return "\n".join(snippets)
    except Exception as e:
        log.warning("Nizo web_search failed: %s", e)
    # 2) Fallback — DuckDuckGo HTML scrape
    try:
        r = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=10,
        )
        for snippet in re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)[:max_results]:
            cleaned = re.sub(r"<[^>]+>", "", snippet).strip()
            cleaned = _html.unescape(cleaned)
            if len(cleaned) > 30:
                snippets.append(f"- {cleaned[:240]}")
        return "\n".join(snippets)
    except Exception as e:
        log.warning("DDG fallback failed: %s", e)
        return ""


def generate_lyrics(theme: str, style: str = "", n_segments: int = 2,
                    research: str = "") -> str:
    """Ask the LLM for sectioned song lyrics on a theme. `n_segments`
    matches YuE's segment count — it splits lyrics by [verse]/[chorus]/
    [bridge]/[outro] markers and generates ~30s per segment. So:
        2 segments  ≈ 1 min  → verse + chorus
        4 segments  ≈ 2 min  → verse + chorus + verse + chorus
        6 segments  ≈ 3 min  → +bridge + chorus
        8 segments  ≈ 4 min  → +verse + outro
    Returns sectioned text with markers. Empty string on failure."""
    import requests
    structures = {
        2: "[verse]\n4-6 lines\n\n[chorus]\n4-6 lines",
        4: "[verse]\n4-6 lines\n\n[chorus]\n4-6 lines\n\n[verse]\n4-6 lines (different from first)\n\n[chorus]\n(repeat the same chorus)",
        6: "[verse]\n4-6 lines\n\n[chorus]\n4-6 lines\n\n[verse]\n4-6 lines (different)\n\n[chorus]\n(repeat)\n\n[bridge]\n4 lines\n\n[chorus]\n(repeat)",
        8: "[verse]\n4-6 lines\n\n[chorus]\n4-6 lines\n\n[verse]\n4-6 lines\n\n[chorus]\n(repeat)\n\n[bridge]\n4 lines\n\n[chorus]\n(repeat)\n\n[verse]\n4 lines\n\n[outro]\n4 lines",
    }
    n = max(2, min(8, n_segments - (n_segments % 2)))
    structure = structures.get(n, structures[2])
    system = (
        "You are a songwriter. Write song lyrics on the user's theme. "
        "Use EXACTLY this structure with section markers in square brackets:\n\n"
        + structure +
        "\n\nKeep it singable: simple rhymes, conversational. Match the style "
        "hint if given. If RESEARCH context is provided, weave concrete "
        "fresh references from it into the lyrics so the song feels current "
        "and specific. Output the lyrics only — no commentary, no quotes, "
        "no preamble."
    )
    user_msg = f"Theme: {theme}"
    if style: user_msg += f"\nStyle: {style}"
    if research: user_msg += f"\n\nRESEARCH (fresh context to draw from):\n{research}"
    try:
        r = requests.post(
            os.environ.get("NIZO_LLM_URL", "http://localhost:8080") + "/v1/chat/completions",
            timeout=20,
            headers={"Content-Type": "application/json"},
            json={
                "model": os.environ.get("NIZO_LLM_MODEL", "Qwen/Qwen3.6-27B"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 400,
                "temperature": 0.8,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        # Strip wrapping markdown / quotes the model adds
        text = text.strip().strip('"').strip("'").strip("`")
        return text
    except Exception as e:
        log.warning("lyric gen failed: %s", e)
        return ""


def compose_with_yue(*, prompt: str, final_prompt: str, lyrics: str,
                      n_segments: int = 2, max_new_tokens: int = 3000):
    """Run YuE-7B in a subprocess and return the mixed song as audio/mp3 +
    metadata. YuE is too heavy for in-process loading (14GB stage1 + 2GB
    stage2 + xcodec). Subprocess isolation also means each song gets a
    fresh CUDA context — important because YuE itself doesn't unload.
    Music sidecar stays alive; YuE is fork-and-forget.
    """
    import subprocess, base64, glob
    yue_dir = "/mnt/ai-models/envs/yue-tmp/inference"
    yue_python = "/mnt/ai-models/envs/voice/bin/python"
    s1_model = "/mnt/ai-models/yue/s1-7B-en-cot"
    s2_model = "/mnt/ai-models/yue/s2-1B"

    # Write genre + lyrics to temp files
    work_dir = tempfile.mkdtemp(prefix="yue_")
    genre_path  = os.path.join(work_dir, "genre.txt")
    lyrics_path = os.path.join(work_dir, "lyrics.txt")
    output_dir  = os.path.join(work_dir, "out")
    os.makedirs(output_dir, exist_ok=True)
    # YuE expects the genre/style descriptor as a single line of tags.
    # Final_prompt is already an LLM-enriched comma-separated style.
    # IMPORTANT: YuE's infer.py uses this string as the OUTPUT FILE NAME PREFIX
    # (with "-" replacing spaces). Linux ext4 has a 255-byte filename limit;
    # YuE appends ~60 chars of suffix (`_tp0@93_T1@0_rp1@1_maxtk3000_<UUID>_mixed.mp3`)
    # so we cap the genre to ~150 chars to stay safely under the limit. We
    # truncate at the last comma boundary to keep tags whole.
    genre_for_yue = final_prompt
    if len(genre_for_yue) > 150:
        cut = genre_for_yue[:150].rsplit(",", 1)[0]
        genre_for_yue = cut if len(cut) >= 80 else genre_for_yue[:150]
        log.info("YuE: truncated genre %d→%d chars to fit filename limit",
                 len(final_prompt), len(genre_for_yue))
    with open(genre_path, "w") as f: f.write(genre_for_yue + "\n")
    # YuE splits lyrics into segments by [verse]/[chorus] markers and needs
    # at LEAST 2 sections (the loop skips the first as a title prompt).
    # If the LLM didn't produce markers, synthesize a verse/chorus split.
    formatted_lyrics = lyrics
    has_markers = ("[verse]" in lyrics.lower() or "[chorus]" in lyrics.lower())
    section_count = lyrics.lower().count("[verse]") + lyrics.lower().count("[chorus]") \
                  + lyrics.lower().count("[bridge]") + lyrics.lower().count("[outro]")
    if not has_markers or section_count < 2:
        # Split raw lyrics into halves; tag as verse + chorus.
        non_empty = [ln for ln in lyrics.splitlines() if ln.strip()]
        mid = max(1, len(non_empty) // 2)
        verse_lines = "\n".join(non_empty[:mid]) or "La la la la"
        chorus_lines = "\n".join(non_empty[mid:]) or verse_lines
        formatted_lyrics = f"[verse]\n{verse_lines}\n\n[chorus]\n{chorus_lines}\n"
    with open(lyrics_path, "w") as f: f.write(formatted_lyrics)

    # Hard cap subprocess timeout — scale with segment count.
    # Empirical: ~3 min for 2 segments → ~12 min for 8 segments. Add 50% buffer.
    yue_timeout_min = max(15, int(n_segments * 3 * 1.5))
    log.info("YuE: spawning subprocess, n_segments=%d, timeout=%dmin, output=%s",
             n_segments, yue_timeout_min, output_dir)
    started = time.time() if 'time' in dir() else __import__("time").time()
    import time as _time
    t0 = _time.time()
    # Pause llama-server (frees ~34 GB VRAM Qwen holds). YuE for >2-min songs
    # needs ~14 GB; without this pause, we hit CUDA OOM mid-generation.
    # Reentrant + try/finally — chat resumes whether YuE succeeds or fails.
    with llama_paused():
        try:
            result = subprocess.run(
                [yue_python, "infer.py",
                 "--stage1_model", s1_model,
                 "--stage2_model", s2_model,
                 "--genre_txt", genre_path,
                 "--lyrics_txt", lyrics_path,
                 "--output_dir", output_dir,
                 "--run_n_segments", str(n_segments),
                 "--max_new_tokens", str(max_new_tokens)],
                cwd=yue_dir,
                capture_output=True, text=True,
                timeout=yue_timeout_min * 60,
            )
        except subprocess.TimeoutExpired:
            log.error("YuE subprocess timed out")
            raise HTTPException(status_code=504, detail="YuE generation timed out (15min)")

    elapsed = _time.time() - t0
    if result.returncode != 0:
        log.error("YuE failed (rc=%d): %s", result.returncode, result.stderr[-2000:])
        raise HTTPException(status_code=500,
                            detail=f"YuE inference failed: {result.stderr[-500:]}")

    # Find the mixed mp3
    mp3_files = sorted(glob.glob(os.path.join(output_dir, "vocoder", "mix", "*.mp3")))
    if not mp3_files:
        raise HTTPException(status_code=500, detail="YuE produced no mixed mp3")
    mp3_path = mp3_files[-1]
    with open(mp3_path, "rb") as f: data = f.read()
    log.info("YuE done in %.1fs: %s (%d bytes)", elapsed, mp3_path, len(data))

    # Persist the expensive artifact OUTSIDE /tmp before replying. A reboot wiped a
    # finished 5-minute render once (June 2026) because /tmp is volatile and the
    # proxy had already 502'd — the song existed and nobody could ever fetch it.
    # ~/.nizo/music-out/ is included in the nightly backup tar.
    try:
        import shutil, re as _re
        out_dir = Path.home() / ".nizo" / "music-out"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = _re.sub(r"[^a-zA-Z0-9]+", "-", (prompt or "song"))[:60].strip("-")
        keep = out_dir / f"{stamp}-yue-{safe}.mp3"
        shutil.copyfile(mp3_path, keep)
        log.info("YuE artifact persisted: %s", keep)
    except Exception as e:
        log.warning("YuE artifact persistence failed (non-fatal): %s", e)

    return JSONResponse({
        "wavBase64": base64.b64encode(data).decode("ascii"),
        "engine": "yue-7b",
        "format": "mp3",
        "durationSec": -1,  # YuE doesn't expose this directly
        "sampleRate": 44100,
        "prompt": prompt,
        "expandedPrompt": final_prompt,
        "lyrics": lyrics,
        "elapsedSec": int(elapsed),
    })


def expand_music_prompt_with_research(raw: str, research: str) -> str:
    """Same as expand_music_prompt but with extra fresh-from-the-web
    context to bias toward specific, current references rather than
    generic descriptors. Returns "" on failure (caller falls back)."""
    import requests
    if len(raw) > 280:
        return raw
    system = (
        "You are a music-prompt enhancer for MusicGen/YuE. Given a brief "
        "theme + a few fresh research snippets, expand into ONE LINE under "
        "240 chars: genre + sub-genre, lead instruments, tempo (BPM), mood, "
        "specific era/style/artist references drawn from the research. "
        "Concrete and musical. Output the prompt only."
    )
    try:
        r = requests.post(
            os.environ.get("NIZO_LLM_URL", "http://localhost:8080") + "/v1/chat/completions",
            timeout=12,
            headers={"Content-Type": "application/json"},
            json={
                "model": os.environ.get("NIZO_LLM_MODEL", "Qwen/Qwen3.6-27B"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Theme: {raw}\n\nRESEARCH:\n{research}"},
                ],
                "max_tokens": 160,
                "temperature": 0.5,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        text = text.strip().strip('"').strip("'").strip("`")
        for prefix in ("prompt:", "description:", "music prompt:", "expanded:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        return text[:240] if text else ""
    except Exception as e:
        log.warning("research-expand failed: %s — using non-research expand", e)
        return ""


def expand_music_prompt(raw: str) -> str:
    """Run a short prompt through the local LLM (Qwen3.6) to produce a
    rich MusicGen-style style descriptor: genre + sub-genre + instrumentation
    + tempo + mood + era. MusicGen is sensitive to phrasing — a detailed
    one-line prompt produces noticeably better music than a 2-word seed.

    Hits llama-server on :8080 (no auth, OpenAI-compatible). Returns the
    raw prompt unchanged if the LLM call fails.
    """
    import requests, json
    if len(raw) > 280:
        return raw  # already detailed; don't bloat further
    system = (
        "You are a music-prompt enhancer for MusicGen. Given a brief request, "
        "expand it into ONE LINE under 240 chars with: genre + sub-genre, lead "
        "instruments, tempo (BPM), mood, era/style reference. Concrete and "
        "musical, no fluff. Output the prompt only — no quotes, no commentary, "
        "no preamble."
    )
    try:
        r = requests.post(
            os.environ.get("NIZO_LLM_URL", "http://localhost:8080") + "/v1/chat/completions",
            timeout=8,
            headers={"Content-Type": "application/json"},
            json={
                "model": os.environ.get("NIZO_LLM_MODEL", "Qwen/Qwen3.6-27B"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": raw},
                ],
                "max_tokens": 120,
                "temperature": 0.4,
                # Disable Qwen <think> reasoning — we just want the prompt.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        # Strip wrapping quotes / markdown the model sometimes adds
        text = text.strip().strip('"').strip("'").strip("`")
        # Drop any leading "Prompt:" / "Description:" prefixes
        for prefix in ("prompt:", "description:", "music prompt:", "expanded:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        if not text:
            return raw
        return text[:240]
    except Exception as e:
        log.warning("prompt expand failed: %s — using raw", e)
        return raw


@app.post("/compose-async")
async def compose_async(body: dict = Body(...)):
    """Async version of /compose — kicks off the work in a background
    thread and returns a job ID immediately. Client polls
    /jobs/{job_id} for status + result. Survives sidecar restart
    (persisted to disk). Designed for long YuE runs (5-15 min) where
    HTTP timeouts are unreliable.
    """
    import time as _t
    job_id = _uuid.uuid4().hex[:12]
    _write_job(job_id,
               status="queued",
               prompt=(body.get("prompt") or "")[:240],
               engine=(body.get("engine") or "musicgen"),
               createdAt=int(_t.time()),
               errorMessage=None)

    def _runner():
        import time as _t2, base64
        started = _t2.time()
        try:
            _write_job(job_id, status="running", startedAt=int(started))
            # Build a synthetic Body and call the existing /compose path
            # via direct function call. This reuses all the LLM enrich +
            # research + YuE/MusicGen routing logic.
            class _Resp:
                def __init__(self, status, content_b: bytes, headers: dict):
                    self.status_code = status
                    self.body_bytes = content_b
                    self.headers = headers
            # Re-enter compose synchronously by inlining its logic. We
            # call the public coroutine via asyncio to avoid duplication.
            import asyncio
            result = asyncio.run(compose(body=body))
            # result is a Starlette Response (Response or JSONResponse).
            mp3_or_wav = result.body if hasattr(result, "body") else b""
            ct = result.headers.get("content-type", "audio/wav") if hasattr(result, "headers") else "audio/wav"
            # If JSON envelope, decode the wavBase64 + extract metadata
            if ct.startswith("application/json"):
                env = json.loads(mp3_or_wav.decode("utf-8"))
                _write_job(job_id,
                           status="done",
                           wavBase64=env.get("wavBase64"),
                           format=env.get("format", "wav"),
                           engine=env.get("engine"),
                           sampleRate=env.get("sampleRate"),
                           expandedPrompt=env.get("expandedPrompt"),
                           lyrics=env.get("lyrics"),
                           elapsedSec=int(_t2.time() - started))
            else:
                _write_job(job_id,
                           status="done",
                           wavBase64=base64.b64encode(mp3_or_wav).decode("ascii"),
                           format="wav",
                           elapsedSec=int(_t2.time() - started))
            log.info("[job %s] done in %ds", job_id, int(_t2.time() - started))
        except HTTPException as he:
            _write_job(job_id, status="failed",
                       errorMessage=str(he.detail)[:400],
                       elapsedSec=int(_t2.time() - started))
        except Exception as e:
            log.exception("[job %s] failed", job_id)
            _write_job(job_id, status="failed",
                       errorMessage=str(e)[:400],
                       elapsedSec=int(_t2.time() - started))

    t = _threading.Thread(target=_runner, name=f"music-job-{job_id}", daemon=True)
    t.start()
    return {"jobId": job_id, "status": "queued"}


@app.get("/jobs")
def list_jobs(limit: int = 50):
    return {"jobs": _list_jobs(limit=limit)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, include_audio: bool = True):
    d = _read_job(job_id)
    if d is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not include_audio:
        d.pop("wavBase64", None)
    return d


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    p = _job_path(job_id)
    if p.exists():
        p.unlink()
        return {"ok": True}
    raise HTTPException(status_code=404, detail="job not found")


@app.post("/compose")
async def compose(body: dict = Body(...)):
    """Text-to-music via MusicGen.

    JSON body:
      prompt:        required, free-text style description
      duration_sec:  default 15, max 60
      size:          "small"|"medium"|"large", default "medium"
      guidance:      classifier-free guidance scale, default 3.0
      expand:        bool, default true — pass prompt through Qwen3.6 first
                     to produce a richer MusicGen descriptor

    Returns audio/wav (32kHz mono). Headers include:
      X-Engine, X-Duration, X-SampleRate, X-Prompt (final prompt sent
      to MusicGen, possibly LLM-enriched)
    """
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    # MusicGen's positional embeddings cap at 1500 tokens ≈ 30s. Going
    # above that triggers a CUDA index-out-of-range assertion that takes
    # the entire CUDA context down. Chunked generation for >30s would
    # need a separate impl — for now we cap.
    duration = min(max(int(body.get("duration_sec") or 15), 4), 30)
    size = (body.get("size") or "medium").lower()
    guidance = float(body.get("guidance") or 3.0)
    expand = bool(body.get("expand", True))
    engine = (body.get("engine") or "musicgen").lower()
    if size not in ("small", "medium", "large"):
        size = "medium"
    if engine not in ("musicgen", "yue"):
        engine = "musicgen"

    # LLM-enrich the prompt if requested.
    final_prompt = expand_music_prompt(prompt) if expand else prompt
    if final_prompt != prompt:
        log.info("prompt expand: \"%s\" -> \"%s\"", prompt[:60], final_prompt[:80])

    # YuE always needs lyrics (it's a vocal model). MusicGen optionally
    # gets lyrics for display only.
    want_lyrics = bool(body.get("lyrics", False)) or engine == "yue"
    lyrics_text = (body.get("lyrics_text") or "").strip()
    # YuE n_segments → song length: 2≈1min, 4≈2min, 6≈3min, 8≈4min
    n_segments = max(2, min(8, int(body.get("n_segments") or 2)))
    # Optional web research — when ON, do a quick search on the theme to
    # pull current/specific references the LLM can weave into both the
    # style prompt and the lyrics. Costs 5-15s but produces noticeably
    # less generic output.
    research_text = ""
    if bool(body.get("research", False)):
        log.info("research: searching for theme \"%s\"", prompt[:60])
        research_text = web_research(prompt, max_results=4)
        if research_text:
            log.info("research: got %d chars of context", len(research_text))
            # Re-expand the prompt now that we have research context
            if expand:
                final_prompt = expand_music_prompt_with_research(prompt, research_text) or final_prompt
    if want_lyrics and not lyrics_text:
        log.info("generating lyrics for theme: \"%s\" (n_segments=%d)", prompt[:60], n_segments)
        lyrics_text = generate_lyrics(theme=prompt, style=final_prompt,
                                      n_segments=n_segments, research=research_text)

    # ── YuE branch — sung vocals via subprocess ──
    if engine == "yue":
        return compose_with_yue(
            prompt=prompt,
            final_prompt=final_prompt,
            lyrics=lyrics_text,
            n_segments=n_segments,
            max_new_tokens=int(body.get("max_new_tokens") or 3000),
        )

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    try:
        import torch, soundfile as sf
        model, proc = get_musicgen(size)
        max_new_tokens = int(duration * 50)
        log.info("MusicGen composing: \"%s\" (%ds, size=%s) — swapping to GPU",
                 final_prompt[:60], duration, size)
        with gpu_swap(model):
            inputs = proc(text=[final_prompt], padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                audio = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    guidance_scale=guidance,
                )
        # MusicGen output: float tensor at model.config.audio_encoder.sampling_rate.
        # Cast to float32 — soundfile rejects float16 (which is what fp16 model outputs).
        sr = model.config.audio_encoder.sampling_rate
        wave = audio[0, 0].detach().cpu().to(dtype=torch.float32).numpy()
        sf.write(out_path, wave, sr)
        log.info("MusicGen done — %d samples @ %dHz", len(wave), sr)
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        # If lyrics were requested, return JSON envelope with base64 WAV
        # + lyrics text + final prompt — easier for clients than juggling
        # X-* headers (which Java's HttpServer proxy can mangle).
        # Otherwise return the raw WAV (back-compat).
        import base64
        if want_lyrics or final_prompt != prompt:
            return JSONResponse({
                "wavBase64": base64.b64encode(wav_bytes).decode("ascii"),
                "engine": f"musicgen-{size}",
                "durationSec": duration,
                "sampleRate": sr,
                "prompt": prompt,
                "expandedPrompt": final_prompt,
                "lyrics": lyrics_text,
            })
        return Response(content=wav_bytes, media_type="audio/wav",
                        headers={"X-Engine": f"musicgen-{size}",
                                 "X-Duration": str(duration),
                                 "X-SampleRate": str(sr)})
    except Exception as e:
        log.exception("compose failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(out_path)
        except OSError: pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("VOICE_HOST", "127.0.0.1")
    port = int(os.environ.get("VOICE_PORT", "7780"))
    log.info("Nizo voice sidecar starting on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
