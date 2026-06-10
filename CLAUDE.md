# Nizo v1 — Working notes for Claude

## What this is

A self-learning, local-first personal agent. Java 21, no Spring. Talks to a local **Qwen3.6-27B (vision-capable)** served by **llama.cpp** on a remote GPU box. Telegram is the primary user channel; REST + CLI are secondary.

**Engine decision history:**
- vLLM FP8: works but ~3.7 tok/s on Blackwell + vllm 0.20.0 (single-GPU; Qwen recommends `--tp-size 8`).
- Ollama Q6: ~36 tok/s text, but [Ollama #14730](https://github.com/ollama/ollama/issues/14730) blocks Qwen3.6 GGUF + mmproj — no vision.
- llama.cpp + Unsloth Q6 + mmproj-F16: ~39 tok/s, vision works, 32K effective ctx (256K OOMs alongside YuE).
- **llama.cpp + Unsloth Q8 + mmproj-F16: ~33 tok/s decode, ~1100 tok/s prefill, vision works, full 256K native ctx, clean OpenAI API on `:8080`.** ← chosen (May 2026, swapped from Q6 for higher fidelity now that VRAM contention with YuE is handled by `llama_paused()`).

## Topology

```
Laptop (you code here)              Server kislay@192.168.4.200
─────────────────────────           ────────────────────────────
nizo_v1/                            /mnt/ai-models/qwen3.6-27b/
├── pom.xml (parent)                ├── gguf-q8/
├── nizo-memory/  (vendored)        │   └── Qwen3.6-27B-UD-Q8_K_XL.gguf  ← LM 33 GB (active)
├── nizo-api/                       ├── gguf-q6/
├── nizo-llm/    (OpenAI client)    │   ├── Qwen3.6-27B-UD-Q6_K_XL.gguf  ← fallback 24 GB
├── nizo-skills/                    │   └── mmproj-F16.gguf              ← vision 885 MB (active)
├── nizo-tools/                     ├── fp8/                              ← vLLM (shelved)
                                    └── logs/
├── nizo-scheduler/
├── nizo-channels/ (Telegram)       /mnt/ai-models/envs/
├── nizo-agent/  (orchestrator)     ├── vllm/                             ← Python 3.12 venv (shelved)
├── nizo-app/    (uber-jar)         ├── serve_llamacpp.sh                 ← THE serve script
├── deploy/server/                  ├── restart_llamacpp.sh
│  ├── serve_llamacpp.sh            └── voice/                            ← venv for voice sidecar
│  ├── restart_llamacpp.sh
│  ├── start_nizo.sh                ~/llama.cpp/build/bin/llama-server    ← OpenAI API on :8080
│  ├── start_voice.sh               ~/.cache/huggingface/token            ← HF_TOKEN source
│  ├── voice_sidecar.py             /opt/cuda/                            ← CUDA 13.1 (set CUDA_HOME)
│  ├── nizo-fan-curve.{py,sh}
│  ├── logrotate-nizo               /etc/systemd/system/nizo-{llama,voice,app}.service
│  └── systemd/                                       ← unit files (mirrored from this dir)
├── scripts/build.sh
├── scripts/wake-server.sh          ← WoL → SSH-poll, used from the laptop
├── .env.example                    ← copy to ~/.nizo/.env on the server
└── CLAUDE.md
```

> The shelved vLLM scripts (`serve_vllm.sh`, `restart_vllm.sh`, `stop_vllm.sh`) were
> removed 2026-05-05. vLLM is documented in the engine-decision section below; restore
> from git history if you need them.

**Default LLM endpoint**: `http://localhost:8080` (works on server; via SSH tunnel on laptop). Model alias: `Qwen/Qwen3.6-27B`.

## Build

```bash
./scripts/build.sh           # full build, skips tests, ~20s clean
./scripts/build.sh --quick   # incremental nizo-app + dependencies
```

Produces `nizo-app/target/nizo.jar` (uber-jar, runnable).

## Run locally

```bash
java -jar nizo-app/target/nizo.jar "Tell me about yourself."
```

Env vars:

| Var | Default | Notes |
|-----|---------|-------|
| `NIZO_LLM_URL` | `http://localhost:8080` | llama.cpp OpenAI endpoint |
| `NIZO_LLM_MODEL` | `Qwen/Qwen3.6-27B` | alias served by llama.cpp (`--alias`) |
| `NIZO_LLM_TOKEN` | (none) | optional bearer |
| `NIZO_LLM_TEMP` | (none) | float (Qwen recs: 1.0 thinking, 0.7 instruct) |
| `NIZO_LLM_MAX_TOKENS` | (none) | int |

**Laptop dev**: open SSH tunnel because the laptop can't direct-route to `192.168.4.200:8080` (different subnet on the home network):

```bash
ssh -fN -L 8080:localhost:8080 kislay@192.168.4.200
```

## Test discipline (read first)

**Never run `mvn test` from the root without permission.** `nizo-memory/nizo-memory-core` has live-Ollama tests (`RealOllamaTest`, `NizoCTOAcceptanceTest`, `NizoCustomerProblemsTest`) that hit `localhost:11434` and a benchmark suite (`LongMemEvalRunner`) that takes minutes. The laptop has been overheated by this once.

Targeted runs only:

```bash
mvn test -pl nizo-memory/nizo-memory-core -Dtest=ClassName
mvn -pl nizo-memory/nizo-memory-core compile
mvn -pl nizo-llm test -Dtest='*Client*'
```

Build script `./scripts/build.sh` already passes `-DskipTests`.

### TEST AS USER (mandatory — don't skip even if user doesn't ask)

For ANY change touching the Stocks pipeline, web UI, chart fences, sub-skill orchestration, or the agent loop: **a build + unit-test-pass is NOT done.** The task ends only when:

1. Built jar deployed to server (`scp` + `systemctl restart nizo-app`)
2. Pipeline kicked end-to-end via the API for at least one ticker
3. **Driven through Chrome** via the `claude-in-chrome` MCP — actually click the button / load the page / inspect the DOM the way a real user would
4. Verified the rendered output: chart fences resolved to `<div class="...-wrap">` widgets, no raw `chart-X {JSON}` text, no `<em>quoteSummary</em>` underscore-bug, no broken/empty widget warnings
5. Server logs cross-checked: every sub-skill that should have fired has a `done in N ms` line; deterministic orchestrator logs `pipeline DONE for ticker=X in Yms (Z chars)`

If Chrome reports "not connected" — re-establish the SSH tunnel (`ssh -fN -L 7777:localhost:7777 kislay@192.168.4.200`) and reload, do not declare done. If the LLM hallucinated a report instead of running the pipeline (warning sign: `tools=1 iters=2` on a stock chat in journal), that's a regression — fix it before reporting success.

Verified-via-Chrome examples that count as "tested":
- HF-black theme: `bgColor: rgb(10, 10, 10)` returned via JS probe
- Chart fence: `(html.match(/class="fin-wrap"/g) || []).length === 1` for chart-financials
- Bug detector: `bug_em_quoteSummary === 0` in DOM after report renders
- Pipeline progress: 5+ sub-agent tiles transition from `idle` → `done` in the live agent-node DOM

Don't wait to be told. The user has explicitly asked for this rule (May 2026): "build ends when testing ends like a user".

### Deploy path (CRITICAL — easy to get wrong)

The systemd unit runs `/home/kislay/nizo_v1/nizo-app/target/nizo.jar`. **Always scp to that
exact path** — copying to `/home/kislay/nizo.jar` will silently keep the old jar running
and any verification will look like the change "didn't work":

```bash
scp nizo-app/target/nizo.jar kislay@192.168.4.200:/home/kislay/nizo_v1/nizo-app/target/nizo.jar
ssh kislay@192.168.4.200 'sudo systemctl restart nizo-app'
```

Confirm with `md5sum` on both sides — local + remote should match. If they don't match
post-scp, the path is wrong.

### Stock pipeline learnings (May 2026 — durable)

These are non-obvious things the pipeline got wrong, where the fix is in code now:

- **Ring buffer eviction by TokenChunk**: A run emits 75K+ TokenChunk events (one per
  output token). The 5K-slot per-chat ring buffer was being saturated, evicting all
  ToolCallStart/Result events before mid-pipeline reload could replay them. Fix:
  `ChatExecution.emit()` excludes TokenChunk + ThinkingChunk from the ring; only durable
  events get a slot.
- **Parallel sub-agent attribution**: When 5 sub-skills run concurrently, the frontend's
  single `stockState.activeAgent` flips between them rapidly. Inner tool calls collapsed
  onto whichever was most-recently-set. Fix: `SubAgentSkillTool` prefixes child callIds
  with parent skill name (`skill_stock_X::id`); frontend's `parseAgentTag` attributes
  per-tile.
- **LLM bypassing skill_stock_analysis**: For ticker inputs like `^NSEI` or `NIFTY 50`,
  the outer LLM would sometimes skip the orchestrator and call individual sub-skills
  directly or answer from training data (~1KB output). Fix: system prompt has an explicit
  directive that any ticker / market-symbol input MUST route via `skill_stock_analysis`
  as the first and only tool call. Belt: `StockReportStore.save()` rejects persisting
  anything < 20KB (a partial run can never overwrite a good prior run).
- **Dedup destroys good reports**: With per-ticker dedup, a partial 1KB run would
  overwrite a 700KB good run when both shared the same ticker. Fix: `save()`'s purge
  `DELETE` clause only removes older rows whose `final_text` is SHORTER than the new one.
  Combined with the 20KB minimum-persist threshold, the library is now monotonic in
  quality.
- **chart-tech NPE (silent failure)**: `TechnicalIndicatorsTool.val()` returned `Double`
  null; downstream `overallSignal()` took primitive `double` → autounbox NPE crashed the
  whole tool. Fix: added `dval()` helper that returns NaN for missing data + null-safe
  `macdTrend` line; `overallSignal()` already null-guards each input.
- **Yahoo + FMP both exhausted**: After heavy testing, Yahoo v8 chart 429s + FMP daily
  quota exhausts. Fix: `HistoricalPriceTool` now has a 60s in-memory + 24h disk-persisted
  bars cache, **plus** a Stooq fallback (`https://stooq.com/q/d/l/?s=…&i=d`) when both
  Yahoo + FMP fail. Stooq has no rate limit. Indices not supported by Stooq — for those
  we rely on Yahoo recovery + disk cache.
- **Caret-prefix index tickers rejected**: `StockQuoteTool`'s ticker regex was
  `[A-Z0-9._-]{1,16}` which rejected `^NSEI`, `^GSPC`, etc. Fix: regex is now
  `\^?[A-Z0-9._-]{1,16}`. Yahoo's v8 chart endpoint serves these symbols fine.
- **Index mode**: When ticker starts with `^`, the orchestrator runs only the analysts
  that apply to indices (news + sentiment + technical, plus bear/bull/trader debate) and
  skips fundamentals + analyst_estimates which are stock-only. Master report uses
  "Market Outlook" + "Bias: X" instead of "Quick Verdict" + "Rating: X".
- **Pipeline parallelism**: `DeterministicStockOrchestratorTool` now runs (a) prefetch
  of historical_price ‖ technical_indicators, (b) bear ‖ bull researcher. Saves ~2 min
  per typical run with no accuracy loss — both research stages see the same analyst
  context, the trader synthesizes both arguments downstream.
- **Empty sub-skill sections**: Some sub-skills (especially news + sentiment) would make
  many tool calls but never produce a written section, leaving "coverage gap" placeholders
  in the master report. Fix: every sub-skill SKILL.md now has STOP-AND-WRITE rules with
  hard tool budgets (5–8 calls) + an explicit sparse-coverage protocol that mandates
  writing whatever section is available before exiting.
- **Chart-fence wraps**: LLMs would wrap `[CHART:type]` placeholders in empty ` ``` `
  fences (from imitating the SKILL.md example formatting). After expansion this produced
  stray empty code blocks around every chart widget. Fix: pre-pass regexes in
  `SubAgentSkillTool.injectChartFences` strip empty fence wraps around both
  `[CHART:type]` placeholders AND already-canonical fences. Rescue's skip-canonical
  check counts backticks at matchStart so canonical fences never get double-rescued.
- **Chart axis cropping + legends**: `lwc-chart-host` was 360px; volume bars + time axis
  shared the bottom region and labels (year markers) were clipped. Fix: chart now 420px
  + `rightPriceScale.scaleMargins.bottom: 0.22` reserves volume pane room + a dynamic
  legend strip (`.lwc-legend`) builds above the chart with colour swatches per series.
- **Search engines bot-flag the home IP (June 2026)**: Bing serves a challenge page
  (`b_no`, 0 rows despite 70KB HTML), DDG html+lite both serve the anomaly wall,
  Ecosia 403s, Yandex redirects to captcha; Brave/Startpage HTML are JS shells.
  Of 7 engines probed, only **Mojeek** (independent UK crawler) returns organic
  results. `WebSearchTool` chain is now SearXNG → Brave API (`BRAVE_API_KEY`) →
  Mojeek → Bing → DDG → SmartProxy; Mojeek carries keyless traffic. If search
  quality matters more later: Brave free tier (2k/mo) or self-host SearXNG behind
  a different egress IP.
- **News via API, not scraping (June 2026)**: the news analyst's web_search→web_fetch
  path was the slowest, least reliable stage (StockTitan 403, WSJ DataDome, Bing/DDG
  empty) — and the SmartProxy paid fallback now rejects our credentials (403 on
  selfcheck; subscription likely lapsed). Fix: `stock_news` tool backed by Finnhub
  `/company-news` (free tier, 60 req/min) — one ~5s call returns months of dated,
  sourced headlines. `stock_news_analyst/SKILL.md` mandates it as the FIRST call;
  web_search is demoted to coverage-gap fallback (some NSE/BSE names) + macro color.

## Module map

| Module | Deps allowed | Purpose |
|---|---|---|
| `nizo-api` | none | Pure interfaces and value records (`LlmClient`, `ChatMessage`, `ToolCall`, `ToolDef`, ...). Zero runtime deps. |
| `nizo-llm` | api, jackson, slf4j | OpenAI-compatible chat client (vLLM/Ollama/llama.cpp). |
| `nizo-memory/*` | (its own deps) | Vendored from previous project. Plain-Java memory: BM25 + vector + tag + KG fusion. **Don't break it.** |
| `nizo-skills` | api, jackson, slf4j | Skill engine. agentskills.io-compatible. Self-authoring loop later. |
| `nizo-tools` | api, jackson, slf4j | Built-in tool catalogue (shell, web, calendar, ...). |
| `nizo-scheduler` | api, cron-utils, slf4j | Cron + NL reminder parsing. |
| `nizo-channels` | api, telegrambots, slf4j | Telegram long-poll, REST, CLI adapters. |
| `nizo-mcp` | api, jackson, slf4j | MCP (Model Context Protocol) client pool. Spawns subprocess servers + registers their tools. |
| `nizo-agent` | api, llm, skills, tools, memory-api, slf4j | Orchestrator: plan → act → reflect → learn. |
| `nizo-app` | api, llm, slf4j-simple | Bootstrap + uber-jar. |

## Security model (must read)

The web channel binds to **127.0.0.1** by default and gates state-changing endpoints
(POST/PUT/DELETE) with a token from `~/.nizo/web-token` (auto-generated on first start;
delivered to the browser as a cookie when you load `/`). Do **not** expose `:7777`
publicly without a real reverse proxy + additional auth.

Other defenses already in place:
- **SSRF guard** ([nizo-tools/.../net/SsrfGuard.java](nizo-tools/src/main/java/ai/nizo/tools/net/SsrfGuard.java)) blocks loopback / RFC1918 / link-local / cloud-metadata for every URL going out of `WebFetchTool`, `WebSearchTool`, `HttpJsonTool`, `SmartProxyClient`. Override with `NIZO_NET_ALLOW_LOOPBACK=1` for dev.
- **Symlink escape blocked** in `FileReadTool`/`FileWriteTool`/`FileListTool` via `toRealPath()`.
- **Shell env scrubbed** to `PATH/HOME/LANG/USER/SHELL/TERM`. `NIZO_SHELL_ENV_EXTRA=A,B` adds more if needed.
- **Bounded HTTP** ([BoundedHttp](nizo-tools/src/main/java/ai/nizo/tools/net/BoundedHttp.java)) caps response bodies at ~5 MB across all web tools.
- **404 short-circuit** in `WebFetchTool` skips the SmartProxy retry on hallucinated URLs (saves ~15 s per LLM-invented URL).
- **Sub-agent LLM timeout** ([SubAgentSkillTool](nizo-skills/src/main/java/ai/nizo/skills/SubAgentSkillTool.java)) caps each `llm.chat()` at 3 min (`NIZO_SUBAGENT_LLM_TIMEOUT_MS`); tool results truncated to 4 KB before re-feeding (`NIZO_SUBAGENT_TOOL_RESULT_MAX_CHARS`).
- **Schema migrations** via [SchemaMigrator](nizo-agent/src/main/java/ai/nizo/agent/store/SchemaMigrator.java); all three SQLite stores use it. Adding a column means a new `Migration` entry, not editing the baseline DDL.
- **Chat inbox bounded** to 64 messages per chat (`NIZO_CHAT_INBOX_CAP`); overflow returns HTTP 429.

## Conventions

- **Plain Java 21.** No Spring. No Lombok. Records over classes, sealed interfaces where useful. Small uber-jar, fast startup.
- **`nizo-api` is the contract.** Other modules depend on it; it depends on nothing.
- **Single user for now.** Multi-user (with `userId` partitioning) is later, but `nizo-memory` already enforces it everywhere — don't undo that.
- **Logging via slf4j-api.** Implementation comes from `slf4j-simple` in `nizo-app` only. Never pull a logger impl into a library module.
- **No Maven profiles for "dev" vs "prod" yet.** Configure via env vars. Add profiles when there's a real deployment difference to manage.

## Server-side state

| Path | Contents |
|---|---|
| `/mnt/ai-models/qwen3.6-27b/gguf-q8/Qwen3.6-27B-UD-Q8_K_XL.gguf` | LM weights (33 GB, **active**) |
| `/mnt/ai-models/qwen3.6-27b/gguf-q6/mmproj-F16.gguf` | vision projector (885 MB, **active**) |
| `/mnt/ai-models/qwen3.6-27b/fp8/` | FP8 safetensors (29 GB, shelved) |
| `/mnt/ai-models/qwen3.6-27b/gguf-q6/Qwen3.6-27B-UD-Q6_K_XL.gguf` | UD-Q6_K_XL.gguf (24 GB, fallback if Q8 footprint becomes a problem) |
| `/mnt/ai-models/qwen3.6-27b/logs/` | download + service logs (`llama-serve.log`, `download.log`) |
| `/mnt/ai-models/caches/hf/` | `HF_HOME` (persisted in `~/.zshrc` and `~/.bashrc`) |
| `/mnt/ai-models/envs/serve_llamacpp.sh` | runs `~/llama.cpp/build/bin/llama-server` |
| `/mnt/ai-models/envs/restart_llamacpp.sh` | clean restart by port |
| `/mnt/ai-models/envs/vllm/` | Python 3.12 venv with vLLM (shelved) |
| `~/llama.cpp/build/bin/llama-server` | the actual server binary |
| `~/.cache/huggingface/token` | source for `HF_TOKEN` env var |

### Restart llama-server

```bash
ssh kislay@192.168.4.200 /mnt/ai-models/envs/restart_llamacpp.sh
```

That script kills any process listening on `:8080`, asks Ollama to evict any loaded model, and `nohup`s `serve_llamacpp.sh`.

GPU: NVIDIA RTX PRO 5000 Blackwell, **48 GB** VRAM (not 72 GB despite lspci string), driver 590.48, CUDA 13.1.

CPU: Ryzen 9 9950X3D, 16C/32T. RAM: 186 GB. Disk: `/mnt/ai-models` is 1.1 TB on `/dev/nvme0n1p6`.

## Decisions worth not relitigating

- **llama.cpp single-GPU, not vLLM.** Qwen's official deploy guidance assumes `--tp-size 8` (8 GPUs). On our single-GPU Blackwell, vLLM 0.20.0 was 3.7 tok/s; llama.cpp gets ~39 tok/s. Llama.cpp also handles vision via `mmproj` (Ollama's path is broken for Qwen3.6, see [#14730](https://github.com/ollama/ollama/issues/14730)).
- **Q8 + mmproj-F16 + 256K ctx, not Q6 or FP8.** Q8 (33 GB) + KV cache (`q8_0/q8_0` at 256K ≈ 9.5 GB) + vision proj (885 MB) ≈ **42.6 / 48 GB** → 5.4 GB free at idle. 256K is the practical max; 512K OOMs on KV alloc (~17 GB needed). Native pretrained context is 262144 so no YARN scaling. FP8 includes a fused vision encoder that pushed past 48 GB. Trade-off vs Q6: -6 tok/s decode (39→33), +higher fidelity, full 256K context (Q6 was effectively 32K because 256K-Q6 + YuE OOM'd before `llama_paused()` existed). Vision tested with 4096×4096 PNG: 4069 image tokens, 16.9s end-to-end for 400 output tokens.
- **VRAM contention with heavy models is solved by `llama_paused()`.** With only 5.4 GB free at idle, YuE-7B (~14 GB) cannot coexist with Qwen. `voice_sidecar.py`'s `compose_with_yue` wraps the YuE subprocess in a `with llama_paused():` block — stops `nizo-llama` systemd unit (frees full 48 GB), runs YuE, restarts on exit (success or failure, reentrant counter for concurrent jobs). Page cache makes Qwen reload in ~5s. Chat is unavailable during YuE inference (~13 min for a 90s n_segments=6 song); iOS Music tab already shows "Generating…" so it's invisible UX. Not a true RAM-park (process restarts) but functionally equivalent because page cache stays warm. Future work: vLLM sleep mode (`POST /sleep?level=1`) for true GPU↔RAM weight swap if multiple heavy models start coexisting.
- **Plain Java for the agent layer.** Spring Boot's value-add doesn't repay the cost (50 MB+ jar, slower iteration). Telegram and cron each have direct Java SDKs that don't need Spring.
- **agentskills.io-compatible skill format.** Hermes Agent and OpenClaw both use it; staying interoperable is free upside.
- **nizo-memory unchanged on import.** It's already proven (LongMemEval mini 5/5). Modify only when an integration need is concrete.
- **`<think>` mode**: Qwen3.6 emits `<think>...</think>` blocks by default. Server flag `--reasoning-format deepseek` puts those in a separate `reasoning_content` JSON field so `content` stays clean. To suppress thinking entirely per-request, send `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}`.
- **Vision input**: send images as base64 data URIs in the OpenAI `image_url` content type. The llama-server can't outbound-fetch URLs (network limited). Format: `data:image/jpeg;base64,<...>`.

## Phases

1. ✅ Plan, server prep, model downloads, Maven multi-module skeleton, mvn package green.
2. ✅ Local LLM serving — llama.cpp + Q8 + mmproj-F16 on `:8080`, ~33 tok/s decode, 256K ctx.
3. ✅ Hello-world chat works end-to-end (laptop → SSH tunnel → llama-server → reply).
4. ✅ Vision verified — model correctly described a real JPEG photo.
5. ⏳ Telegram channel — long-poll bot wired to a chat handler.
6. ⏳ Memory integration via `nizo-memory`.
7. ⏳ Tool calling end-to-end with Qwen3.6.
8. ⏳ Scheduler + NL reminders.
9. ⏳ Skills v1 (load + execute + persist; agentskills.io-compatible).
10. ⏳ Core tools (shell, web fetch, calendar, email).
11. ⏳ Reflection loop (post-task → skill refinement).
12. ⏳ (Optional) Unsloth LoRA fine-tune pipeline on collected trajectories.

## iOS app + voice/music stack (May 2026)

iOS app at `/Users/kislaysinha/claude_projects/nizo_ios/`. Repo separate
from this one. Tabs: **Chat / Stock / Music / Inspector / Settings**.

### Voice (STT + TTS) sidecar — `deploy/server/voice_sidecar.py`

FastAPI on `127.0.0.1:7780`, proxied through Nizo's `/api/voice/*` and
`/api/music/*`. Lazy-loads:

- **Whisper-large-v3-turbo** (faster-whisper, int8_float16) — STT
- **MMS-TTS** per-language (hi/ta/te/bn/mr/gu/pa/ml/kn/ur) — native Indic TTS
- **XTTS-v2** (Coqui, CPML license) — multilingual TTS w/ voice cloning
- **Kokoro-82M** (Apache 2.0) — fast English TTS, ~74ms TTFT vs XTTS ~6.9s
- **MusicGen** small/medium/large (CC-BY-NC) — text→instrumental music

**Mode routing on `/speak`:**
- Indic langs → MMS (regardless of mode)
- `mode=fast` (default) → Kokoro (en only)
- `mode=natural` → XTTS

**MusicGen on `/compose`:**
- 30s hard cap (positional embeddings limit at 1500 tokens — going over
  triggers a CUDA assertion that nukes the entire CUDA context)
- LLM prompt expansion via Qwen3.6 (`expand: true`, default on)
- LLM lyrics generation (`lyrics: true`) — instrumental + displayed lyrics
- size-aware loader unloads previous variant on switch (Qwen takes 33 GB
  at Q8, can't keep both medium AND large MusicGen resident)

### YuE-7B vocal song gen — WIRED (May 2026)

Located at `/mnt/ai-models/yue/{s1-7B-en-cot,s2-1B}` (17 GB total) plus
`/mnt/ai-models/envs/yue-tmp/inference/xcodec_mini_infer/` (1.8 GB).
Vendored repo at `/mnt/ai-models/envs/yue-tmp/`. Patched to use SDPA
attention (flash-attn unbuilt for our cu128 nightly).

**VRAM contention solved by `llama_paused()` in `compose_with_yue`:**
At Q8 + 256K, Qwen reserves ~42.6 / 48 GB → 5.4 GB free, far short of
YuE's ~14 GB need. The compose path:
1. Lyrics + prompt expansion via Qwen (LLM-only, no YuE yet)
2. `with llama_paused():` — `sudo systemctl stop nizo-llama`, wait for
   ≥30 GB free (typically <1s), full 48 GB available
3. Spawn YuE subprocess (Stage1 7B → Stage2 1B → vocoder), 13 min for
   a 90s n_segments=6 song
4. `finally:` restart `nizo-llama` (5s warm reload via page cache)
   regardless of success / OOM / timeout / SIGKILL

YuE spawns under subprocess isolation so each song gets a fresh CUDA
context. Genre prompt is truncated to 150 chars to stay under Linux's
255-byte filename limit (YuE's infer.py uses the prompt as the output
file's name prefix).

**Async job system** (staged but not yet deployed):
`/compose-async` returns `{jobId}` immediately and runs the compose in
a background thread; state persisted to `/home/kislay/.nizo/music-jobs/`
so jobs survive sidecar restart. iOS / web client should poll
`/api/music/jobs/{id}` for completion. Not yet wired in clients —
synchronous `/compose` is fine for now (iPad URLSession bumped to 30 min).

**Calibration TODO:** `n_segments=6` produces 90s, not 180s, because YuE
maps n_segments to lyrics-section count and each section is ~15s. To
hit user-requested duration in seconds, generate longer-per-section
lyrics OR scale n_segments by 2× (n=12 for 3 min).

### iOS architecture

Audio session config: `NizoAudioSession` (singleton) — set once at boot
with `.playAndRecord/.default`, `setActive(true)`, never deactivated.
Components don't touch session config — eliminates the 560557684
"Session activation failed" race.

- **Wake word**: SFSpeechRecognizer en-US, on-device, listening for "nizo"
  variants + stop words ("stop", "wait", "pause"). Owns its own
  AVAudioEngine. Pauses during recording + TTS playback.
- **Voice pipeline**: per-cycle metrics (rec/stt/chat/tts ms), topic-aware
  acks ("Looking up Sydney weather"), tap-to-stop activity strip,
  conversation mode (5s follow-up listen after each reply).
- **Markdown render**: `NizoMarkdownView` with table support and
  `‍```chart` fenced JSON → SwiftUI Charts.
- **Music studio**: live timer, 18 Kokoro voices + XTTS catalog, MusicGen
  small/medium/large picker, optional LLM lyrics.

### What's blocked / TBD

- **In-app WireGuard**: Apple personal-team accounts can't use
  NetworkExtensions. Workaround: WireGuard iOS app's on-demand rules
  (`https://apps.apple.com/app/wireguard/id1441195209`).
- **YuE n_segments→duration calibration**: see "Calibration TODO" in
  YuE-7B section.
- **vLLM sleep mode migration** (optional): replaces systemctl
  stop/start in `llama_paused()` with `POST /sleep?level=1` for true
  GPU↔RAM weight swap. Worth it only if multiple heavy models start
  coexisting (right now YuE is the only contender).
- **Voice barge-in during TTS**: needs migration to single shared
  AVAudioEngine with AEC via `.voiceChat` mode + AVAudioPlayerNode
  for output. ~half day.
- **Server-side stock report cache**: iOS has local cache; web doesn't.
  Should unify in a sqlite `stock_reports` table on the Dell.

### Backups

`nizo-backup.timer` (03:30 daily, `Persistent=true`) runs
[deploy/server/nizo-backup.sh](deploy/server/nizo-backup.sh): WAL-safe `sqlite3 .backup`
snapshots of every `~/.nizo/*.db` + tar of skills/config/jobs → Dell
(`kislay@192.168.5.90:~/backups/nizo/`, 14-day rotation, remote tar verified before
success). Kimaya→Dell auth: dedicated ed25519 key (`kimaya-backup`). Restore: extract,
copy `db/*` into `~/.nizo/`, rest of `.nizo/` as-is, restart nizo-app.

### Server systemd

`nizo-llama` (llama-server :8080), `nizo-voice` (sidecar :7780),
`nizo-app` (Java :7777). All `enabled` for boot, `Restart=always`.
Manual: `sudo systemctl restart nizo-{llama,voice,app}`.
Wake-on-LAN target: `bc:fc:e7:6b:57:dc` (Dell ethernet).
