# Nizo v1 — Live session context

> Living document. Update whenever a meaningful capability ships or a decision changes.
> CLAUDE.md is the stable spec; this file tracks what's actually wired and what's queued.
> Last updated: 2026-05-02

## What's running right now

**Server** kislay@192.168.4.200
- `llama-server` on `:8080` — Qwen3.6-27B-Q6_K_XL.gguf + mmproj-F16.gguf, ~39 tok/s
- `nizo-app` (Java) on `:7777` — uber-jar at `/home/kislay/nizo_v1/nizo-app/target/nizo.jar`
  - Bootstrap: 22 tools, 8 skills, SmartProxy enabled (sub-account `smart-t081nydtonpg`)
  - Skills loaded: `stock_analysis`, `stock_fundamentals_analyst`, `stock_news_analyst`,
    `stock_sentiment_analyst`, `stock_technical_analyst`, `stock_bull_researcher`,
    `stock_bear_researcher`, `stock_trader`

**Laptop** (you code here)
- Persistent SSH tunnel via LaunchAgent `dev.kislay.nizo.tunnel` — forwards both `:7777` and `:8080`
- Open Chrome to **http://localhost:7777** to use Nizo

## Capabilities shipped

### Stock Research Lab (top-bar tab)
- Dedicated `📈 Stocks` view, toggleable from `Chat`
- Big ticker input + 9 quick chips (AAPL/NVDA/MSFT/GOOGL/AMZN/META/TSLA/HDFCBANK.NS/INFY.NS)
- **Multi-agent pipeline visualization** — Sanity → 4 Analysts (parallel) → Bear/Bull debate → Trader synthesis. Each agent node lights green when its sub-skill completes
- Live event ticker
- Final report renders with charts inline

### Multi-agent orchestrator (TradingAgents-inspired)
- 7 sub-skills as separate tools the orchestrator invokes:
  - `stock_fundamentals_analyst` — 5y income/balance/cashflow + Buffett scorecard
  - `stock_news_analyst` — last-6-months catalysts + macro
  - `stock_sentiment_analyst` — retail/social sentiment
  - `stock_technical_analyst` — chart patterns, RSI/MACD, support/resistance
  - `stock_bull_researcher` — long thesis with three pillars
  - `stock_bear_researcher` — short thesis with ranked risks
  - `stock_trader` — final synthesis, Buy/Hold/Avoid + fair-value range + position size
- Main `stock_analysis` SKILL.md orchestrates them in sequence
- Verified end-to-end on NVDA: 8 ✅ Buffett checks, BUY-on-dips verdict, 4 charts rendered

### Web search bot-bypass chain (`WebSearchTool`)
1. SearXNG (env-gated `SEARXNG_BASE_URL`) — not deployed yet
2. Brave Search API (env-gated `BRAVE_API_KEY`) — not configured
3. **Bing scrape** — primary no-key fallback, `setLang=en&cc=US`, Linux Firefox UA
4. DuckDuckGo (currently 202-blocked, last resort)
5. **SmartProxy** universal scraper — env-gated, `js_render:true` REQUIRED (verified)
   - `~/.nizo/.env` on server (mode 600) holds `SMARTPROXY_USERNAME` + `SMARTPROXY_PASSWORD`
   - Endpoint `https://scraper.smartproxy.org/v1/query`, source `uni_scraper`
   - **Important**: `js_render:false` returns "Find item by ID failed" 500s; must be true

### Charts (Chart.js)
- Markdown ` ```chart ` fenced JSON blocks render as real interactive charts
- Simplified spec: `{type, title, labels, datasets}` or raw Chart.js config
- Dark theme defaults set in head `<script>`
- Color palette auto-assigned to datasets

### Tools/Skills inspector (right panel)
- All 22 tools listed with name, NATIVE/MCP origin badge, expandable schema
- All 8 skills listed with description, when-to-use, tags
- Per-tool **usage badges**: count, avg duration, error count, last-used relative time
- Bottom-of-panel **Recent Activity (last 100)** with timestamp, args preview, click-to-jump button to the conversation that triggered the call
- Auto-refreshes every 10 s

### Voice (in flight — see "In flight" below)

### Other
- Browser-resize fix (`.msg max-width: clamp(720px, 92%, 1280px)`)
- Aurora visual uplift (drifting gradient blobs behind everything)
- Slash autocomplete popover (Linear-style)
- Server-owned chat workers (per-chat virtual thread, ring buffer 5000 events, multiple SSE subscribers)
- Markdown renderer (~120 LOC, XSS-safe DOM nodes)
- File download from `~/.nizo/workspace/`
- Paste screenshots / drag-drop any file type
- Token telemetry + condense recommendation chip

## Architecture

```
┌─ Laptop ──────────────────┐         ┌─ Server (192.168.4.200) ───────────────────────┐
│                           │  SSH    │                                                 │
│  Chrome → localhost:7777  │ tunnel  │  nizo-app :7777 (Java 21 uber-jar)              │
│  Chrome → localhost:8080  │ ──────▶ │   ├─ AgentLoop (chat-executor, virtual threads) │
│  (LaunchAgent persistent) │         │   ├─ ToolRegistry (22 tools, all MeasuredTool)  │
│                           │         │   ├─ SkillLoader (~/.nizo/skills/)              │
│                           │         │   ├─ UsageTracker (in-mem counters + recent)    │
│                           │         │   ├─ SqliteSessionStore (~/.nizo/sessions.db)   │
│                           │         │   ├─ SqliteUserFactStore (~/.nizo/memory.db)    │
│                           │         │   ├─ McpClientPool (MCP servers, ext processes) │
│                           │         │   └─ WebChannel (HTTP + SSE + EventSource)      │
│                           │         │                                                 │
│                           │         │  llama-server :8080 — Qwen3.6-27B + mmproj      │
│                           │         │   (24 GB Q6_K_XL + 885 MB vision projector)     │
│                           │         │                                                 │
│                           │         │  ~/.nizo/                                       │
│                           │         │   ├─ skills/ (8 SKILL.md files)                 │
│                           │         │   ├─ workspace/ (sandbox for FileTools/ShellTool)│
│                           │         │   ├─ sessions.db, memory.db                     │
│                           │         │   ├─ mcp.json                                   │
│                           │         │   └─ .env (SMARTPROXY_*, mode 600)              │
└───────────────────────────┘         └─────────────────────────────────────────────────┘
```

## Module map (Maven)

| Module           | Depends on              | Purpose |
|---               |---                      |---|
| `nizo-api`       | nothing                 | Pure interfaces + value records (Tool, ChatMessage, ToolDef) |
| `nizo-llm`       | api, jackson, slf4j     | OpenAI-compatible chat client |
| `nizo-memory/*`  | (vendored)              | BM25 + vector + tag + KG fusion |
| `nizo-skills`    | api, jackson, slf4j     | Skill engine (agentskills.io-compatible) |
| `nizo-tools`     | api, jackson, slf4j     | Built-in tools (web/finance/file/shell/memory) + UsageTracker + MeasuredTool |
| `nizo-mcp`       | api, jackson            | MCP client pool, JSON-RPC over stdio |
| `nizo-scheduler` | api, cron-utils         | Cron + NL reminder parsing |
| `nizo-channels`  | api, agent              | Web (HTTP+SSE), CLI, Telegram (TBD) |
| `nizo-agent`     | api, llm, skills, tools, memory-api | Orchestrator: plan → act → reflect → learn |
| `nizo-app`       | api, llm, slf4j-simple, channels, agent | Bootstrap + uber-jar |

## Key decisions worth not relitigating

- **llama.cpp single-GPU** (not vLLM). vLLM 3.7 tok/s vs llama.cpp 39 tok/s on Blackwell single GPU.
- **Q6 + mmproj-F16** (not FP8 / Q8). Fits ~28 GB total, leaves 20 GB headroom for KV + voice models.
- **Plain Java 21**, no Spring, no Lombok. Records, sealed interfaces, virtual threads, pattern matching.
- **agentskills.io-compatible** skill format. Hermes Agent / OpenClaw use it; staying interoperable is free upside.
- **In-memory `UsageTracker`** (not sqlite). Fresh on restart. Adding persistence is a 50-line change later.
- **`MeasuredTool` decorator wraps every tool** at registration time in `Bootstrap`. Transparent to tools; ThreadLocal `chatId` propagation set in `AgentLoop.runStreaming`.
- **EventSource for SSE** (not fetch+ReadableStream). Safari (WebKit) closes `text/event-stream` fetch streams aggressively; native `EventSource` works.
- **Bing primary, SmartProxy fallback** for web search/fetch. DDG bot-blocks server scrapes (202 anomaly modal).

## In flight (current session)

### Voice features (just started)
- **STT**: Whisper Large-v3-turbo via faster-whisper int8 (~2 GB VRAM, 99 langs)
- **TTS**: XTTS-v2 (Coqui, ~1.8 GB, 17 langs incl. Hindi, voice cloning)
- **VAD**: Silero VAD (~1 MB)
- **Architecture**: Python sidecar on `:7780` (FastAPI), Java `VoiceTranscribeTool` + `VoiceSpeakTool` call into it
- **VRAM budget**: 24 GB (LM) + 0.9 GB (mmproj) + 2 GB (Whisper) + 1.8 GB (XTTS) = ~28.7 GB out of 48 GB. Plenty of headroom.

## Open punch list (deferred)

- Per-tool enable/disable toggle (registry filter)
- Edit tool/skill from inspector (open SKILL.md inline, save back)
- BuffettScorecardTool — Java port of kimaya's deterministic ✅/⚠️/❌ scorecard
- StockFundamentalsTool — Yahoo `quoteSummary` for structured income/balance/cashflow without scraping
- Chart-block parser hardening (one block in a recent NVDA report failed parse on a stray non-whitespace char)
- Memory layer integration into AgentLoop (recall + remember per turn) — TODO #16 from CLAUDE.md
- Reflection loop (post-turn self-critique → save_skill) — TODO #17
- Scheduler + NL reminders port from kimaya — TODO #18
- Telegram channel resurrection — TODO #19
- SearXNG container on the server (so search never goes through Bing scrape)

## Build + run

```bash
# From the laptop (always sync first):
./scripts/build-on-server.sh                  # full build (default)
./scripts/build-on-server.sh --quick          # nizo-app + deps only

# Restart the running app
ssh kislay@192.168.4.200 'cd /home/kislay/nizo_v1 && bash deploy/server/start_nizo.sh'

# Restart llama-server
ssh kislay@192.168.4.200 /mnt/ai-models/envs/restart_llamacpp.sh
```

## Pitfalls (so we stop re-discovering them)

- **`./scripts/sync.sh` alone is not enough** — the HTML lives inside the jar. Need `build-on-server.sh` to rebuild the uber-jar.
- **`rsync --update` (not `--ignore-existing`)** for `~/.nizo/skills/` so SKILL.md bug-fixes propagate. Already fixed in `start_nizo.sh`.
- **SmartProxy needs `js_render:true`** — `false` returns "Find item by ID failed" 500s.
- **DuckDuckGo HTML / lite endpoints both bot-block server scrapes**. Use Bing.
- **Bing without `setLang=en&cc=US&setMkt=en-US`** localizes to source-IP geography (returned German MOTOR-TALK from our server).
- **`el(tag, attrs, ...children)` in `index.html`** must coerce numbers to strings or it throws and silently kills the render.
- **EventSource expects named events** — `addEventListener('ToolCallStart', ...)` for each, not `onmessage`.
- **Never run `mvn test` from root** — `nizo-memory/nizo-memory-core` has live-Ollama tests that overheat the laptop.
