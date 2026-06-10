# Nizo

A local-first personal agent powered by Qwen3.6-27B running on a home GPU box.

Java 21. No Spring. Telegram + REST channels. Self-authoring skills. Persistent memory with bi-temporal knowledge graph (vendored from `nizo-memory`).

## Architecture

```
You ─Telegram─▶ nizo-channels ─▶ nizo-agent ─▶ nizo-llm ─▶ vLLM (Qwen3.6-27B-FP8 @ 192.168.4.200:8000)
                                  │       │       │
                                  ▼       ▼       ▼
                            nizo-skills nizo-tools nizo-memory
```

## Build

```bash
./scripts/build.sh
java -jar nizo-app/target/nizo.jar "Hello, who are you?"
```

## Modules

- **nizo-api** — interfaces and value records, zero deps.
- **nizo-llm** — OpenAI-compatible client for vLLM / Ollama.
- **nizo-memory** — vendored layered memory (BM25 + vector + tag + knowledge graph).
- **nizo-skills** — skill engine, agentskills.io-compatible.
- **nizo-tools** — shell, web, file, calendar, email tools.
- **nizo-scheduler** — cron + natural-language reminders.
- **nizo-channels** — Telegram, REST, CLI.
- **nizo-agent** — orchestrator and reflection loop.
- **nizo-app** — bootstrap + uber-jar.

See [CLAUDE.md](CLAUDE.md) for working notes.
