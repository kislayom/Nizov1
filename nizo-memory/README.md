# nizo-memory

Standalone Java 21 long-term memory service. No Spring. Deployable as an embeddable library or a local HTTP server.

## What it does

Four-tier memory model (WORKING / EPISODIC / SEMANTIC / PROCEDURAL) with hybrid retrieval — BM25 + vector cosine + knowledge-graph traversal + tag-scan, fused via true reciprocal-rank fusion. Multi-user isolation. LLM-based fact extraction with a 10-category schema. Bi-temporal knowledge graph with contradiction detection. Periodic reflection worker that distils old episodes into stable facts.

Designed to be the memory backend for Nizo (personal finance agent) but useful anywhere a stateful agent needs a durable brain.

## Modules

```
nizo-memory-api/       # Pure interfaces (MemoryService, GraphService, Node, Edge) — zero deps
nizo-memory-core/      # Implementation (recall, KG, extraction, reflection, eval harness)
nizo-memory-client/    # HTTP client for consumers
nizo-memory-server/    # Standalone HTTP server (JDK HttpServer)
```

## Quick start

```bash
# Build
mvn clean install -DskipTests

# Run the server (config auto-loads from ./nizo-memory.yaml if present)
java -jar nizo-memory-server/target/nizo-memory-server-*.jar --config nizo-memory.yaml
```

The server listens on `127.0.0.1:8765` by default and persists to `~/.nizo/memory.db` so data survives reinstalls.

## Configuration

Full schema is in `nizo-memory.yaml` at the repo root. Every section is optional — anything omitted falls back to defaults baked into `NizoConfig.defaults()`.

Key knobs:
- `storage.path` — SQLite location (default `~/.nizo/memory.db`)
- `embedder.*` — `ollama` (nomic-embed-text), `onnx`, or `none`
- `extraction.*` — LLM for fact extraction (default `qwen2.5:14b`)
- `reflection.*` — periodic EPISODIC → SEMANTIC worker
- `graph.enabled` — toggle the knowledge-graph channel
- `recall.min_similarity_floor`, `recall.min_top_score` — retrieval floors

## HTTP endpoints

Roughly: `/v1/memory/{remember,learn,recall,stats,health,extract,inspect,forget,forget-user,pin,reconfirm,import}` plus `/v1/compact`.

See `MemoryHttpServer.java` for the full routing table.

## Benchmarks

LongMemEval mini (5-item, qwen2.5:14b + nomic-embed-text): **5/5 = 100%**. See `bench/baseline-report-v2.json`.

To run against the full LongMemEval dataset once downloaded:

```bash
java -cp nizo-memory-core/target/nizo-memory-core-*.jar \
  ai.nizo.memory.eval.LongMemEvalRunner \
  --config nizo-memory.yaml \
  --dataset bench/longmemeval_s.jsonl \
  --limit 50 \
  --out bench/run.json
```

## Testing

```bash
mvn test                                      # unit + journey tests (no Ollama)
mvn test -Dtest='*OllamaLive*'                # Ollama-dependent tests (needs qwen2.5:14b + nomic-embed-text pulled)
```

282 unit + journey tests pass without Ollama. The `RealOllamaTest`, `NizoCTOAcceptanceTest`, `NizoCustomerProblemsTest` suites require a running Ollama at `http://localhost:11434`.

## Key design choices

- **Plain Java 21** — no Spring, no Lombok, no JPA. JAR is small, startup is fast.
- **RRF fusion** across 4 retrieval channels with weighted reciprocal rank; weights scaled so a single-channel rank-1 hit clears the `min_top_score` threshold.
- **Bi-temporal graph edges** (`validFrom` / `validTo` / `invalidatedAt`) with source-priority conflict resolution (`user_stated > conversation > extracted`).
- **Customer-visible controls**: `inspect`, `pin`, `reconfirm`, `forget-about`, `forget-user` (GDPR-grade cascade into graph).
- **Provenance** — every derived fact carries `source_message_id` + `source_excerpt` so agents can answer "why do you think X about me?".
- **Confidence decay** — extracted facts halve in confidence every 180 days unless reconfirmed. Pinned and user-stated facts don't decay.

## Status

Functional alpha. Unit tests green, LongMemEval mini 5/5. Not yet shipped: Postgres/Qdrant backends, auth, encryption at rest, LongMemEval full dataset run, a published benchmark number.

## License

TBD — add a `LICENSE` file before making the repo public.
