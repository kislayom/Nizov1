# nizo-memory — Working notes for Claude

This file is for Claude (and future-Claude). The README is for humans. Everything below is **how to work in this repo without burning the user's machine or making bad assumptions**.

For project description, build commands, and config schema → read `README.md` and `nizo-memory.yaml`.

---

## Test discipline (read first)

**Never run `mvn test` without permission.** The full suite has live-Ollama tests (`RealOllamaTest`, `NizoCTOAcceptanceTest`, `NizoCustomerProblemsTest`) that hit `localhost:11434` and a benchmark suite (`LongMemEvalRunner`) that takes minutes. The user's machine has been overheated by this once.

**Default to targeted runs:**

```bash
mvn test -pl nizo-memory-core -Dtest=ClassName        # one class
mvn test -pl nizo-memory-core -Dtest='*Isolation*'    # pattern, single module
mvn -pl nizo-memory-core compile                       # compile-only check
```

To run the full suite, **ask first**. The "safe" subset (no Ollama needed) is everything except the three live-Ollama classes named above.

---

## Module layout

```
nizo-memory-api/      ~30 files   Pure interfaces and records, zero deps. The contract.
nizo-memory-core/     ~58 files   All implementation. Where most work happens.
nizo-memory-server/   ~3 files    JDK HttpServer wrapping core. Many endpoints.
nizo-memory-client/   ~1 file     HTTP client implementing MemoryService.
bench/                            LongMemEval datasets and historical run reports.
config/                           (empty placeholder)
nizo-memory.yaml                  Default config — ALL knobs documented inline.
```

Inside `nizo-memory-core/src/main/java/ai/nizo/memory/`:

| Package | What's there |
|---|---|
| `LayeredMemoryService.java` | The MemoryService implementation. Orchestrates BM25 + vector + tag + graph channels with RRF fusion + scoring + MMR dedup + token-budget packing. |
| `MemoryFactory.java` | Wiring entry point. `fromConfig(NizoConfig)` builds the whole stack. Returns a `Bundle` exposing the underlying store/index/graph for advanced callers. |
| `store/` | `SqliteMemoryStore` (memory items + FTS5), connection pragmas, contradiction demotion. |
| `vector/` | `InMemoryVectorIndex` partitioned by userId. Brute-force cosine for now (Qdrant planned). |
| `graph/` | `KnowledgeGraph`, `SqliteGraphStore`, `ContradictionDetector`, `GraphTraversalEngine`. Bi-temporal edges. |
| `extract/` | `ExtractionPipeline` (LLM-based, 10-category schema) + `GraphFactRouter` (extracted facts → nodes/edges). |
| `compact/` | `CompactionService` — mid-session context compression. |
| `verify/` | `FactVerifier` — self-healing fact verification before recall items reach the prompt. |
| `embed/` | `OnnxEmbedder` — in-process ONNX Runtime embeddings, optional. |
| `llm/` | `OllamaModelClient`, `OllamaEmbeddingClient` — direct HTTP to local Ollama. |
| `reflect/` | Periodic reflection worker — distils old EPISODIC into stable SEMANTIC. |
| `seed/` | `WorldKnowledgeSeed` — bootstrap PROCEDURAL heuristics for a new user. Idempotent. |
| `session/` | Per-conversation state (working memory, recent turns). |
| `canonical/` | "Top-of-prompt" canonical fact index — the table of contents agents see first. |
| `facet/` | Faceted recall (e.g. preference-only, profile-only). |
| `eval/` | `LongMemEvalRunner` — benchmark harness. |
| `config/` | `NizoConfig` record + YAML parsing. |
| `util/` | `DataPaths`, `Tokens`, `Tags`, `Fts`, `Vectors`, `Json`, `Http`. |

---

## Conventions (don't break these)

- **Plain Java 21.** No Spring. No Lombok. No JPA. Records over classes, sealed interfaces where it helps. JAR stays small, startup stays fast.
- **Every operation is scoped to `userId`.** This is a security boundary. Storage, vector index, graph — all partitioned. Tests in `MultiUserIsolationTest` enforce this.
- **All extraction sources are tagged with one of**: `user_stated > conversation > extracted` (priority order). Used by `ContradictionDetector` and demotion logic.
- **Memory tiers**: WORKING (always in prompt) > SEMANTIC (consolidated facts) > PROCEDURAL (heuristics) > EPISODIC (raw events). Recall scoring includes a tier boost.
- **`learnFact()` stores at SEMANTIC, `remember()` stores at EPISODIC.** Tier is the type, not a parameter.
- **Content is written to be retrieval-friendly.** Stored facts include the words a user would naturally use to ask about them. Example: "User's current job role is X" not "User works as X" — so the query "what's my role" matches via FTS. The extraction pipeline handles this; don't undo it.
- **Bi-temporal edges, never destructive deletion.** Edges are invalidated (`validTo` + `invalidatedAt`), not deleted. Preserves audit trail. Same for facts under contradiction — `demoteContradicted` lowers confidence rather than deleting.
- **`source = "user_stated"` carries 0.95 confidence by default** and cannot be downgraded by lower-priority sources.

---

## Common bugs and how to recognize them

| Symptom | Likely cause | Fix |
|---|---|---|
| Recall returns junk top results because FTS matched a noisy keyword (e.g. "current" matching a stock price) | Real embedder not wired or query has overlap with high-frequency words in unrelated content | Check `embedder.backend` in config; ensure stored content is keyword-rich for the question |
| New profile/preference doesn't outrank old contradicted one | `demoteContradicted` not called for that category, or runs AFTER the new fact is stored | Demotion must happen BEFORE `learnFact` for the new value. See `ExtractionPipeline.demoteContradicted` |
| Cross-user data leak | Some new query method missed the `userId` filter | All store/index/graph methods take `userId` as first param. Always |
| Tag-based recall returns nothing | Empty query + no embedder + only tag scan available; check `requiredTags` is in pool-building | Tag scan is the third channel in `LayeredMemoryService.recall`; verify it ran |
| Stale results returned for queries with no real match | Vector top-K returns "filler" items with score ~0 | `MIN_RELEVANT_SIMILARITY = 0.01` threshold filters these — items must have a SIGNAL (FTS hit / vector above threshold / tag / graph) to be in the pool |
| `mvn compile` says "Nothing to compile" but new method isn't visible | Stale install of API JAR | `mvn clean install -DskipTests` from repo root |

---

## Architectural decisions worth not relitigating

- **Memory layer doesn't run jobs.** No background pollers, no notification logic, no agent intelligence. The agent (Nizo) handles cascading recall, deciding what to surface, when to remind. Memory just stores and retrieves accurately.
- **Recall is associative, not query-driven.** The agent passes whatever stimulus it has (current message, date, recent entities) and memory returns relevant items. The agent re-queries with enriched context if it wants depth.
- **The agent enriches queries; the memory layer doesn't NLP.** "What's my role" might need `role job title position work` to match well — but that enrichment lives in the agent's recall strategy, not in memory. (The extraction side handles its half by storing rich content.)
- **Contradiction handling is at the memory tier, not just KG.** When a RESOLUTION or PROFILE change comes in, old contradicted facts are demoted before the new fact is stored. This is what makes "Where do I work?" return Apple after "I left Google for Apple" instead of returning both.

---

## When in doubt

1. **Read the test for the area you're touching.** Tests are 12.9k LOC for ~12k LOC of main code — they document intended behaviour.
2. **`mvn -pl nizo-memory-api compile && mvn -pl nizo-memory-core compile` is your fastest sanity check.** Don't run tests as a smoke check.
3. **Don't add new endpoints to `MemoryHttpServer` without first deciding whether the operation belongs in `MemoryService` (interface) or only in the server.** API additions ripple to the client module too.
4. **`MemoryHttpServer` already has a lot of endpoints** — `extract`, `inspect`, `forget`, `forget-user`, `pin`, `reconfirm`, `import`, `reflect`, `surface`, `index`. Check `MemoryHttpServer.java` lines 82–101 for the full route table before adding a new one.
5. **Benchmark before you optimize.** LongMemEval results live in `bench/`. Compare against `bench/baseline-report-v2.json`.

---

## Status snapshot

- 282+ unit + journey tests passing without Ollama (last verified earlier).
- LongMemEval mini: **5/5 = 100%**.
- Functional alpha. Not shipped: Postgres/Qdrant, auth, encryption at rest, full LongMemEval run.
- The standalone server is wired and runnable; integration into Kimaya/Nizo is the next big step.
