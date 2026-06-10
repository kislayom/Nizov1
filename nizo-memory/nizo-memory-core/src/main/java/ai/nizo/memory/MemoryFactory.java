package ai.nizo.memory;

import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.GraphTraversal;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.compact.CompactionService;
import ai.nizo.memory.config.NizoConfig;
import ai.nizo.memory.embed.OnnxEmbedder;
import ai.nizo.memory.extract.ExtractionPipeline;
import ai.nizo.memory.extract.GraphFactRouter;
import ai.nizo.memory.graph.GraphTraversalEngine;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.llm.OllamaEmbeddingClient;
import ai.nizo.memory.llm.OllamaModelClient;
import ai.nizo.memory.seed.WorldKnowledgeSeed;
import ai.nizo.memory.util.DataPaths;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.verify.FactVerifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.time.Duration;

/**
 * One-stop construction for the memory tier. Builds the full stack:
 *
 * <ul>
 *   <li>{@link #local} — in-process layered memory (SQLite + vector index)</li>
 *   <li>{@link #localWithGraph} — adds knowledge graph integration (entity resolution, graph-boosted recall)</li>
 *   <li>{@link #graph} — standalone knowledge graph</li>
 *   <li>{@link #extraction} — extraction pipeline (LLM-based fact extraction → graph + memory)</li>
 *   <li>{@link #compaction} — mid-session context compression</li>
 *   <li>{@link #verifier} — self-healing fact verification</li>
 *   <li>{@link #fromConfig} — wire the whole stack from a parsed YAML config</li>
 * </ul>
 */
public final class MemoryFactory {

    private static final Logger log = LoggerFactory.getLogger(MemoryFactory.class);

    private MemoryFactory() {}

    /** In-process layered memory without graph integration. */
    public static MemoryService local(Path sqlitePath,
                                      EmbeddingClient embedder,
                                      ModelClient summariser,
                                      int consolidateEveryN,
                                      double confidenceFloor) {
        SqliteMemoryStore store = new SqliteMemoryStore(sqlitePath);
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        return new LayeredMemoryService(store, index, embedder, summariser,
                consolidateEveryN, confidenceFloor);
    }

    /**
     * In-process layered memory WITH knowledge graph integration.
     * The graph channel is added to the recall pipeline — entity resolution
     * and 1-hop traversal contribute candidates and a scoring boost.
     */
    public static MemoryService localWithGraph(Path sqlitePath,
                                                EmbeddingClient embedder,
                                                ModelClient summariser,
                                                GraphService graph,
                                                GraphTraversal traversal,
                                                int consolidateEveryN,
                                                double confidenceFloor) {
        SqliteMemoryStore store = new SqliteMemoryStore(sqlitePath);
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        return new LayeredMemoryService(store, index, embedder, summariser,
                graph, traversal, consolidateEveryN, confidenceFloor);
    }

    /** Standalone knowledge graph backed by SQLite. */
    public static GraphService graph(Path sqlitePath) {
        SqliteGraphStore graphStore = new SqliteGraphStore(sqlitePath);
        return new KnowledgeGraph(graphStore);
    }

    /** Graph traversal engine for hop-based expansion. */
    public static GraphTraversal traversal(Path sqlitePath, GraphService graph) {
        SqliteGraphStore graphStore = new SqliteGraphStore(sqlitePath);
        return new GraphTraversalEngine(graphStore, graph);
    }

    /** Extraction pipeline — LLM-based fact extraction routed to graph + memory. */
    public static ExtractionService extraction(ModelClient extractor,
                                                GraphService graph,
                                                MemoryService memory) {
        GraphFactRouter router = new GraphFactRouter(graph);
        return new ExtractionPipeline(extractor, router, memory);
    }

    /**
     * Seed default world knowledge (extraction guardrails, terminology,
     * health/finance/tech facts) into a user's PROCEDURAL memory. Idempotent —
     * safe to call on every startup.
     *
     * <p>Use this for new users; for batch-refreshing existing users with
     * an updated knowledge bundle, see {@link WorldKnowledgeSeed#uploadBatch}.
     *
     * @return number of heuristics seeded (0 if already done)
     */
    public static int seedWorldKnowledge(MemoryService memory, String userId) {
        return WorldKnowledgeSeed.seedIfNeeded(memory, userId);
    }

    /**
     * Try to create an in-process ONNX embedder from a model directory.
     * Returns {@code null} if ONNX Runtime isn't on the classpath or
     * the model files aren't found — caller should fall back to Ollama.
     */
    public static EmbeddingClient onnxEmbedder(Path modelDir, int maxSeqLen) {
        return OnnxEmbedder.tryCreate(modelDir, maxSeqLen);
    }

    /** Compaction service for mid-session context compression. */
    public static CompactionService compaction(ModelClient model, MemoryService memory) {
        if (model == null) return null;
        return new CompactionService(model, memory);
    }

    /**
     * Self-healing fact verifier. Verifies top-k recalled facts against
     * live context before they enter the prompt.
     */
    public static FactVerifier verifier(ModelClient model, int maxFacts,
                                        double outdatedPenalty, double unverifiablePenalty) {
        if (model == null) return FactVerifier.passThrough();
        return new FactVerifier(model, maxFacts, outdatedPenalty, unverifiablePenalty);
    }

    // ===== YAML-driven wiring ===============================================

    /**
     * Build the memory tier from a parsed config. Honours every flag:
     * <ul>
     *   <li>storage.path → SQLite location</li>
     *   <li>vector.backend → vector index (only "inmemory" supported today)</li>
     *   <li>embedder.{enabled,backend,…} → Ollama / ONNX / null</li>
     *   <li>consolidation.{enabled,backend,…} → summariser ModelClient or null</li>
     *   <li>graph.enabled → knowledge graph + 1-hop traversal channel</li>
     *   <li>recall.{consolidate_every_n, confidence_floor} → recall defaults</li>
     * </ul>
     *
     * <p>Returns a {@link MemoryService} fully wired and ready to serve recall
     * requests. Side effect: opens SQLite connections.
     */
    public static MemoryService fromConfig(NizoConfig config) {
        return buildBundle(config).service();
    }

    /**
     * Composite result of wiring the memory stack. Exposes the concrete
     * backing handles so callers (e.g. the reflection worker, HTTP stats
     * endpoints) can work against the same {@link SqliteMemoryStore} and
     * {@link InMemoryVectorIndex} instances as {@link LayeredMemoryService}.
     */
    public record Bundle(
            MemoryService service,
            SqliteMemoryStore store,
            InMemoryVectorIndex index,
            GraphService graph /* nullable */) {}

    /** F1-friendly builder: same behaviour as {@link #fromConfig} but also
     *  returns the underlying handles so you can attach a reflection worker
     *  against the same store + vector index. */
    public static Bundle buildBundle(NizoConfig config) {
        NizoConfig c = config.withDefaults();
        Path dbPath = DataPaths.resolve(c.storage().path());
        log.info("nizo-memory data file: {}", dbPath);

        EmbeddingClient embedder = embedderFromConfig(c);
        ModelClient summariser = consolidationModelFromConfig(c);
        int consolidateEveryN = c.consolidation().consolidateEveryN();
        double confidenceFloor = c.recall().confidenceFloor();
        double minSim = c.recall().minSimilarityFloorOr(0.55);
        double minTop = c.recall().minTopScoreOr(0.55);

        SqliteMemoryStore store = new SqliteMemoryStore(dbPath);
        InMemoryVectorIndex index = new InMemoryVectorIndex();

        MemoryService svc;
        GraphService graph = null;
        if (c.graph().enabled()) {
            graph = graph(dbPath);
            GraphTraversal traversal = traversal(dbPath, graph);
            svc = new LayeredMemoryService(store, index, embedder, summariser,
                    graph, traversal, consolidateEveryN, confidenceFloor, minSim, minTop);
        } else {
            svc = new LayeredMemoryService(store, index, embedder, summariser,
                    null, null, consolidateEveryN, confidenceFloor, minSim, minTop);
        }
        return new Bundle(svc, store, index, graph);
    }

    /**
     * Build the embedder selected by the config. Returns {@code null} when
     * embedding is disabled — recall falls back to FTS / BM25 only.
     */
    public static EmbeddingClient embedderFromConfig(NizoConfig config) {
        NizoConfig.EmbedderConfig ec = config.embedder();
        if (ec == null || !ec.enabled()) return null;
        String backend = ec.backend() == null ? "ollama" : ec.backend().toLowerCase();
        return switch (backend) {
            case "ollama" -> {
                NizoConfig.OllamaConfig o = ec.ollama() == null
                        ? new NizoConfig.OllamaConfig(null, null, null, null)
                        : ec.ollama();
                yield new OllamaEmbeddingClient(
                        o.baseUrlOr("http://localhost:11434"),
                        o.modelOr("nomic-embed-text"),
                        Duration.ofSeconds(o.timeoutSecondsOr(30)));
            }
            case "onnx" -> {
                NizoConfig.OnnxConfig on = ec.onnx() == null
                        ? NizoConfig.OnnxConfig.defaults()
                        : ec.onnx();
                EmbeddingClient onnx = OnnxEmbedder.tryCreate(
                        Path.of(on.modelDir()), on.maxSeqLenOr(128));
                if (onnx == null) {
                    log.warn("ONNX embedder unavailable for {} — disabling embedder",
                            on.modelDir());
                }
                yield onnx;
            }
            case "none" -> null;
            default -> {
                log.warn("Unknown embedder backend '{}' — disabling embedder", backend);
                yield null;
            }
        };
    }

    /** Build the consolidation model — null when disabled. */
    public static ModelClient consolidationModelFromConfig(NizoConfig config) {
        NizoConfig.ConsolidationConfig cc = config.consolidation();
        if (cc == null || !cc.enabled()) return null;
        return ollamaFor(cc.backend(), cc.ollama(), 0.2, 60);
    }

    /** Extraction pipeline — null when disabled or graph/memory missing. */
    public static ExtractionService extractionFromConfig(NizoConfig config,
                                                          GraphService graph,
                                                          MemoryService memory) {
        NizoConfig.ExtractionConfig ec = config.extraction();
        if (ec == null || !ec.enabled()) return null;
        if (graph == null || memory == null) {
            log.warn("Extraction enabled but graph/memory missing — disabling");
            return null;
        }
        ModelClient extractor = ollamaFor(ec.backend(), ec.ollama(), 0.1, 60);
        if (extractor == null) return null;
        return new ExtractionPipeline(extractor, new GraphFactRouter(graph), memory);
    }

    /** Compaction service — null when disabled. */
    public static CompactionService compactionFromConfig(NizoConfig config, MemoryService memory) {
        NizoConfig.CompactionConfig cc = config.compaction();
        if (cc == null || !cc.enabled()) return null;
        ModelClient model = ollamaFor(cc.backend(), cc.ollama(), 0.1, 60);
        if (model == null) return null;
        return new CompactionService(model, memory, cc.reserveBufferTokens());
    }

    /**
     * F1 — Build the reflection worker. Returns {@code null} when disabled
     * or when dependencies (LLM / store) are missing. The caller is
     * responsible for calling {@code start()} and {@code close()}.
     */
    public static ai.nizo.memory.reflect.ReflectionService reflectionFromConfig(
            NizoConfig config,
            MemoryService memory,
            SqliteMemoryStore store,
            ai.nizo.memory.vector.VectorIndex vectorIndex,
            EmbeddingClient embedder) {
        NizoConfig.ReflectionConfig rc = config.reflection();
        if (rc == null || !rc.enabled()) return null;
        ModelClient llm = ollamaFor(rc.backend(), rc.ollama(), 0.1, 60);
        if (llm == null) {
            log.warn("Reflection enabled but LLM backend missing — disabling");
            return null;
        }
        return new ai.nizo.memory.reflect.ReflectionService(
                memory, store, vectorIndex, embedder, llm,
                Duration.ofMinutes(rc.tickIntervalMinutesOr(60)),
                Duration.ofHours(rc.minEpisodeAgeHoursOr(2)),
                rc.minEpisodesPerTickOr(6),
                rc.maxEpisodesPerBatchOr(40),
                rc.duplicateThresholdOr(0.92));
    }

    /**
     * Fact verifier — pass-through (never null) when disabled so callers can
     * always invoke it without null-checking.
     */
    public static FactVerifier verifierFromConfig(NizoConfig config) {
        NizoConfig.VerifierConfig vc = config.verifier();
        if (vc == null || !vc.enabled()) return FactVerifier.passThrough();
        ModelClient model = ollamaFor(vc.backend(), vc.ollama(), 0.0, 30);
        if (model == null) return FactVerifier.passThrough();
        return new FactVerifier(model, vc.maxFactsToVerify(),
                vc.outdatedPenalty(), vc.unverifiablePenalty());
    }

    // ----- helpers ---------------------------------------------------------

    /**
     * Resolve a {@link ModelClient} for one of the LLM-using sections.
     * Currently only the {@code "ollama"} backend is supported; everything else
     * (including {@code "none"}) yields {@code null}.
     */
    private static ModelClient ollamaFor(String backend,
                                          NizoConfig.OllamaConfig oc,
                                          double defaultTemp,
                                          int defaultTimeout) {
        String b = backend == null ? "ollama" : backend.toLowerCase();
        if (!"ollama".equals(b)) {
            if (!"none".equals(b)) {
                log.warn("Unsupported LLM backend '{}' — treating as none", backend);
            }
            return null;
        }
        NizoConfig.OllamaConfig o = oc == null
                ? new NizoConfig.OllamaConfig(null, null, null, null)
                : oc;
        return new OllamaModelClient(
                o.baseUrlOr("http://localhost:11434"),
                o.modelOr("qwen2.5:14b"),
                o.temperatureOr(defaultTemp),
                Duration.ofSeconds(o.timeoutSecondsOr(defaultTimeout)));
    }
}
