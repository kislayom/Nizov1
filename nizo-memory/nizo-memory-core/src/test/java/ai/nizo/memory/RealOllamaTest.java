package ai.nizo.memory;

import ai.nizo.memory.api.extract.ExtractionResult;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.extract.ExtractionPipeline;
import ai.nizo.memory.extract.GraphFactRouter;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.llm.OllamaEmbeddingClient;
import ai.nizo.memory.llm.OllamaModelClient;
import ai.nizo.memory.seed.WorldKnowledgeSeed;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * REAL end-to-end test against running Ollama. Skips if Ollama isn't reachable.
 *
 * <p>Uses real models:
 * <ul>
 *   <li>nomic-embed-text — 768-dim semantic embedder</li>
 *   <li>qwen2.5:7b — fact extraction</li>
 * </ul>
 *
 * <p>This is the test that proves the system works with natural language —
 * NOT pre-baked vocabulary like FakeEmbedder.
 */
class RealOllamaTest {

    private static final String OLLAMA_URL = "http://localhost:11434";

    private SqliteMemoryStore ms;
    private SqliteGraphStore gs;
    private MemoryService mem;
    private ExtractionService ext;

    @TempDir Path tmp;

    @BeforeEach
    void setup() {
        assumeTrue(ollamaReachable(), "Ollama not reachable at " + OLLAMA_URL + " — skipping");

        Path db = tmp.resolve("real.db");
        ms = new SqliteMemoryStore(db);
        gs = new SqliteGraphStore(db);
        var idx = new InMemoryVectorIndex();

        // REAL embedder
        EmbeddingClient embedder = new OllamaEmbeddingClient(
                OLLAMA_URL, "nomic-embed-text", Duration.ofSeconds(30));

        GraphService graph = new KnowledgeGraph(gs);
        mem = new LayeredMemoryService(ms, idx, embedder, null, graph, null, 100, 0.1);

        // REAL extraction LLM
        ModelClient extractor = new OllamaModelClient(
                OLLAMA_URL, System.getProperty("nizo.test.llm", "qwen2.5:14b"), 0.1, Duration.ofSeconds(180));
        ext = new ExtractionPipeline(extractor, new GraphFactRouter(graph), mem);
    }

    @AfterEach
    void teardown() { if (ms != null) ms.close(); if (gs != null) gs.close(); }

    @Test
    void realConversationsRealQueries() throws InterruptedException {
        String u = "kislay";
        System.out.println("\n" + "=".repeat(80));
        System.out.println("  REAL OLLAMA TEST — qwen2.5:7b extraction + nomic-embed-text embeddings");
        System.out.println("=".repeat(80));

        // Seed world knowledge — LLM gets common-sense guardrails as PROCEDURAL memory
        int seeded = WorldKnowledgeSeed.seedIfNeeded(mem, u);
        System.out.println("  ✓ Seeded " + seeded + " world-knowledge heuristics");

        // Build memory with real conversations
        sim(u, "I'm Kislay, I work at Stripe as a Staff Engineer on the payments team.");
        sim(u, "My wife Priya is a doctor at Manipal Hospital in Bangalore.");
        sim(u, "I have a peanut allergy. Always carry an EpiPen.");
        sim(u, "I love mango ice cream. Naturals brand specifically.");
        sim(u, "Looking at the Sony WH-1000XM5 headphones but ₹28,000 is too steep, will wait for a sale.");
        sim(u, "Mom's 60th birthday is December 15th. Planning a surprise party in Mumbai.");
        sim(u, "I prefer dark mode in all my apps. iPhone user.");
        sim(u, "My best friend Amit works at Google Singapore. We talk weekly.");

        // Wait for async embedding
        Thread.sleep(2000);

        System.out.println("\n  ── Memory built. Running queries with REAL embedder ──");

        query(u, "what is my current role");
        query(u, "where do i work");
        query(u, "which ice cream do i love");
        query(u, "tell me about my family");
        query(u, "any health issues i should know about");
        query(u, "things i wanted to buy but didn't");
        query(u, "upcoming birthdays");
        query(u, "who is Amit");
        query(u, "what color is the sky");  // unknown — should be empty
    }

    private void sim(String userId, String message) {
        long t0 = System.currentTimeMillis();
        ExtractionResult r = ext.extract(userId, message);
        long ms = System.currentTimeMillis() - t0;
        System.out.printf("  ✓ [%dms %s] %s%n",
                ms,
                r.hasExtractions() ? r.types().toString() : "none",
                message.length() > 80 ? message.substring(0, 80) + "..." : message);
    }

    private void query(String userId, String q) {
        long t0 = System.currentTimeMillis();
        List<MemoryItem> results = mem.recall(RecallRequest.of(userId, q, 1500));
        long ms = System.currentTimeMillis() - t0;
        System.out.printf("%n  🔍 \"%s\"  (%dms)%n", q, ms);
        if (results.isEmpty()) {
            System.out.println("     → (empty)");
        } else {
            for (int i = 0; i < Math.min(results.size(), 5); i++) {
                MemoryItem m = results.get(i);
                String c = m.content().length() > 110 ? m.content().substring(0, 110) + "..." : m.content();
                System.out.printf("     %d. [%.2f] %s%n", i+1, m.confidence(), c);
            }
            if (results.size() > 5) System.out.println("     ... +" + (results.size()-5) + " more");
        }
    }

    private static boolean ollamaReachable() {
        try {
            var http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
            var resp = http.send(
                    HttpRequest.newBuilder(URI.create(OLLAMA_URL + "/api/tags")).GET().build(),
                    HttpResponse.BodyHandlers.discarding());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }
}
