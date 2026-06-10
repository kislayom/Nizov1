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
 * Nizo-owner perspective: how does the v2 world-knowledge corpus hold up
 * across diverse user personas (geography, profession, culture)?
 *
 * <p>This is not a unit test in the assertEquals sense. It runs three real
 * users through extract + recall against live Ollama and prints everything
 * so a human (the owner) can read the output and judge:
 * <ul>
 *   <li>Did extraction use the corpus correctly? (e.g. M-Pesa = mobile money,
 *       not a job; CDMX = Mexico City; EP = Singapore Employment Pass)</li>
 *   <li>Did recall return the RIGHT facts, not just semantically near ones?</li>
 *   <li>Did "no idea" queries correctly return empty? (relevance floor)</li>
 *   <li>Was the latency acceptable?</li>
 * </ul>
 *
 * <p>Skips automatically if Ollama isn't reachable.
 */
class NizoOwnerScenariosTest {

    private static final String OLLAMA_URL = "http://localhost:11434";

    @TempDir Path tmp;

    @Test
    void diversePersonasAcrossGeographies() throws InterruptedException {
        assumeTrue(ollamaReachable(), "Ollama not reachable — skipping");

        line();
        out("  NIZO OWNER SCENARIOS — does v2 corpus disambiguate across cultures?");
        out("  qwen2.5:7b extraction + nomic-embed-text embeddings");
        line();

        // Each persona gets a fresh DB so context doesn't leak
        runPersona("sofia",
                "Sofia, Mexican fintech founder in Mexico City",
                List.of(
                        "Soy Sofía. I run a small fintech startup in CDMX, building a Pix-style instant-payment app for Mexico.",
                        "We just raised pre-seed at $1.2M from Y Combinator alums. My co-founder Diego handles engineering.",
                        "Mom hosts Día de los Muertos every year, this November will be the first time my baby joins us.",
                        "I have lactose intolerance — no leche, but mezcal con limón is my drink.",
                        "Looking at the new MacBook Air M4 but at $1,499 I'll wait for Black Friday."
                ),
                List.of(
                        "what does sofia do for work",
                        "any food restrictions",
                        "what holidays does she celebrate",
                        "things she wants to buy but is waiting on",
                        "what's the weather in Tokyo today"   // unanswerable — should be empty
                ));

        runPersona("aisha",
                "Aisha, Kenyan UX designer in Nairobi",
                List.of(
                        "I'm Aisha, a senior UX designer at Safaricom in Nairobi. Working on M-Pesa redesign for the next quarter.",
                        "My husband Brian teaches at University of Nairobi, computer science department.",
                        "Saving up for Hajj in 2027 — pilgrimage to Mecca, big milestone for us as Muslims.",
                        "Eliud Kipchoge is my hero. Tried running my first half-marathon last month, took 2hr 14min.",
                        "Considering moving to Cape Town for a remote role at Andela but the salary in ZAR is tricky."
                ),
                List.of(
                        "what is aisha's profession",
                        "religious or spiritual things she does",
                        "any sports she does",
                        "where might she move",
                        "what brand of car does she drive"  // unanswerable — should be empty
                ));

        runPersona("rohan",
                "Rohan, Indian NRI software engineer in Singapore",
                List.of(
                        "I'm Rohan. Moved from Bangalore to Singapore 3 years back on EP, working as a Staff MLE at Grab.",
                        "Got my SG PR last month. Salary just bumped to S$240,000 base + 30% bonus + stock.",
                        "Sister Ananya getting married in Bengaluru next December, big traditional Tam Brahm wedding.",
                        "Doing CFA Level 3 in November while working — switching from ML to quant trading next year.",
                        "Thinking of buying a 2BHK condo in Bishan, but at S$1.4M with ABSD as PR it's a stretch."
                ),
                List.of(
                        "what is rohan's current role",
                        "what's his immigration status",
                        "any upcoming family events",
                        "what is he studying",
                        "what's his favorite movie"  // unanswerable — should be empty
                ));

        line();
        out("  DONE — review the output above as Nizo's owner.");
        out("  Look for: correct extraction categories, recall quality, latency, empty results when nothing matches.");
        line();
    }

    private void runPersona(String userId, String description,
                             List<String> messages, List<String> queries) throws InterruptedException {
        // Fresh stack per persona so context doesn't pollute
        Path db = tmp.resolve(userId + ".db");
        SqliteMemoryStore ms = new SqliteMemoryStore(db);
        SqliteGraphStore gs = new SqliteGraphStore(db);
        InMemoryVectorIndex idx = new InMemoryVectorIndex();
        EmbeddingClient embedder = new OllamaEmbeddingClient(
                OLLAMA_URL, "nomic-embed-text", Duration.ofSeconds(30));
        GraphService graph = new KnowledgeGraph(gs);
        // Production-grade similarity floor (0.45) for nomic-embed-text — same
        // value MemoryFactory.fromConfig() uses. Without it, recall returns
        // semantic noise on unanswerable queries ("weather in Tokyo today").
        MemoryService mem = new LayeredMemoryService(
                ms, idx, embedder, null, graph, null, 100, 0.1, 0.45);
        ModelClient extractor = new OllamaModelClient(
                OLLAMA_URL, System.getProperty("nizo.test.llm", "qwen2.5:14b"), 0.1, Duration.ofSeconds(180));
        ExtractionService ext = new ExtractionPipeline(extractor, new GraphFactRouter(graph), mem);

        try {
            out("");
            out("  ╔══════════════════════════════════════════════════════════════════════╗");
            out("  ║  PERSONA: " + pad(description, 60) + "║");
            out("  ╚══════════════════════════════════════════════════════════════════════╝");

            int seeded = WorldKnowledgeSeed.seedIfNeeded(mem, userId);
            out("  • Seeded " + seeded + " world-knowledge heuristics for " + userId);
            out("");
            out("  ── Building memory with " + messages.size() + " messages ──");

            long extractionTotal = 0;
            for (String message : messages) {
                long t0 = System.currentTimeMillis();
                ExtractionResult r = ext.extract(userId, message);
                long dur = System.currentTimeMillis() - t0;
                extractionTotal += dur;
                String tag = r.hasExtractions() ? r.types().toString() : "[none]";
                String preview = message.length() > 75 ? message.substring(0, 75) + "..." : message;
                System.out.printf("  ✓ [%6dms %s] %s%n", dur, tag, preview);
            }

            // Wait for async embedding before recall
            Thread.sleep(2000);

            out("");
            out("  ── Querying with " + queries.size() + " questions ──");
            for (String q : queries) {
                long t0 = System.currentTimeMillis();
                List<MemoryItem> results = mem.recall(RecallRequest.of(userId, q, 1500));
                long dur = System.currentTimeMillis() - t0;
                System.out.printf("%n  🔍 \"%s\"  (%dms)%n", q, dur);
                if (results.isEmpty()) {
                    out("     → (empty) ✓ correct if question is unanswerable");
                } else {
                    for (int i = 0; i < Math.min(results.size(), 5); i++) {
                        MemoryItem m = results.get(i);
                        String c = m.content().length() > 100 ? m.content().substring(0, 100) + "..." : m.content();
                        System.out.printf("     %d. [%.2f] %s%n", i + 1, m.confidence(), c);
                    }
                    if (results.size() > 5) System.out.println("     ... +" + (results.size() - 5) + " more");
                }
            }

            out("");
            out("  Avg extraction latency: " + (extractionTotal / messages.size()) + "ms");
        } finally {
            ms.close();
            gs.close();
        }
    }

    private static String pad(String s, int n) {
        if (s.length() >= n) return s.substring(0, n);
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < n) sb.append(' ');
        return sb.toString();
    }

    private static void line() { out("=".repeat(80)); }
    private static void out(String s) { System.out.println(s); }

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
