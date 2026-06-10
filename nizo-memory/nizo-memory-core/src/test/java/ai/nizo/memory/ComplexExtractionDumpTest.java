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
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
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
 * Prints raw extraction (JSON-like) and the resulting stored facts + a
 * matching recall, for the most complex multi-category messages. Designed
 * for showing a human the full input/output without editorialising.
 */
class ComplexExtractionDumpTest {

    private static final String OLLAMA_URL = "http://localhost:11434";
    @TempDir Path tmp;

    @Test
    void dumpComplexExtractions() throws Exception {
        assumeTrue(ollamaReachable(), "Ollama not reachable");

        List<Scenario> scenarios = List.of(
                new Scenario("rohan-equity",
                        "I'm Rohan. Moved from Bangalore to Singapore 3 years back on EP, working as a Staff MLE at Grab. Got my SG PR last month. Salary just bumped to S$240,000 base + 30% bonus + 400K stock vesting over 4 years.",
                        List.of("what is rohan's role and compensation",
                                "his immigration status")),
                new Scenario("sofia-fundraise",
                        "We just closed our pre-seed at $1.2M from Y Combinator alums and a couple of Mexican angels. My co-founder Diego handles engineering, I run product and fundraising. Building a Pix-style instant-payment app for Mexico.",
                        List.of("who are Sofia's investors",
                                "what does her co-founder do")),
                new Scenario("aisha-career-move",
                        "Considering a Staff Designer role at Andela in Cape Town — would be $180K in ZAR but with the rand volatility that's tricky. Brian is supportive but worried about moving the kids mid-school-year. Will decide by end of November.",
                        List.of("where is Aisha thinking of moving",
                                "what's the concern about moving"))
        );

        ObjectMapper jsonPretty = new ObjectMapper();
        jsonPretty.enable(SerializationFeature.INDENT_OUTPUT);

        for (Scenario s : scenarios) {
            // Fresh stack per scenario
            Path db = tmp.resolve(s.userId + ".db");
            var ms = new SqliteMemoryStore(db);
            var gs = new SqliteGraphStore(db);
            var idx = new InMemoryVectorIndex();
            EmbeddingClient embedder = new OllamaEmbeddingClient(
                    OLLAMA_URL, "nomic-embed-text", Duration.ofSeconds(30));
            GraphService graph = new KnowledgeGraph(gs);
            MemoryService mem = new LayeredMemoryService(
                    ms, idx, embedder, null, graph, null, 100, 0.1, 0.45);
            ModelClient extractor = new OllamaModelClient(
                    OLLAMA_URL, System.getProperty("nizo.test.llm", "qwen2.5:14b"), 0.1, Duration.ofSeconds(180));
            ExtractionService ext = new ExtractionPipeline(extractor, new GraphFactRouter(graph), mem);

            System.out.println();
            System.out.println("================================================================================");
            System.out.println("SCENARIO: " + s.userId);
            System.out.println("================================================================================");
            System.out.println();
            System.out.println("INPUT MESSAGE:");
            System.out.println(s.message);
            System.out.println();

            long t0 = System.currentTimeMillis();
            ExtractionResult r = ext.extract(s.userId, s.message);
            long dur = System.currentTimeMillis() - t0;

            System.out.println("EXTRACTION RESULT (" + dur + "ms):");
            System.out.println("  categories: " + r.types());
            System.out.println("  count: " + r.count());
            System.out.println();
            System.out.println("RAW EXTRACTED JSON (from LLM, parsed):");
            System.out.println(jsonPretty.writeValueAsString(r.raw()));
            System.out.println();

            // Wait a beat for async embedding
            Thread.sleep(1500);

            System.out.println("STORED FACTS (everything in memory after extraction):");
            var everything = mem.recall(RecallRequest.of(s.userId, s.message, 4000));
            int i = 1;
            for (MemoryItem m : everything) {
                System.out.printf("  %d. [tier=%s conf=%.2f] %s%n",
                        i++, m.tier(), m.confidence(), m.content());
            }
            System.out.println();

            for (String q : s.queries) {
                System.out.println("QUERY: \"" + q + "\"");
                var hits = mem.recall(RecallRequest.of(s.userId, q, 1500));
                if (hits.isEmpty()) {
                    System.out.println("  → (empty)");
                } else {
                    int j = 1;
                    for (MemoryItem m : hits) {
                        if (j > 5) break;
                        System.out.printf("  %d. [%.2f] %s%n", j++, m.confidence(), m.content());
                    }
                }
                System.out.println();
            }

            ms.close();
            gs.close();
        }
    }

    private record Scenario(String userId, String message, List<String> queries) {}

    private static boolean ollamaReachable() {
        try {
            var http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
            return http.send(HttpRequest.newBuilder(URI.create(OLLAMA_URL + "/api/tags")).GET().build(),
                    HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
        } catch (Exception e) { return false; }
    }
}
