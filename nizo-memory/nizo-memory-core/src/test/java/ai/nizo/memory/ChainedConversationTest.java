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
 * One long chained conversation about a single user, followed by a mix of
 * queries:
 * <ul>
 *   <li>KNOWN — the fact was taught, should be recalled correctly.</li>
 *   <li>UNKNOWN — the fact was NEVER taught; a memory system worth its
 *       salt must return empty (or very low-confidence loose hits), never
 *       fabricated answers.</li>
 *   <li>CHAIN — requires combining multiple stored facts + world corpus
 *       context (e.g. "best bank if i move to Singapore" → recall SG plan
 *       + bank knowledge).</li>
 * </ul>
 *
 * <p>The messages deliberately use world-corpus vocabulary (Tam Brahm,
 * Pongal, Kolam, Tirupati, SIP, PPF, EPF, NPS, EP visa, SG PR, ₹ crore,
 * EpiPen, gluten intolerance, Ladakh altitude, filter kaapi, etc.) so the
 * extraction LLM has to rely on the bundled heuristics for disambiguation.
 */
class ChainedConversationTest {

    private static final String OLLAMA_URL = "http://localhost:11434";
    @TempDir Path tmp;

    @Test
    void chainedConversationProbe() throws Exception {
        assumeTrue(ollamaReachable(), "Ollama not reachable");

        String user = "priya";

        Path db = tmp.resolve("priya.db");
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

        // ─── CONVERSATION (12 turns) ────────────────────────────────────
        List<String> conversation = List.of(
                "I'm Priya Reddy, VP Product at PhonePe in Bangalore. Joined 4 years ago, came over from Paytm where I was PM-II.",
                "Husband Arjun is a cardiologist at Apollo Bannerghatta. We have a 3-year-old daughter, Ira.",
                "Parents live in Chennai — Dad's a retired PSU bank manager, Mom used to teach math at DAV.",
                "We're Tam Brahm. Visit Tirupati every Diwali, and do kolam together during Pongal.",
                "I do an SIP of ₹80,000 a month — Nifty 50 and Next 50 index funds via Zerodha. Small PPF, plus EPF + NPS tier-1.",
                "Severe peanut allergy, always carry an EpiPen. Also mildly gluten-intolerant, avoid wheat when I can.",
                "Training for the Ladakh Marathon in September. Altitude is going to be brutal; need to peak long runs at 20km by August.",
                "Succession, Severance, and The Bear are my recent obsessions. Weekend mornings are filter kaapi, Ira on my lap, and a slow newspaper.",
                "Thinking of applying for Singapore EP through PhonePe's SG office next year. End goal is SG PR for Ira's schooling.",
                "Looking at the Porsche Taycan 4S but ₹1.8 crore is insane in India. Will wait for Tesla's India launch to get a proper EV.",
                "Diwali gift plan — Apple Vision Pro for Arjun (he's been drooling over it), Dior J'adore for Mom, a new Tata Nexon EV for Dad to replace his 2009 Alto.",
                "Got promoted to Senior VP Product last week. Team's doubling from 12 PMs to 24 by Q1."
        );

        System.out.println();
        header("CHAINED CONVERSATION — Priya, VP Product at PhonePe, Bangalore");
        System.out.println();

        long extractionTotal = 0;
        for (int i = 0; i < conversation.size(); i++) {
            String msg = conversation.get(i);
            long t0 = System.currentTimeMillis();
            ExtractionResult r = ext.extract(user, msg);
            long dur = System.currentTimeMillis() - t0;
            extractionTotal += dur;
            String tag = r.hasExtractions() ? r.types().toString() : "[none]";
            System.out.printf("Turn %2d  [%6dms %s]%n", i + 1, dur, tag);
            System.out.println("  " + msg);
        }
        Thread.sleep(2000);
        System.out.println();
        System.out.printf("Extraction avg: %dms across %d turns%n",
                extractionTotal / conversation.size(), conversation.size());
        System.out.println();

        // ─── QUERIES ────────────────────────────────────────────────────
        header("KNOWN — facts explicitly stated in the conversation");
        askAll(mem, user, List.of(
                "what's my current job title",
                "where did i work before PhonePe",
                "who's in my family",
                "any health risks i carry",
                "what festivals or traditions do we observe",
                "how am i investing for the long term",
                "what shows am i watching",
                "what's my morning routine",
                "what's my long-term relocation plan",
                "any upcoming purchases i'm considering",
                "what are my diwali gift ideas",
                "any recent work milestone"
        ));

        header("UNKNOWN — never taught; honest answer is empty / low-signal");
        askAll(mem, user, List.of(
                "what's my blood type",
                "what school does Ira attend",
                "do i have a pet",
                "what's my favorite book",
                "who's my best friend",
                "do i drink or smoke",
                "what's my annual bonus",
                "what car do i currently drive",
                "what's my LinkedIn URL",
                "where was i born",
                "how many siblings do i have",
                "what's my favourite colour"
        ));

        header("CHAIN-REASONING — require combining facts + corpus context");
        askAll(mem, user, List.of(
                "given my relocation plan, what currency should i start saving in",
                "given my health issues, what should i pack for the Ladakh marathon",
                "what kind of gift would i appreciate for my upcoming birthday",
                "am i eligible for tax-deductible retirement contributions this year",
                "should i be worried about my diet during Ramadan",
                "if i want to help Ira apply to IITs later, what should i be saving for now"
        ));

        ms.close();
        gs.close();
    }

    private void askAll(MemoryService mem, String user, List<String> queries) throws Exception {
        for (String q : queries) {
            long t0 = System.currentTimeMillis();
            var hits = mem.recall(RecallRequest.of(user, q, 1500));
            long dur = System.currentTimeMillis() - t0;
            System.out.printf("%n❓ \"%s\"  (%dms, %d hits)%n", q, dur, hits.size());
            if (hits.isEmpty()) {
                System.out.println("   → (empty)");
            } else {
                int j = 1;
                for (MemoryItem m : hits) {
                    if (j > 5) break;
                    String c = m.content().length() > 110 ? m.content().substring(0, 110) + "..." : m.content();
                    System.out.printf("   %d. [%.2f] %s%n", j++, m.confidence(), c);
                }
                if (hits.size() > 5) System.out.println("   ... +" + (hits.size() - 5) + " more");
            }
        }
        System.out.println();
    }

    private static void header(String s) {
        System.out.println("================================================================================");
        System.out.println("  " + s);
        System.out.println("================================================================================");
    }

    private static boolean ollamaReachable() {
        try {
            var http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
            return http.send(HttpRequest.newBuilder(URI.create(OLLAMA_URL + "/api/tags")).GET().build(),
                    HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
        } catch (Exception e) { return false; }
    }
}
