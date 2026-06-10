package ai.nizo.memory.recall;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Behavioural tests for the strong-signal short-circuit (OMEGA Phase 2.5 port)
 * inside {@link LayeredMemoryService#recall}.
 *
 * <p>The short-circuit fires when the top FTS hit covers ≥ 70 % of the query's
 * meaningful tokens AND beats the runner-up by ≥ 20 percentage points. When it
 * fires, the vector and graph channels are suppressed; the result must still
 * include the dominant FTS hit at the top.
 *
 * <p>These tests use a {@link CountingEmbedder} so we can assert the vector
 * channel was actually skipped, not just visually inspect output.
 */
class StrongSignalShortCircuitTest {

    private static final List<String> VOCAB = List.of(
            "wife", "priya", "doctor", "cardiologist",
            "weekend", "trip", "sydney", "kids",
            "allergy", "peanut", "favourite", "thai");

    private SqliteMemoryStore store;
    private InMemoryVectorIndex index;
    private CountingEmbedder embedder;
    private LayeredMemoryService svc;

    @BeforeEach
    void setUp(@TempDir Path tmp) {
        store = new SqliteMemoryStore(tmp.resolve("strong-sig.db"));
        index = new InMemoryVectorIndex();
        embedder = new CountingEmbedder(VOCAB);
        svc = new LayeredMemoryService(store, index, embedder,
                new FakeModelClient("never used"), 999, 0.0);
    }

    @Test
    void firesWhenTopFtsHitDominates() {
        // Top-1 covers ALL query tokens; distractors share at most ONE token,
        // so FTS returns multiple hits but with a clear leader.
        svc.learnFact("u", "User's wife is Priya, a cardiologist", "test", 0.9);
        svc.learnFact("u", "Wife mentioned in passing", "test", 0.9);                 // shares "wife"
        svc.learnFact("u", "Priya called yesterday about the trip", "test", 0.9);     // shares "priya"
        svc.learnFact("u", "Cardiologist appointment was useful", "test", 0.9);       // shares "cardi"

        int embedCallsBefore = embedder.callCount();
        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "wife priya cardiologist", 1500));

        // Strong-signal must have fired → query embedding should NOT have been generated.
        assertEquals(embedCallsBefore, embedder.callCount(),
                "vector channel must be suppressed when strong-signal fires");
        assertFalse(hits.isEmpty(), "the dominant FTS hit must still surface");
        assertTrue(hits.get(0).content().toLowerCase().contains("cardiologist")
                        && hits.get(0).content().toLowerCase().contains("priya"),
                "the dominant hit (covering all query tokens) must rank first; got: "
                        + hits.get(0).content());
    }

    @Test
    void doesNotFireWhenTwoCloseCompetitors() {
        // Two facts cover the same query tokens equally — no dominant hit.
        svc.learnFact("u", "I have peanut allergy at home", "test", 0.9);
        svc.learnFact("u", "Severe peanut allergy, EpiPen prescribed", "test", 0.9);
        svc.learnFact("u", "Trip to Sydney next weekend", "test", 0.9);

        int embedCallsBefore = embedder.callCount();
        svc.recall(RecallRequest.of("u", "peanut allergy", 1500));

        assertTrue(embedder.callCount() > embedCallsBefore,
                "vector channel must run when no single FTS hit dominates");
    }

    @Test
    void doesNotFireWhenOnlyOneFtsHit() {
        // Single FTS hit gives no comparison — short-circuit must NOT fire,
        // because we can't measure the "gap" without a runner-up.
        svc.learnFact("u", "User's wife is Priya, a cardiologist", "test", 0.9);

        int embedCallsBefore = embedder.callCount();
        svc.recall(RecallRequest.of("u", "wife priya cardiologist", 1500));

        assertTrue(embedder.callCount() > embedCallsBefore,
                "single-FTS-hit case must still run vector channel — no dominance signal");
    }

    @Test
    void doesNotFireWhenTopHitOnlyPartiallyMatches() {
        // Top-1 covers only ~33% of query tokens — below the 70% bar.
        svc.learnFact("u", "User has kids", "test", 0.9);
        svc.learnFact("u", "User likes thai food", "test", 0.9);

        int embedCallsBefore = embedder.callCount();
        svc.recall(RecallRequest.of("u", "wife priya cardiologist sydney trip", 1500));

        assertTrue(embedder.callCount() > embedCallsBefore,
                "must NOT short-circuit when top hit doesn't cover most query tokens");
    }

    @Test
    void shortCircuitResultStillRespectsTokenBudget() {
        // Even when short-circuiting, downstream phases (token budget, abstention)
        // must still enforce their guarantees.
        for (int i = 0; i < 5; i++) {
            svc.learnFact("u", "User's wife is Priya the cardiologist (mention #" + i + ")",
                    "test", 0.9);
        }
        svc.learnFact("u", "Random low-overlap fact about kids", "test", 0.9);

        // Tiny budget — must trim even on the short-circuit path.
        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "wife priya cardiologist", 30));

        assertFalse(hits.isEmpty(), "should still surface at least one hit");
        int totalTokens = hits.stream().mapToInt(MemoryItem::tokens).sum();
        assertTrue(totalTokens <= 60,  // small slack for header/separator counting
                "token budget must be respected even on short-circuit; got " + totalTokens);
    }

    /**
     * Embedder that records call counts so tests can assert the vector channel
     * was skipped. Delegates to {@link FakeEmbedder} for actual vectors.
     */
    private static final class CountingEmbedder implements ai.nizo.memory.api.model.EmbeddingClient {
        private final FakeEmbedder delegate;
        private int calls = 0;

        CountingEmbedder(List<String> vocab) {
            this.delegate = new FakeEmbedder(vocab);
        }

        @Override public int dimensions() { return delegate.dimensions(); }

        @Override
        public float[] embed(String text) {
            calls++;
            return delegate.embed(text);
        }

        @Override
        public java.util.List<float[]> embedBatch(java.util.List<String> texts) {
            calls += texts.size();
            return delegate.embedBatch(texts);
        }

        int callCount() { return calls; }
    }
}
