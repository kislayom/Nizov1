package ai.nizo.memory.recall;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Constructor;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Behavioural tests for the adaptive-retry pass (Wave-1 #3 — OMEGA Phase 7.5
 * port) inside {@link LayeredMemoryService#recall}.
 *
 * <p>The retry fires only when pass-1 returned an empty result <em>and</em>
 * the query has meaningful tokens. It re-runs once with thresholds relaxed
 * by 0.6×.
 *
 * <p>Since {@code LayeredMemoryService} is {@code final} we instrument the
 * embedder instead: each {@code recallInternal} pass embeds the query exactly
 * once, so {@code embedder.queryEmbedCalls} == number of passes for a single
 * {@code recall()} invocation.
 */
class AdaptiveRetryTest {

    private static final List<String> VOCAB = List.of(
            "anniversary", "weekend", "trip", "wife",
            "priya", "kids", "thai", "food",
            "schedule", "appointment", "tomorrow");

    private SqliteMemoryStore store;
    private InMemoryVectorIndex index;
    private CountingEmbedder embedder;

    @BeforeEach
    void setUp(@TempDir Path tmp) {
        store = new SqliteMemoryStore(tmp.resolve("retry.db"));
        index = new InMemoryVectorIndex();
        embedder = new CountingEmbedder(VOCAB);
    }

    @Test
    void doesNotRetryWhenLegacyTestThresholdIsZero() {
        // svc with minTopScore = 0.0 is the test default. Adaptive retry must
        // be a no-op here so legacy tests aren't affected.
        LayeredMemoryService svc = newSvc(0.0);
        svc.learnFact("u", "User's wife is Priya, a cardiologist", "test", 0.9);
        embedder.resetCounts();

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "wife priya", 1500));

        assertFalse(hits.isEmpty());
        assertEquals(1, embedder.queryEmbedCalls,
                "minTopScore=0 means adaptive retry must be skipped (no double embed)");
    }

    @Test
    void doesNotRetryWhenPassOneSucceeds() {
        LayeredMemoryService svc = newSvc(0.30);
        svc.learnFact("u", "User's wife is Priya, a cardiologist", "test", 0.9);
        embedder.resetCounts();

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "wife priya", 1500));

        assertFalse(hits.isEmpty(), "pass-1 must already have results");
        assertEquals(1, embedder.queryEmbedCalls,
                "adaptive retry must NOT re-embed the query when pass-1 succeeded");
    }

    @Test
    void retryFiresWhenPassOneIsEmpty() {
        // Stored facts share NO tokens with the query — pass-1 returns empty
        // because there are no FTS or vector hits to score. Adaptive retry
        // must fire exactly one extra pass (verified via embed-count = 2),
        // even though pass-2 will also be empty.
        LayeredMemoryService svc = newSvc(0.30);
        svc.learnFact("u", "User has anniversary trip", "test", 0.9);
        embedder.resetCounts();

        // Query token "appointment" appears in vocab but NOT in any stored fact.
        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "appointment tomorrow", 1500));

        assertTrue(hits.isEmpty(),
                "no items match — both passes must abstain honestly");
        assertEquals(2, embedder.queryEmbedCalls,
                "adaptive retry must have fired exactly one extra pass (2 query embeds)");
    }

    @Test
    void retryDoesNotResurrectBelowFloor() {
        // Pass-1 returns hits. Even with high minTopScore, retry should NOT
        // fire because pass-1 wasn't empty — confirms the contract that we
        // only retry on abstention, not on weak-but-present results.
        LayeredMemoryService svc = newSvc(0.30);
        svc.learnFact("u", "User has anniversary trip planned for the weekend", "test", 0.9);
        embedder.resetCounts();

        svc.recall(RecallRequest.of("u", "weekend anniversary trip", 1500));

        assertEquals(1, embedder.queryEmbedCalls,
                "non-empty pass-1 must NOT trigger an extra embed via retry");
    }

    @Test
    void doesNotRetryOnMeaninglessQuery() {
        LayeredMemoryService svc = newSvc(0.30);
        svc.learnFact("u", "User has anniversary trip", "test", 0.9);
        embedder.resetCounts();

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "?", 1500));

        assertTrue(hits.isEmpty(), "meaningless query must abstain entirely");
        assertEquals(0, embedder.queryEmbedCalls,
                "meaningless query short-circuits before any embed; retry must NOT fire");
    }

    /**
     * Construct via reflection because the configurable-minTopScore
     * constructor is the 10-arg one (not exposed in the convenience set we
     * normally use in tests).
     */
    private LayeredMemoryService newSvc(double minTopScore) {
        try {
            Constructor<LayeredMemoryService> ctor = LayeredMemoryService.class.getDeclaredConstructor(
                    SqliteMemoryStore.class,
                    InMemoryVectorIndex.class,
                    EmbeddingClient.class,
                    ai.nizo.memory.api.model.ModelClient.class,
                    ai.nizo.memory.api.graph.GraphService.class,
                    ai.nizo.memory.api.graph.GraphTraversal.class,
                    int.class,
                    double.class,
                    double.class,
                    double.class);
            return ctor.newInstance(store, index, embedder,
                    new FakeModelClient("never used"),
                    null, null,
                    999, 0.0, 0.01, minTopScore);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("LayeredMemoryService constructor signature changed", e);
        }
    }

    /** Embedder that records query-embed calls so retry passes can be counted. */
    private static final class CountingEmbedder implements EmbeddingClient {
        private final FakeEmbedder delegate;
        int queryEmbedCalls = 0;

        CountingEmbedder(List<String> vocab) { this.delegate = new FakeEmbedder(vocab); }

        void resetCounts() { queryEmbedCalls = 0; }

        @Override public int dimensions() { return delegate.dimensions(); }

        @Override
        public float[] embed(String text) {
            queryEmbedCalls++;
            return delegate.embed(text);
        }

        @Override
        public List<float[]> embedBatch(List<String> texts) {
            queryEmbedCalls += texts.size();
            return delegate.embedBatch(texts);
        }
    }
}
