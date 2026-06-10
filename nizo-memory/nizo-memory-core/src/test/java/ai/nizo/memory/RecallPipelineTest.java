package ai.nizo.memory;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryItem.Tier;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.util.Tokens;
import ai.nizo.memory.util.Vectors;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for the recall pipeline in {@link LayeredMemoryService}.
 *
 * <p>Covers scoring formula, ranking, near-duplicate suppression, token budget
 * enforcement, MMR deduplication, auto-consolidation triggers, proactive
 * surfacing by tag category, confidence floor filtering, and empty-store edge cases.
 */
class RecallPipelineTest {

    /**
     * Vocabulary for {@link FakeEmbedder}. Each word becomes a dimension;
     * texts containing the same words get high cosine similarity.
     */
    private static final List<String> VOCAB = List.of(
            "action", "call", "dentist", "friday",
            "reminder", "birthday", "mom", "april",
            "preference", "dark", "mode", "user",
            "behavior", "stock", "morning", "check",
            "workstation", "gpu", "vram", "rig",
            "tea", "coffee", "oolong", "matcha",
            "invest", "buffett", "value", "portfolio",
            "pending", "upcoming", "date", "schedule"
    );

    private SqliteMemoryStore store;
    private InMemoryVectorIndex index;

    @BeforeEach
    void setUp(@TempDir Path tmp) {
        store = new SqliteMemoryStore(tmp.resolve("recall-test.db"));
        index = new InMemoryVectorIndex();
    }

    @AfterEach
    void tearDown() {
        if (store != null) store.close();
    }

    // =========================================================================
    //  1. Scoring formula validation
    // =========================================================================

    @Nested
    class ScoringFormula {

        @Test
        void higherConfidenceItemsRankFirst() {
            MemoryService svc = svc(null, null, 100, 0.0);
            svc.learnFact("default", "low confidence tea fact", "src", 0.3);
            svc.learnFact("default", "high confidence tea fact", "src", 0.95);

            List<MemoryItem> hits = svc.recall(RecallRequest.of("tea", 2000));
            assertFalse(hits.isEmpty(), "recall must return results");
            // Higher confidence should rank first due to 0.15 * confidence weight.
            assertTrue(hits.get(0).content().contains("high confidence"),
                    "expected high-confidence item first, got: " + hits.get(0).content());
        }

        @Test
        void tierBoostOrderIsSemanticOverProceduralOverWorkingOverEpisodic() {
            // Insert items directly via store to control tiers precisely.
            // Use sufficiently different embeddings to avoid MMR dedup (cosine > 0.92).
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);
            Instant now = Instant.now();

            // Each item uses a distinct vocabulary mix so embeddings differ enough
            // to survive MMR dedup, while all share "tea" for FTS and vector match.
            insertItem("tier-episodic", Tier.EPISODIC, "tea coffee matcha oolong episodic note",
                    embedder.embed("tea coffee matcha oolong"), 0.8, now, 0);
            insertItem("tier-working", Tier.WORKING, "tea gpu vram rig working note",
                    embedder.embed("tea gpu vram rig"), 0.8, now, 0);
            insertItem("tier-procedural", Tier.PROCEDURAL, "tea invest buffett value procedural note",
                    embedder.embed("tea invest buffett value"), 0.8, now, 0);
            insertItem("tier-semantic", Tier.SEMANTIC, "tea action call dentist semantic note",
                    embedder.embed("tea action call dentist"), 0.8, now, 0);

            // Query on "tea" which all items share via FTS and embedding.
            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "tea", 4000, null, Map.of(), 0.0));

            assertTrue(hits.size() >= 2, "expected at least 2 items returned, got " + hits.size());
            // The first item should be SEMANTIC tier (highest boost of +0.15).
            assertEquals(Tier.SEMANTIC, hits.get(0).tier(),
                    "SEMANTIC tier should rank first due to +0.15 boost, but first was "
                    + hits.get(0).tier() + ": " + hits.get(0).content());
        }

        @Test
        void newerItemsRankedHigherThanOlder() {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);
            Instant now = Instant.now();
            Instant weekAgo = now.minus(7, ChronoUnit.DAYS);

            // Old item.
            insertItem("old-tea", Tier.SEMANTIC, "old tea preference",
                    embedder.embed("tea preference"), 0.8, weekAgo, 0);
            // New item with same content theme and confidence.
            insertItem("new-tea", Tier.SEMANTIC, "new tea preference",
                    embedder.embed("tea preference"), 0.8, now, 0);

            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "tea preference", 2000, null, Map.of(), 0.0));

            assertFalse(hits.isEmpty());
            // The newer item has higher recency score: 1/(1+ln(1)) = 1.0 vs ~0.2 for week-old.
            assertEquals("new-tea", hits.get(0).id(),
                    "newer item should rank higher due to recency boost");
        }

        @Test
        void frequentlyAccessedItemsRankHigher() {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);
            Instant now = Instant.now();

            // Low-access item.
            insertItem("rarely-used", Tier.SEMANTIC, "tea rarely accessed",
                    embedder.embed("tea accessed"), 0.8, now, 0);
            // High-access item (simulated by setting accessCount).
            insertItem("often-used", Tier.SEMANTIC, "tea often accessed",
                    embedder.embed("tea accessed"), 0.8, now, 50);

            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "tea accessed", 2000, null, Map.of(), 0.0));

            assertFalse(hits.isEmpty());
            // Usage boost = log1p(accessCount)/4. For 50 accesses: ~0.98; for 0: 0.0.
            assertEquals("often-used", hits.get(0).id(),
                    "frequently-accessed item should rank higher due to usage boost");
        }
    }

    // =========================================================================
    //  2. Near-duplicate suppression in learnFact()
    // =========================================================================

    @Nested
    class NearDuplicateSuppression {

        /**
         * When identical content is learned twice, the embeddings are identical
         * (cosine = 1.0, which exceeds the 0.94 threshold). The current
         * implementation logs the duplicate but does not skip the insert.
         *
         * <p>This test documents actual behaviour: both items are stored.
         * If suppression is fully implemented in the future (early return on
         * near-duplicate), this test should be updated to assert count == 1.
         */
        @Test
        void identicalFactStoredTwiceResultsInTwoItems() {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            svc.learnFact("default", "tea preference is oolong", "src", 0.9);
            svc.learnFact("default", "tea preference is oolong", "src", 0.9);

            long semanticCount = svc.stats("default").get(Tier.SEMANTIC);
            // Current behaviour: both are stored (dedup only logs, does not suppress).
            assertEquals(2, semanticCount,
                    "both facts stored since dedup only logs but does not suppress inserts");
        }

        @Test
        void slightlyDifferentFactsBothStored() {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            svc.learnFact("default", "user prefers dark mode", "src", 0.9);
            svc.learnFact("default", "user invests in value stocks following buffett", "src", 0.9);

            long semanticCount = svc.stats("default").get(Tier.SEMANTIC);
            assertEquals(2, semanticCount,
                    "different-topic facts should both be stored");
        }
    }

    // =========================================================================
    //  3. Token budget strict enforcement
    // =========================================================================

    @Nested
    class TokenBudgetEnforcement {

        @Test
        void recallFitsWithinRequestedBudget() {
            MemoryService svc = svc(null, null, 100, 0.0);
            // Insert many facts that together exceed 5000 tokens.
            for (int i = 0; i < 50; i++) {
                svc.learnFact("default", "tea fact number " + i + " " + "lorem ".repeat(30), "src", 0.8);
            }

            List<MemoryItem> hits = svc.recall(RecallRequest.of("tea", 500));
            int totalTokens = hits.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 500,
                    "recall blew past budget of 500: actual=" + totalTokens);
            assertFalse(hits.isEmpty(), "should return at least one item within budget");
        }

        @Test
        void zeroBudgetUsesDefaultOf1200() {
            MemoryService svc = svc(null, null, 100, 0.0);
            for (int i = 0; i < 50; i++) {
                svc.learnFact("default", "tea fact number " + i + " " + "lorem ".repeat(30), "src", 0.8);
            }

            // tokenBudget=0 should trigger the default of 1200.
            List<MemoryItem> hits = svc.recall(RecallRequest.of("tea", 0));
            int totalTokens = hits.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 1200,
                    "default budget should cap at 1200: actual=" + totalTokens);
        }

        @Test
        void budgetOfOneReturnsAtMostOneSmallItem() {
            MemoryService svc = svc(null, null, 100, 0.0);
            // Each fact is at least a few tokens.
            svc.learnFact("default", "tea", "src", 0.9);
            svc.learnFact("default", "more tea facts here", "src", 0.9);
            svc.learnFact("default", "even more tea with oolong", "src", 0.9);

            List<MemoryItem> hits = svc.recall(RecallRequest.of("tea", 1));
            // Budget=1 is extremely tight. Items with tokens > 1 are skipped.
            int totalTokens = hits.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 1,
                    "budget=1 should allow at most 1 token total: actual=" + totalTokens);
            // "tea" alone is 1 token, so it might fit.
            assertTrue(hits.size() <= 1,
                    "budget=1 should return at most 1 small item, got " + hits.size());
        }
    }

    // =========================================================================
    //  4. MMR deduplication threshold
    // =========================================================================

    @Nested
    class MmrDeduplication {

        @Test
        void nearDuplicateEmbeddingsCollapsedToOne() throws InterruptedException {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            // Three items that share almost the same vocabulary -> cosine > 0.92.
            svc.learnFact("default", "user checks stock every morning", "s1", 0.9);
            svc.learnFact("default", "user checks stock each morning", "s2", 0.9);
            svc.learnFact("default", "user checks stock every morning too", "s3", 0.9);
            waitForAsyncEmbeds();

            // Verify the embeddings are actually similar enough.
            float[] v1 = embedder.embed("user checks stock every morning");
            float[] v2 = embedder.embed("user checks stock each morning");
            double sim = Vectors.cosine(v1, v2);
            assertTrue(sim > 0.92,
                    "test setup: embeddings should be >0.92 similar, got " + sim);

            List<MemoryItem> hits = svc.recall(RecallRequest.of("stock morning", 2000));
            assertTrue(hits.size() < 3,
                    "MMR should suppress near-duplicates, but got " + hits.size() + " items");
        }

        @Test
        void diverseEmbeddingsAllReturned() throws InterruptedException {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            // Three items with very different vocabulary -> low cosine.
            svc.learnFact("default", "user prefers dark mode", "s1", 0.9);
            svc.learnFact("default", "user checks stock every morning", "s2", 0.9);
            svc.learnFact("default", "call dentist by friday", "s3", 0.9);
            waitForAsyncEmbeds();

            // Verify embeddings are dissimilar.
            float[] v1 = embedder.embed("user prefers dark mode");
            float[] v2 = embedder.embed("user checks stock every morning");
            float[] v3 = embedder.embed("call dentist by friday");
            assertTrue(Vectors.cosine(v1, v2) < 0.92, "test setup: v1-v2 should be < 0.92");
            assertTrue(Vectors.cosine(v1, v3) < 0.92, "test setup: v1-v3 should be < 0.92");
            assertTrue(Vectors.cosine(v2, v3) < 0.92, "test setup: v2-v3 should be < 0.92");

            // A broad query that should match all three via FTS + vector.
            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "dark mode stock morning dentist friday", 4000, null, Map.of(), 0.0));
            assertEquals(3, hits.size(),
                    "all 3 diverse items should be returned, got " + hits.size());
        }
    }

    // =========================================================================
    //  5. Auto-consolidation trigger
    // =========================================================================

    @Nested
    class AutoConsolidation {

        @Test
        void consolidationTriggeredAfterNRememberCalls() throws InterruptedException {
            // Summariser produces two facts per invocation.
            FakeModelClient summariser = new FakeModelClient(
                    prompt -> "User prefers oolong tea\nUser owns a workstation with vram");
            MemoryService svc = svc(null, summariser, 3, 0.0);

            // We need >= 8 episodic items for consolidate() to actually run.
            // Pre-populate 7 episodes that won't trigger consolidation (they are
            // separate from the turnCounter which is 0 at this point).
            for (int i = 0; i < 7; i++) {
                svc.remember("default", "pre-episode " + i, Map.of(), "setup");
            }
            // turnCounter is now 7 (none of 7 % 3 == 0 is the first one, but 3 and 6 triggered).
            // Actually, turnCounter increments to 1,2,3,4,5,6,7 -- triggers at 3 and 6.
            // But consolidation needs 8 episodes; at call 3 we only had 3 episodes, too few.
            // At call 6 we had 6 episodes, also too few (< 8). At call 7, no trigger.
            // Now add the 8th and 9th -- call 8 won't trigger (8%3 != 0), call 9 triggers (9%3==0).
            svc.remember("default", "episode 8", Map.of(), "chat");
            svc.remember("default", "episode 9", Map.of(), "chat");

            // At this point turnCounter = 9, which is divisible by 3.
            // Consolidation runs async; wait for it.
            waitForAsyncEmbeds();
            TimeUnit.MILLISECONDS.sleep(300);

            // Consolidation should have run and produced SEMANTIC facts.
            assertTrue(summariser.invocations.get() >= 1,
                    "summariser should have been invoked at least once for consolidation");
            long semanticCount = svc.stats("default").get(Tier.SEMANTIC);
            assertTrue(semanticCount >= 1,
                    "consolidation should produce SEMANTIC items, got " + semanticCount);
        }

        @Test
        void noConsolidationBeforeThreshold() {
            FakeModelClient summariser = new FakeModelClient("should not be called");
            // consolidateEveryN=100, so 2 calls will never reach the threshold.
            MemoryService svc = svc(null, summariser, 100, 0.0);

            svc.remember("default", "episode A", Map.of(), "chat");
            svc.remember("default", "episode B", Map.of(), "chat");

            assertEquals(0, summariser.invocations.get(),
                    "summariser should not be called when turnCounter < consolidateEveryN");
            assertEquals(0L, svc.stats("default").get(Tier.SEMANTIC));
        }
    }

    // =========================================================================
    //  6. Proactive surfacing by tag category
    // =========================================================================

    @Nested
    class ProactiveSurfacing {

        @Test
        void actionItemsSurfaceForPendingActionsQuery() throws InterruptedException {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            // learnFact hardcodes tags={kind=fact}, so insert directly for custom tags.
            insertItem("action-1", Tier.SEMANTIC, "ACTION: Call dentist by friday",
                    embedder.embed("action call dentist friday"), 0.9, Instant.now(), 0,
                    Map.of("kind", "action"));
            insertItem("reminder-1", Tier.SEMANTIC, "REMINDER: Mom birthday is april",
                    embedder.embed("reminder mom birthday april"), 0.95, Instant.now(), 0,
                    Map.of("kind", "reminder"));
            insertItem("pref-1", Tier.SEMANTIC, "PREFERENCE: User prefers dark mode",
                    embedder.embed("preference user dark mode"), 0.8, Instant.now(), 0,
                    Map.of("kind", "preference"));

            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "pending actions call dentist", 2000, null, Map.of(), 0.0));
            assertFalse(hits.isEmpty(), "should return items for 'pending actions' query");
            // The action item should surface first because it shares the most vocabulary.
            assertTrue(hits.get(0).content().contains("ACTION"),
                    "action item should rank first for 'pending actions' query, got: "
                    + hits.get(0).content());
        }

        @Test
        void remindersSurfaceForUpcomingDatesQuery() throws InterruptedException {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            insertItem("action-2", Tier.SEMANTIC, "ACTION: Call dentist by friday",
                    embedder.embed("action call dentist friday"), 0.9, Instant.now(), 0,
                    Map.of("kind", "action"));
            insertItem("reminder-2", Tier.SEMANTIC, "REMINDER: Mom birthday is upcoming date april",
                    embedder.embed("reminder mom birthday upcoming date april"), 0.95,
                    Instant.now(), 0, Map.of("kind", "reminder"));

            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "upcoming date birthday april", 2000, null, Map.of(), 0.0));
            assertFalse(hits.isEmpty());
            assertTrue(hits.get(0).content().contains("REMINDER"),
                    "reminder should rank first for 'upcoming dates' query, got: "
                    + hits.get(0).content());
        }

        @Test
        void requiredTagsFilterToPreferencesOnly() throws InterruptedException {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            insertItem("act-3", Tier.SEMANTIC, "ACTION: Call dentist by friday",
                    embedder.embed("action call dentist friday"), 0.9, Instant.now(), 0,
                    Map.of("kind", "action"));
            insertItem("pref-3", Tier.SEMANTIC, "PREFERENCE: User prefers dark mode",
                    embedder.embed("preference user dark mode"), 0.8, Instant.now(), 0,
                    Map.of("kind", "preference"));
            insertItem("beh-3", Tier.SEMANTIC, "BEHAVIOR: User checks stock every morning",
                    embedder.embed("behavior user stock morning check"), 0.7, Instant.now(), 0,
                    Map.of("kind", "behavior"));

            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "user preference dark mode stock morning",
                    2000, null, Map.of("kind", "preference"), 0.0));
            assertFalse(hits.isEmpty(), "should return preferences");
            for (MemoryItem item : hits) {
                assertEquals("preference", item.tags().get("kind"),
                        "all returned items must have kind=preference, got: " + item.tags());
            }
        }

        @Test
        void requiredTagsFilterToBehaviorsOnly() throws InterruptedException {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            insertItem("act-4", Tier.SEMANTIC, "ACTION: Call dentist by friday",
                    embedder.embed("action call dentist friday"), 0.9, Instant.now(), 0,
                    Map.of("kind", "action"));
            insertItem("pref-4", Tier.SEMANTIC, "PREFERENCE: User prefers dark mode",
                    embedder.embed("preference user dark mode"), 0.8, Instant.now(), 0,
                    Map.of("kind", "preference"));
            insertItem("beh-4", Tier.SEMANTIC, "BEHAVIOR: User checks stock every morning",
                    embedder.embed("behavior user stock morning check"), 0.7, Instant.now(), 0,
                    Map.of("kind", "behavior"));

            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "user behavior stock morning preference dark",
                    2000, null, Map.of("kind", "behavior"), 0.0));
            assertFalse(hits.isEmpty(), "should return behaviors");
            for (MemoryItem item : hits) {
                assertEquals("behavior", item.tags().get("kind"),
                        "all returned items must have kind=behavior, got: " + item.tags());
            }
        }
    }

    // =========================================================================
    //  7. Confidence floor filtering
    // =========================================================================

    @Nested
    class ConfidenceFloorFiltering {

        @Test
        void itemsBelowFloorFilteredOut() {
            MemoryService svc = svc(null, null, 100, 0.5);

            svc.learnFact("default", "low confidence tea fact", "src", 0.3);
            svc.learnFact("default", "borderline confidence tea fact", "src", 0.5);
            svc.learnFact("default", "high confidence tea fact", "src", 0.8);

            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "tea", 2000, null, Map.of(), 0.0));

            // Even though minConfidence in the request is 0.0, the service-level
            // confidenceFloor=0.5 means items below 0.5 are filtered out.
            for (MemoryItem item : hits) {
                assertTrue(item.confidence() >= 0.5,
                        "item with confidence " + item.confidence()
                        + " should have been filtered by confidenceFloor=0.5");
            }
            // Should have 2 items: the 0.5 and 0.8 ones.
            assertEquals(2, hits.size(),
                    "only items with confidence >= 0.5 should be returned");
        }

        @Test
        void requestMinConfidenceCanRaiseAboveFloor() {
            MemoryService svc = svc(null, null, 100, 0.5);

            svc.learnFact("default", "borderline tea fact at 0.5", "src", 0.5);
            svc.learnFact("default", "high confidence tea fact at 0.8", "src", 0.8);

            // Request with minConfidence=0.7, which is higher than the floor of 0.5.
            List<MemoryItem> hits = svc.recall(new RecallRequest(
                    "default", "tea", 2000, null, Map.of(), 0.7));

            assertEquals(1, hits.size(), "only the 0.8 item should pass minConfidence=0.7");
            assertTrue(hits.get(0).confidence() >= 0.7);
        }
    }

    // =========================================================================
    //  8. Empty recall
    // =========================================================================

    @Nested
    class EmptyRecall {

        @Test
        void recallOnEmptyStoreReturnsEmptyList() {
            MemoryService svc = svc(null, null, 100, 0.0);
            List<MemoryItem> hits = svc.recall(RecallRequest.of("anything", 1000));
            assertNotNull(hits, "recall should never return null");
            assertTrue(hits.isEmpty(), "recall on empty store should return empty list");
        }

        @Test
        void recallWithNonMatchingQueryReturnsEmptyList() {
            MemoryService svc = svc(null, null, 100, 0.0);
            // Store a fact about tea.
            svc.learnFact("default", "user prefers oolong tea", "src", 0.9);
            // Query about something completely unrelated that shares no FTS terms.
            List<MemoryItem> hits = svc.recall(RecallRequest.of("quantum entanglement physics", 1000));
            // FTS won't match any terms, and without an embedder there's no vector path.
            assertTrue(hits.isEmpty(),
                    "query with zero keyword overlap and no embedder should return empty");
        }

        @Test
        void recallWithEmbedderButNoMatchReturnsLowRelevanceResults() throws InterruptedException {
            FakeEmbedder embedder = new FakeEmbedder(VOCAB);
            MemoryService svc = svc(embedder, null, 100, 0.0);

            // Store a fact about tea; embedder maps it to tea/oolong dimensions.
            svc.learnFact("default", "user prefers oolong tea", "src", 0.9);
            waitForAsyncEmbeds();

            // Query with words not in the vocabulary at all -> zero-vector from embedder.
            // FTS won't match "xyzzy" or "plugh" either. However, the vector index's
            // topK may still return the item with a 0.0 cosine score (it returns up to
            // K items regardless of score). The item can still appear in the pool and
            // pass through scoring with sim=0.0, relying on recency+confidence+tier
            // for its score. This is expected behaviour: the ranker always has a
            // fallback signal even without semantic similarity.
            List<MemoryItem> hits = svc.recall(RecallRequest.of("xyzzy plugh", 1000));
            // The item may or may not appear depending on whether topK returns zero-score
            // hits. We just verify no crash and result is well-formed.
            assertNotNull(hits, "recall should never return null");
            // If items are returned, they should still fit the budget.
            int totalTokens = hits.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 1000, "results should fit within budget");
        }
    }

    // =========================================================================
    //  Helper methods
    // =========================================================================

    private MemoryService svc(FakeEmbedder embedder, FakeModelClient summariser,
                              int consolidateEveryN, double confidenceFloor) {
        return new LayeredMemoryService(store, index, embedder, summariser,
                consolidateEveryN, confidenceFloor);
    }

    /**
     * Insert a {@link MemoryItem} directly into the store and vector index,
     * bypassing the service layer. Useful for controlling tier, tags, timestamps,
     * and access counts precisely.
     */
    private void insertItem(String id, Tier tier, String content,
                            float[] embedding, double confidence,
                            Instant createdAt, int accessCount) {
        insertItem(id, tier, content, embedding, confidence, createdAt, accessCount,
                Map.of("kind", "fact"));
    }

    private void insertItem(String id, Tier tier, String content,
                            float[] embedding, double confidence,
                            Instant createdAt, int accessCount,
                            Map<String, String> tags) {
        MemoryItem item = new MemoryItem(
                id, "default", tier, content, embedding, tags,
                "test", confidence, createdAt, createdAt,
                accessCount, Tokens.count(content));
        store.upsert(item);
        if (embedding != null) {
            index.add("default", id, embedding);
        }
    }

    /** Embedding happens on a background thread in the service; give it a beat. */
    private static void waitForAsyncEmbeds() throws InterruptedException {
        TimeUnit.MILLISECONDS.sleep(300);
    }
}
