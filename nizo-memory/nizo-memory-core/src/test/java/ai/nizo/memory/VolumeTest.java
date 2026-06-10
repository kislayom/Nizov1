package ai.nizo.memory;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Production-scale volume and load tests. These simulate what happens when
 * nizo-memory is deployed for real users over weeks and months:
 *
 * <ul>
 *   <li><strong>Large corpus</strong>: 10,000+ items in a single user's store</li>
 *   <li><strong>Multi-user at scale</strong>: 50 concurrent users, all querying</li>
 *   <li><strong>Write throughput</strong>: sustained high-rate inserts</li>
 *   <li><strong>Recall quality under load</strong>: precision doesn't degrade at scale</li>
 *   <li><strong>Token budget under pressure</strong>: doesn't blow up with huge candidate pools</li>
 *   <li><strong>Consolidation at scale</strong>: handles hundreds of episodic items</li>
 *   <li><strong>Cross-user isolation at scale</strong>: 50 users, verify zero leakage</li>
 * </ul>
 *
 * <p>All tests have explicit latency SLAs. If recall takes &gt; 2 seconds on a
 * corpus of 10K items, something is architecturally wrong.
 */
class VolumeTest {

    private static final List<String> VOCAB = List.of(
            // people
            "sarah", "wife", "rahul", "friend", "boss", "mom", "dad", "colleague",
            // work
            "work", "google", "apple", "amazon", "meta", "engineer", "manager", "project",
            // finance
            "stock", "hdfc", "reliance", "portfolio", "invest", "bank", "mutual",
            // personal
            "birthday", "trip", "japan", "india", "doctor", "dentist", "meeting",
            // preferences
            "prefer", "dark", "mode", "coffee", "tea", "morning", "evening",
            // actions
            "call", "email", "book", "schedule", "remind", "action", "task", "pending",
            // misc
            "important", "urgent", "deadline", "review", "report", "analysis"
    );

    private SqliteMemoryStore store;
    private InMemoryVectorIndex index;
    private FakeEmbedder embedder;

    @TempDir Path tmpDir;

    @BeforeEach
    void setup() {
        store = new SqliteMemoryStore(tmpDir.resolve("volume.db"));
        index = new InMemoryVectorIndex();
        embedder = new FakeEmbedder(VOCAB);
    }

    @AfterEach
    void teardown() { store.close(); }

    // =========================================================================
    //  1. Large corpus — 10,000 items, recall still fast and accurate
    // =========================================================================

    @Nested
    @DisplayName("Large corpus (10K items)")
    class LargeCorpus {

        @Test
        @DisplayName("10,000 items — recall completes under 2 seconds")
        void tenThousandItemsRecallLatency() {
            MemoryService svc = svc(embedder, null, 1000, 0.1);
            String userId = "volume-user";

            // Insert 10,000 items — mix of episodic and semantic
            for (int i = 0; i < 8000; i++) {
                svc.remember(userId, "Conversation message " + i + " about topic " + (i % 50)
                        + " discussing " + VOCAB.get(i % VOCAB.size()), Map.of(), "chat");
            }
            for (int i = 0; i < 2000; i++) {
                svc.learnFact(userId, "Fact " + i + ": user's " + VOCAB.get(i % VOCAB.size())
                        + " related information about " + VOCAB.get((i + 7) % VOCAB.size()),
                        "extracted", 0.5 + (i % 5) * 0.1);
            }

            // Verify corpus size
            Map<MemoryItem.Tier, Long> stats = svc.stats(userId);
            long total = stats.values().stream().mapToLong(Long::longValue).sum();
            assertEquals(10000, total, "Should have 10,000 items");

            // Time a recall query
            long start = System.currentTimeMillis();
            List<MemoryItem> results = svc.recall(RecallRequest.of(userId, "stock invest portfolio hdfc", 500));
            long elapsed = System.currentTimeMillis() - start;

            assertFalse(results.isEmpty(), "Should return results from 10K corpus");
            assertTrue(elapsed < 2000,
                    "Recall on 10K corpus should complete under 2s, took " + elapsed + "ms");

            // Token budget must still be respected
            int totalTokens = results.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 500, "Token budget must hold under scale, got " + totalTokens);
        }

        @Test
        @DisplayName("10,000 items — needle still found at top of results")
        void needleInTenThousandHaystacks() {
            MemoryService svc = svc(embedder, null, 1000, 0.1);
            String userId = "needle-user";

            // Insert 9,999 noise items
            for (int i = 0; i < 9999; i++) {
                svc.remember(userId, "Generic noise message number " + i + " nothing specific here "
                        + "about daily routine and general topics", Map.of(), "chat");
            }

            // Insert the needle — a very specific high-confidence fact
            svc.learnFact(userId,
                    "User's mother's birthday is December 25th, she lives in Mumbai India",
                    "user_stated", 0.95);

            // Query for the needle
            long start = System.currentTimeMillis();
            List<MemoryItem> results = svc.recall(RecallRequest.of(userId,
                    "mom birthday december india", 500));
            long elapsed = System.currentTimeMillis() - start;

            assertFalse(results.isEmpty(), "Should find the needle");
            assertTrue(results.get(0).content().contains("birthday")
                            || results.get(0).content().contains("December"),
                    "Birthday fact should be top result, got: " + results.get(0).content());
            assertTrue(elapsed < 2000, "Needle search took " + elapsed + "ms");
        }

        @Test
        @DisplayName("10,000 items — multiple recalls are consistently fast")
        void multipleRecallsUnderLoad() {
            MemoryService svc = svc(embedder, null, 1000, 0.1);
            String userId = "multi-recall";

            for (int i = 0; i < 10000; i++) {
                svc.learnFact(userId, "Fact " + i + " about " + VOCAB.get(i % VOCAB.size())
                        + " involving " + VOCAB.get((i + 3) % VOCAB.size()),
                        "test", 0.5 + (i % 5) * 0.1);
            }

            // Run 20 different queries, each should be fast
            String[] queries = {
                    "stock portfolio invest", "birthday december mom",
                    "work google engineer", "trip japan schedule",
                    "call dentist friday", "dark mode preference",
                    "meeting boss project", "email colleague report",
                    "coffee morning routine", "bank hdfc mutual",
                    "sarah wife family", "rahul friend college",
                    "deadline urgent review", "amazon apple meta",
                    "doctor appointment health", "book schedule remind",
                    "important task pending", "analysis report review",
                    "tea evening prefer", "action call email"
            };

            long totalMs = 0;
            for (String query : queries) {
                long start = System.currentTimeMillis();
                List<MemoryItem> hits = svc.recall(RecallRequest.of(userId, query, 300));
                totalMs += System.currentTimeMillis() - start;
                assertNotNull(hits);
            }

            double avgMs = totalMs / (double) queries.length;
            assertTrue(avgMs < 1000,
                    "Average recall across 20 queries should be under 1s, avg was " + avgMs + "ms");
        }
    }

    // =========================================================================
    //  2. Multi-user concurrent load
    // =========================================================================

    @Nested
    @DisplayName("Multi-user concurrent load")
    class ConcurrentMultiUser {

        @Test
        @DisplayName("50 users writing and reading concurrently — no crashes, no data loss")
        void fiftyConcurrentUsers() throws InterruptedException {
            MemoryService svc = svc(embedder, null, 1000, 0.1);
            int numUsers = 50;
            int factsPerUser = 20;
            int recallsPerUser = 5;
            ExecutorService pool = Executors.newFixedThreadPool(10);
            CountDownLatch writeLatch = new CountDownLatch(numUsers);
            CountDownLatch readLatch = new CountDownLatch(numUsers);
            AtomicInteger errors = new AtomicInteger();

            // Phase 1: All users write concurrently
            for (int u = 0; u < numUsers; u++) {
                String userId = "user-" + u;
                int userNum = u;
                pool.submit(() -> {
                    try {
                        for (int f = 0; f < factsPerUser; f++) {
                            svc.learnFact(userId,
                                    "User " + userNum + " fact " + f + " about "
                                            + VOCAB.get((userNum + f) % VOCAB.size()),
                                    "test", 0.8);
                        }
                    } catch (Exception e) {
                        errors.incrementAndGet();
                    } finally {
                        writeLatch.countDown();
                    }
                });
            }
            assertTrue(writeLatch.await(30, TimeUnit.SECONDS), "Writes should complete in 30s");
            assertEquals(0, errors.get(), "No errors during concurrent writes");

            // Phase 2: All users read concurrently
            for (int u = 0; u < numUsers; u++) {
                String userId = "user-" + u;
                int userNum = u;
                pool.submit(() -> {
                    try {
                        // Verify stats are correct for this user
                        long count = svc.stats(userId).get(MemoryItem.Tier.SEMANTIC);
                        if (count != factsPerUser) {
                            errors.incrementAndGet();
                        }

                        // Run recalls
                        for (int r = 0; r < recallsPerUser; r++) {
                            List<MemoryItem> hits = svc.recall(RecallRequest.of(userId,
                                    VOCAB.get((userNum + r) % VOCAB.size()), 500));
                            // Verify no cross-user contamination
                            for (MemoryItem m : hits) {
                                if (!userId.equals(m.userId())) {
                                    errors.incrementAndGet();
                                }
                            }
                        }
                    } catch (Exception e) {
                        errors.incrementAndGet();
                    } finally {
                        readLatch.countDown();
                    }
                });
            }
            assertTrue(readLatch.await(30, TimeUnit.SECONDS), "Reads should complete in 30s");
            assertEquals(0, errors.get(), "No errors or cross-user leaks during concurrent reads");

            pool.shutdown();
        }

        @Test
        @DisplayName("10 users, 100 facts each — isolation holds at scale")
        void isolationAtScale() {
            MemoryService svc = svc(embedder, null, 1000, 0.1);
            int numUsers = 10;
            int factsPerUser = 100;

            // Each user gets unique content
            for (int u = 0; u < numUsers; u++) {
                String userId = "scaled-user-" + u;
                for (int f = 0; f < factsPerUser; f++) {
                    svc.learnFact(userId,
                            "User" + u + " specific fact number " + f + " about "
                                    + VOCAB.get((u * 7 + f) % VOCAB.size()),
                            "test", 0.8);
                }
            }

            // Verify each user has exactly their facts
            for (int u = 0; u < numUsers; u++) {
                String userId = "scaled-user-" + u;
                Map<MemoryItem.Tier, Long> stats = svc.stats(userId);
                assertEquals(factsPerUser, stats.get(MemoryItem.Tier.SEMANTIC),
                        userId + " should have exactly " + factsPerUser + " facts");

                // Recall should only return this user's items
                List<MemoryItem> hits = svc.recall(RecallRequest.of(userId, "fact user specific", 2000));
                for (MemoryItem m : hits) {
                    assertEquals(userId, m.userId(),
                            "Recall for " + userId + " returned item owned by " + m.userId());
                    assertTrue(m.content().contains("User" + u),
                            userId + " recall returned wrong user's content: " + m.content());
                }
            }
        }
    }

    // =========================================================================
    //  3. Write throughput
    // =========================================================================

    @Nested
    @DisplayName("Write throughput")
    class WriteThroughput {

        @Test
        @DisplayName("5,000 facts inserted in under 10 seconds")
        void bulkInsertThroughput() {
            MemoryService svc = svc(null, null, 1000, 0.0); // no embedder for raw insert speed
            String userId = "throughput-user";

            long start = System.currentTimeMillis();
            for (int i = 0; i < 5000; i++) {
                svc.learnFact(userId, "Throughput fact " + i + " content data", "bench", 0.8);
            }
            long elapsed = System.currentTimeMillis() - start;

            assertEquals(5000L, svc.stats(userId).get(MemoryItem.Tier.SEMANTIC));
            assertTrue(elapsed < 10000,
                    "5,000 inserts should complete in under 10s, took " + elapsed + "ms");

            double factsPerSecond = 5000.0 / (elapsed / 1000.0);
            // Log for visibility — not a hard assertion, just informational
            System.out.println("Write throughput: " + String.format("%.0f", factsPerSecond) + " facts/sec");
        }

        @Test
        @DisplayName("1,000 remember() calls with embeddings don't block")
        void rememberWithEmbeddings() throws InterruptedException {
            MemoryService svc = svc(embedder, null, 1000, 0.0);
            String userId = "embed-throughput";

            long start = System.currentTimeMillis();
            for (int i = 0; i < 1000; i++) {
                svc.remember(userId, "Episode " + i + " about " + VOCAB.get(i % VOCAB.size()),
                        Map.of(), "chat");
            }
            long elapsed = System.currentTimeMillis() - start;

            // remember() should return immediately; embedding happens async
            assertTrue(elapsed < 5000,
                    "1,000 remember() calls should return quickly, took " + elapsed + "ms");

            // Wait for async embedding to finish
            Thread.sleep(2000);

            // All should be stored
            assertEquals(1000L, svc.stats(userId).get(MemoryItem.Tier.EPISODIC));
        }
    }

    // =========================================================================
    //  4. Recall quality under scale
    // =========================================================================

    @Nested
    @DisplayName("Recall quality at scale")
    class RecallQualityAtScale {

        @Test
        @DisplayName("Specific facts still surface above noise at 5,000 items")
        void precisionAtFiveThousand() {
            MemoryService svc = svc(embedder, null, 1000, 0.1);
            String userId = "precision-user";

            // 4,990 noise facts
            for (int i = 0; i < 4990; i++) {
                svc.learnFact(userId, "Generic fact " + i + " about " + VOCAB.get(i % VOCAB.size())
                        + " and " + VOCAB.get((i + 5) % VOCAB.size()),
                        "extracted", 0.6);
            }

            // 10 high-value specific facts
            svc.learnFact(userId, "User's wife Sarah works as a surgeon at City Hospital", "user_stated", 0.95);
            svc.learnFact(userId, "User works at Google as a principal engineer since 2019", "user_stated", 0.95);
            svc.learnFact(userId, "User holds HDFC Bank stock, portfolio value 15 lakhs", "user_stated", 0.9);
            svc.learnFact(userId, "User's mom birthday is December 25th in Mumbai", "user_stated", 0.95);
            svc.learnFact(userId, "User prefers dark mode and drinks coffee every morning", "user_stated", 0.9);
            svc.learnFact(userId, "User has a pending dentist appointment on Friday", "user_stated", 0.85);
            svc.learnFact(userId, "User is planning a trip to Japan with Sarah in December", "conversation", 0.85);
            svc.learnFact(userId, "User's friend Rahul works at Amazon in Seattle", "conversation", 0.8);
            svc.learnFact(userId, "User has daily standup meeting at 10am with boss", "extracted", 0.8);
            svc.learnFact(userId, "User invests following value investing principles like Buffett", "conversation", 0.75);

            // Query for specific facts — they should surface despite 4,990 noise items
            var sarahHits = svc.recall(RecallRequest.of(userId, "wife sarah hospital surgeon", 500));
            assertTrue(sarahHits.stream().anyMatch(m -> m.content().contains("Sarah") && m.content().contains("surgeon")),
                    "Sarah the surgeon should surface in 5K corpus");

            var stockHits = svc.recall(RecallRequest.of(userId, "hdfc stock portfolio invest bank", 500));
            assertTrue(stockHits.stream().anyMatch(m -> m.content().contains("HDFC")),
                    "HDFC stock fact should surface in 5K corpus");

            var birthdayHits = svc.recall(RecallRequest.of(userId, "mom birthday december mumbai", 500));
            assertTrue(birthdayHits.stream().anyMatch(m -> m.content().contains("birthday")),
                    "Birthday fact should surface in 5K corpus");
        }

        @Test
        @DisplayName("Tag filtering works correctly at scale")
        void tagFilteringAtScale() {
            MemoryService svc = svc(null, null, 1000, 0.0);
            String userId = "tag-scale";

            // 500 actions, 500 preferences, 500 behaviors
            for (int i = 0; i < 500; i++) {
                svc.remember(userId, "Action item " + i + ": do something",
                        Map.of("kind", "action"), "test");
                svc.remember(userId, "Preference " + i + ": likes something",
                        Map.of("kind", "preference"), "test");
                svc.remember(userId, "Behavior " + i + ": does something regularly",
                        Map.of("kind", "behavior"), "test");
            }

            // Filter to actions only — should not return preferences or behaviors
            List<MemoryItem> actions = svc.recall(new RecallRequest(
                    userId, "items", 5000, null, Map.of("kind", "action"), 0.0));
            assertFalse(actions.isEmpty(), "Should return actions");
            for (MemoryItem m : actions) {
                assertEquals("action", m.tags().get("kind"),
                        "All items should have kind=action, got: " + m.tags());
            }
            // Token budget should still limit the number returned even with 500 matches
            int totalTokens = actions.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 5000, "Token budget must hold");
        }
    }

    // =========================================================================
    //  5. Consolidation at scale
    // =========================================================================

    @Nested
    @DisplayName("Consolidation at scale")
    class ConsolidationAtScale {

        @Test
        @DisplayName("Consolidation with 200 episodic items produces semantic facts")
        void consolidateLargeEpisodicCorpus() {
            FakeModelClient summariser = new FakeModelClient(prompt -> {
                // Count the number of events in the prompt
                long eventCount = prompt.lines().filter(l -> l.startsWith("- ")).count();
                // Produce proportional facts
                StringBuilder facts = new StringBuilder();
                for (int i = 0; i < Math.min(eventCount / 5, 10); i++) {
                    facts.append("Consolidated fact ").append(i).append(" from large corpus\n");
                }
                return facts.toString();
            });
            MemoryService svc = svc(null, summariser, 1000, 0.1);
            String userId = "consolidate-scale";

            // Insert 200 episodic items
            for (int i = 0; i < 200; i++) {
                svc.remember(userId, "Episode " + i + ": user discussed " + VOCAB.get(i % VOCAB.size())
                        + " with " + VOCAB.get((i + 3) % VOCAB.size()),
                        Map.of(), "chat");
            }

            assertEquals(200L, svc.stats(userId).get(MemoryItem.Tier.EPISODIC));
            assertEquals(0L, svc.stats(userId).get(MemoryItem.Tier.SEMANTIC));

            // Consolidate
            svc.consolidate(userId);

            long semanticAfter = svc.stats(userId).get(MemoryItem.Tier.SEMANTIC);
            assertTrue(semanticAfter > 0,
                    "Consolidation should produce semantic facts from 200 episodes, got " + semanticAfter);
        }
    }

    // =========================================================================
    //  6. SQLite under pressure
    // =========================================================================

    @Nested
    @DisplayName("Storage pressure")
    class StoragePressure {

        @Test
        @DisplayName("Large content items (10KB each) × 1000 — store and recall work")
        void largeContentItems() {
            MemoryService svc = svc(null, null, 1000, 0.0);
            String userId = "large-content";

            // Each item is ~10KB of text
            String padding = "x".repeat(10_000);
            for (int i = 0; i < 1000; i++) {
                svc.learnFact(userId, "Large fact " + i + " about stock invest: " + padding,
                        "test", 0.8);
            }

            assertEquals(1000L, svc.stats(userId).get(MemoryItem.Tier.SEMANTIC));

            // Recall should still work — token budget will limit what comes back
            List<MemoryItem> hits = svc.recall(RecallRequest.of(userId, "stock invest", 500));
            // With 10KB items, each item is ~2500 tokens, so budget of 500 should return 0
            // or very few items (depending on token counting)
            int totalTokens = hits.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 500, "Token budget must hold even with large items");
        }

        @Test
        @DisplayName("FTS search on large corpus returns results quickly")
        void ftsAtScale() {
            String userId = "fts-scale";

            // Insert 5000 items directly into store
            Instant now = Instant.now();
            for (int i = 0; i < 5000; i++) {
                store.upsert(new MemoryItem(
                        UUID.randomUUID().toString(), userId,
                        MemoryItem.Tier.SEMANTIC,
                        "Fact " + i + " about " + VOCAB.get(i % VOCAB.size()),
                        null, Map.of(), "test", 0.8,
                        now, now, 0, 5));
            }

            long start = System.currentTimeMillis();
            List<MemoryItem> results = store.ftsSearch(userId, "stock invest", 30);
            long elapsed = System.currentTimeMillis() - start;

            assertFalse(results.isEmpty(), "FTS should find results in 5K corpus");
            assertTrue(elapsed < 500, "FTS on 5K items should be under 500ms, took " + elapsed + "ms");
            for (MemoryItem m : results) {
                assertEquals(userId, m.userId(), "FTS results must be user-scoped");
            }
        }

        @Test
        @DisplayName("Database file persists and reopens with large corpus")
        void persistenceAtScale() {
            Path dbPath = tmpDir.resolve("persist-scale.db");
            String userId = "persist-user";

            // Write 2000 items, close store
            try (SqliteMemoryStore s1 = new SqliteMemoryStore(dbPath)) {
                Instant now = Instant.now();
                for (int i = 0; i < 2000; i++) {
                    s1.upsert(new MemoryItem(
                            "id-" + i, userId, MemoryItem.Tier.SEMANTIC,
                            "Persistent fact " + i, null, Map.of(), "test",
                            0.8, now, now, 0, 5));
                }
                assertEquals(2000L, s1.countsByTier(userId).get(MemoryItem.Tier.SEMANTIC));
            }

            // Reopen and verify
            try (SqliteMemoryStore s2 = new SqliteMemoryStore(dbPath)) {
                assertEquals(2000L, s2.countsByTier(userId).get(MemoryItem.Tier.SEMANTIC),
                        "All 2000 items should survive store reopen");

                // FTS should work after reopen
                List<MemoryItem> hits = s2.ftsSearch(userId, "Persistent fact", 10);
                assertFalse(hits.isEmpty(), "FTS should work after reopen");
            }
        }
    }

    // =========================================================================
    //  7. Concurrent reads and writes (mixed workload)
    // =========================================================================

    @Nested
    @DisplayName("Mixed read/write workload")
    class MixedWorkload {

        @Test
        @DisplayName("Writers and readers running simultaneously — no corruption")
        void concurrentReadWrite() throws InterruptedException {
            MemoryService svc = svc(null, null, 1000, 0.0);
            String userId = "mixed-workload";

            // Pre-populate with some data
            for (int i = 0; i < 100; i++) {
                svc.learnFact(userId, "Seed fact " + i + " about " + VOCAB.get(i % VOCAB.size()),
                        "test", 0.8);
            }

            ExecutorService pool = Executors.newFixedThreadPool(6);
            AtomicInteger errors = new AtomicInteger();
            AtomicInteger writes = new AtomicInteger();
            AtomicInteger reads = new AtomicInteger();
            CountDownLatch done = new CountDownLatch(6);

            // 3 writer threads
            for (int w = 0; w < 3; w++) {
                int writerNum = w;
                pool.submit(() -> {
                    try {
                        for (int i = 0; i < 200; i++) {
                            svc.learnFact(userId,
                                    "Writer" + writerNum + " fact " + i + " about "
                                            + VOCAB.get((writerNum * 50 + i) % VOCAB.size()),
                                    "concurrent", 0.7);
                            writes.incrementAndGet();
                        }
                    } catch (Exception e) {
                        errors.incrementAndGet();
                    } finally {
                        done.countDown();
                    }
                });
            }

            // 3 reader threads
            for (int r = 0; r < 3; r++) {
                pool.submit(() -> {
                    try {
                        for (int i = 0; i < 100; i++) {
                            List<MemoryItem> hits = svc.recall(RecallRequest.of(userId,
                                    VOCAB.get(i % VOCAB.size()), 300));
                            for (MemoryItem m : hits) {
                                if (!userId.equals(m.userId())) {
                                    errors.incrementAndGet();
                                }
                            }
                            reads.incrementAndGet();
                        }
                    } catch (Exception e) {
                        errors.incrementAndGet();
                    } finally {
                        done.countDown();
                    }
                });
            }

            assertTrue(done.await(60, TimeUnit.SECONDS), "All threads should complete in 60s");
            assertEquals(0, errors.get(),
                    "No errors during concurrent read/write. Writes=" + writes.get()
                    + " Reads=" + reads.get());
            assertEquals(600, writes.get(), "All 600 writes should complete");
            assertEquals(300, reads.get(), "All 300 reads should complete");

            // Final count should include seed + written items
            long total = svc.stats(userId).get(MemoryItem.Tier.SEMANTIC);
            assertEquals(100 + 600, total, "All items should be persisted");

            pool.shutdown();
        }
    }

    // =========================================================================
    //  8. Edge: empty queries / extreme budgets at scale
    // =========================================================================

    @Nested
    @DisplayName("Edge cases at scale")
    class EdgeCasesAtScale {

        @Test
        @DisplayName("Empty query on large corpus doesn't crash or hang")
        void emptyQueryOnLargeCorpus() {
            MemoryService svc = svc(null, null, 1000, 0.0);
            String userId = "empty-query";

            for (int i = 0; i < 2000; i++) {
                svc.learnFact(userId, "Fact " + i, "test", 0.8);
            }

            long start = System.currentTimeMillis();
            List<MemoryItem> hits = svc.recall(RecallRequest.of(userId, "", 500));
            long elapsed = System.currentTimeMillis() - start;

            assertNotNull(hits);
            assertTrue(elapsed < 2000, "Empty query should not hang, took " + elapsed + "ms");
        }

        @Test
        @DisplayName("Budget=1 on large corpus still returns fast")
        void tinyBudgetLargeCorpus() {
            MemoryService svc = svc(null, null, 1000, 0.0);
            String userId = "tiny-budget";

            for (int i = 0; i < 2000; i++) {
                svc.learnFact(userId, "Fact about stock " + i + " invest portfolio", "test", 0.8);
            }

            long start = System.currentTimeMillis();
            List<MemoryItem> hits = svc.recall(RecallRequest.of(userId, "stock invest", 1));
            long elapsed = System.currentTimeMillis() - start;

            int totalTokens = hits.stream().mapToInt(MemoryItem::tokens).sum();
            assertTrue(totalTokens <= 1, "Budget=1 must be respected even at scale");
            assertTrue(elapsed < 2000, "Tiny budget query should be fast, took " + elapsed + "ms");
        }

        @Test
        @DisplayName("Max budget on large corpus returns many results but caps at 20")
        void maxBudgetLargeCorpus() {
            MemoryService svc = svc(null, null, 1000, 0.0);
            String userId = "max-budget";

            for (int i = 0; i < 2000; i++) {
                svc.learnFact(userId, "Fact about stock " + i, "test", 0.8);
            }

            List<MemoryItem> hits = svc.recall(RecallRequest.of(userId, "stock", 999999));
            assertTrue(hits.size() <= 20, "Recall hard cap is 20 items, got " + hits.size());
        }
    }

    // =========================================================================
    //  Helpers
    // =========================================================================

    private MemoryService svc(FakeEmbedder emb, FakeModelClient summariser,
                              int consolidateEveryN, double confidenceFloor) {
        return new LayeredMemoryService(store, index,
                emb, summariser, consolidateEveryN, confidenceFloor);
    }
}
