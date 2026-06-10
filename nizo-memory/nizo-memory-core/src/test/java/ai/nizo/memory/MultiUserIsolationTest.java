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
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Multi-user isolation tests. These prove that two users sharing the same
 * backing store and vector index can NEVER see each other's memories.
 *
 * <p>This is a security-critical property: a SaaS deployment with many
 * users on one nizo-memory instance must guarantee zero data leakage.
 *
 * <h3>What we test:</h3>
 * <ul>
 *   <li>Recall isolation: user A's facts are invisible to user B's queries</li>
 *   <li>Stats isolation: user A's counts don't include user B's items</li>
 *   <li>Consolidation isolation: consolidating user A doesn't create facts for user B</li>
 *   <li>Tag-based recall isolation: tag queries are user-scoped</li>
 *   <li>Vector search isolation: similar embeddings across users don't leak</li>
 *   <li>FTS isolation: full-text search is user-scoped</li>
 *   <li>Delete isolation: deleting user A's data doesn't affect user B</li>
 *   <li>GDPR forgetUser: completely purges one user without affecting others</li>
 * </ul>
 */
class MultiUserIsolationTest {

    private static final String USER_ALICE = "alice";
    private static final String USER_BOB = "bob";
    private static final String USER_CAROL = "carol";

    private static final List<String> VOCAB = List.of(
            "work", "google", "apple", "engineer", "manager",
            "stock", "hdfc", "reliance", "invest", "portfolio",
            "wife", "husband", "sarah", "rahul", "mom", "dad",
            "dentist", "doctor", "call", "email", "meeting",
            "dark", "mode", "coffee", "tea", "preference",
            "japan", "trip", "birthday", "december", "friday"
    );

    private SqliteMemoryStore store;
    private InMemoryVectorIndex index;
    private FakeEmbedder embedder;
    private MemoryService memory;

    @TempDir Path tmpDir;

    @BeforeEach
    void setup() {
        store = new SqliteMemoryStore(tmpDir.resolve("multi-user.db"));
        index = new InMemoryVectorIndex();
        embedder = new FakeEmbedder(VOCAB);
        FakeModelClient summariser = new FakeModelClient(prompt -> {
            if (prompt.contains("Alice")) return "Alice works at Google as an engineer";
            if (prompt.contains("Bob")) return "Bob works at Apple as a manager";
            return "No facts extracted";
        });
        memory = new LayeredMemoryService(store, index, embedder, summariser, 100, 0.1);
    }

    @AfterEach
    void teardown() { store.close(); }

    // ===== CORE ISOLATION: recall never leaks across users =====

    @Test
    @DisplayName("Alice's facts are invisible to Bob's recall queries")
    void recallIsUserScoped() {
        // Alice stores personal facts
        memory.learnFact(USER_ALICE, "Alice works at Google as a senior engineer", "conversation", 0.9);
        memory.learnFact(USER_ALICE, "Alice's wife Sarah is a surgeon", "conversation", 0.85);

        // Bob stores completely different facts
        memory.learnFact(USER_BOB, "Bob works at Apple as a product manager", "conversation", 0.9);
        memory.learnFact(USER_BOB, "Bob's friend Rahul is in Delhi", "conversation", 0.85);

        // Alice queries — should only see her own facts
        List<MemoryItem> aliceResults = memory.recall(RecallRequest.of(USER_ALICE, "where do I work", 2000));
        String aliceText = aliceResults.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        assertTrue(aliceText.contains("Google"), "Alice should see her Google fact");
        assertFalse(aliceText.contains("Apple"), "Alice must NOT see Bob's Apple fact");
        assertFalse(aliceText.contains("Bob"), "Alice must NOT see anything about Bob");

        // Bob queries — should only see his own facts
        List<MemoryItem> bobResults = memory.recall(RecallRequest.of(USER_BOB, "where do I work", 2000));
        String bobText = bobResults.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        assertTrue(bobText.contains("Apple"), "Bob should see his Apple fact");
        assertFalse(bobText.contains("Google"), "Bob must NOT see Alice's Google fact");
        assertFalse(bobText.contains("Alice"), "Bob must NOT see anything about Alice");
    }

    @Test
    @DisplayName("Episodic memories are isolated between users")
    void episodicMemoryIsolation() {
        memory.remember(USER_ALICE, "Had a great meeting with my boss today about the promotion",
                Map.of("kind", "message"), "conversation");
        memory.remember(USER_BOB, "Feeling stressed about the project deadline",
                Map.of("kind", "message"), "conversation");

        List<MemoryItem> aliceRecall = memory.recall(RecallRequest.of(USER_ALICE, "meeting boss promotion", 1000));
        assertFalse(aliceRecall.isEmpty(), "Alice should recall her meeting");
        for (MemoryItem m : aliceRecall) {
            assertFalse(m.content().contains("stressed"), "Alice must not see Bob's stress");
            assertEquals(USER_ALICE, m.userId(), "All items must belong to Alice");
        }

        List<MemoryItem> bobRecall = memory.recall(RecallRequest.of(USER_BOB, "meeting boss promotion", 1000));
        // Bob has no meeting memory — should be empty or not contain Alice's
        for (MemoryItem m : bobRecall) {
            assertFalse(m.content().contains("promotion"), "Bob must not see Alice's promotion talk");
            assertEquals(USER_BOB, m.userId(), "All items must belong to Bob");
        }
    }

    // ===== STATS ISOLATION =====

    @Test
    @DisplayName("Per-user stats are accurate and isolated")
    void statsArePerUser() {
        memory.learnFact(USER_ALICE, "Alice fact 1", "test", 0.9);
        memory.learnFact(USER_ALICE, "Alice fact 2", "test", 0.9);
        memory.learnFact(USER_ALICE, "Alice fact 3", "test", 0.9);

        memory.learnFact(USER_BOB, "Bob fact 1", "test", 0.9);

        memory.remember(USER_ALICE, "Alice episodic 1", Map.of(), "test");

        Map<MemoryItem.Tier, Long> aliceStats = memory.stats(USER_ALICE);
        Map<MemoryItem.Tier, Long> bobStats = memory.stats(USER_BOB);

        assertEquals(3L, aliceStats.get(MemoryItem.Tier.SEMANTIC), "Alice has 3 semantic facts");
        assertEquals(1L, aliceStats.get(MemoryItem.Tier.EPISODIC), "Alice has 1 episodic item");

        assertEquals(1L, bobStats.get(MemoryItem.Tier.SEMANTIC), "Bob has 1 semantic fact");
        assertEquals(0L, bobStats.get(MemoryItem.Tier.EPISODIC), "Bob has 0 episodic items");
    }

    // ===== CONSOLIDATION ISOLATION =====

    @Test
    @DisplayName("Consolidating Alice's memories doesn't create facts for Bob")
    void consolidationIsPerUser() {
        // Alice has enough episodic items to trigger consolidation
        for (int i = 0; i < 10; i++) {
            memory.remember(USER_ALICE, "Alice episode " + i + " about work at Google",
                    Map.of(), "conversation");
        }
        // Bob has a few items — not enough to consolidate
        memory.remember(USER_BOB, "Bob episode 1", Map.of(), "conversation");
        memory.remember(USER_BOB, "Bob episode 2", Map.of(), "conversation");

        long aliceSemanticBefore = memory.stats(USER_ALICE).get(MemoryItem.Tier.SEMANTIC);
        long bobSemanticBefore = memory.stats(USER_BOB).get(MemoryItem.Tier.SEMANTIC);

        // Consolidate Alice only
        memory.consolidate(USER_ALICE);

        long aliceSemanticAfter = memory.stats(USER_ALICE).get(MemoryItem.Tier.SEMANTIC);
        long bobSemanticAfter = memory.stats(USER_BOB).get(MemoryItem.Tier.SEMANTIC);

        assertTrue(aliceSemanticAfter > aliceSemanticBefore,
                "Alice should have new semantic facts after consolidation");
        assertEquals(bobSemanticBefore, bobSemanticAfter,
                "Bob's semantic count must not change when Alice consolidates");
    }

    // ===== TAG-BASED RECALL ISOLATION =====

    @Test
    @DisplayName("Tag-based queries are user-scoped")
    void tagBasedRecallIsolation() {
        memory.remember(USER_ALICE, "Call dentist by Friday",
                Map.of("kind", "action"), "conversation");
        memory.remember(USER_ALICE, "Email boss about promotion",
                Map.of("kind", "action"), "conversation");

        memory.remember(USER_BOB, "Book flight to Japan",
                Map.of("kind", "action"), "conversation");

        // Alice asks for her actions
        List<MemoryItem> aliceActions = memory.recall(new RecallRequest(
                USER_ALICE, "pending actions", 2000, null, Map.of("kind", "action"), 0.0));
        assertEquals(2, aliceActions.size(), "Alice should see her 2 actions");
        for (MemoryItem m : aliceActions) {
            assertEquals(USER_ALICE, m.userId());
            assertFalse(m.content().contains("Japan"), "Alice must not see Bob's Japan booking");
        }

        // Bob asks for his actions
        List<MemoryItem> bobActions = memory.recall(new RecallRequest(
                USER_BOB, "pending actions", 2000, null, Map.of("kind", "action"), 0.0));
        assertEquals(1, bobActions.size(), "Bob should see his 1 action");
        assertEquals(USER_BOB, bobActions.get(0).userId());
        assertTrue(bobActions.get(0).content().contains("Japan"));
    }

    // ===== VECTOR SEARCH ISOLATION =====

    @Test
    @DisplayName("Similar embeddings across users don't leak via vector search")
    void vectorSearchIsolation() throws InterruptedException {
        // Both users store facts about the same topic (stocks) with similar embeddings
        memory.learnFact(USER_ALICE, "Alice invests in HDFC Bank stock", "conversation", 0.9);
        memory.learnFact(USER_BOB, "Bob invests in Reliance stock", "conversation", 0.9);

        // Wait for async embeddings
        Thread.sleep(300);

        // Alice queries about stocks — should only see her HDFC fact
        List<MemoryItem> aliceStocks = memory.recall(RecallRequest.of(USER_ALICE, "stock invest portfolio", 1000));
        assertFalse(aliceStocks.isEmpty());
        for (MemoryItem m : aliceStocks) {
            assertEquals(USER_ALICE, m.userId(), "All vector results must belong to Alice");
            assertFalse(m.content().contains("Reliance"), "Alice must not see Bob's Reliance stock");
        }

        // Bob queries about stocks — should only see his Reliance fact
        List<MemoryItem> bobStocks = memory.recall(RecallRequest.of(USER_BOB, "stock invest portfolio", 1000));
        assertFalse(bobStocks.isEmpty());
        for (MemoryItem m : bobStocks) {
            assertEquals(USER_BOB, m.userId(), "All vector results must belong to Bob");
            assertFalse(m.content().contains("HDFC"), "Bob must not see Alice's HDFC stock");
        }
    }

    // ===== FTS ISOLATION =====

    @Test
    @DisplayName("Full-text search is user-scoped")
    void ftsIsolation() {
        memory.learnFact(USER_ALICE, "Alice prefers dark mode in all applications", "test", 0.9);
        memory.learnFact(USER_BOB, "Bob prefers light mode and drinks coffee", "test", 0.9);

        // Direct FTS search via store
        List<MemoryItem> aliceFts = store.ftsSearch(USER_ALICE, "mode preference", 10);
        for (MemoryItem m : aliceFts) {
            assertEquals(USER_ALICE, m.userId(), "FTS results must belong to querying user");
        }

        List<MemoryItem> bobFts = store.ftsSearch(USER_BOB, "mode preference", 10);
        for (MemoryItem m : bobFts) {
            assertEquals(USER_BOB, m.userId(), "FTS results must belong to querying user");
        }
    }

    // ===== VECTOR INDEX PARTITIONING =====

    @Test
    @DisplayName("Vector index is partitioned by user — size() reflects per-user counts")
    void vectorIndexPartitioning() {
        float[] v1 = {1, 0, 0};
        float[] v2 = {0, 1, 0};
        float[] v3 = {0, 0, 1};

        index.add(USER_ALICE, "a1", v1);
        index.add(USER_ALICE, "a2", v2);
        index.add(USER_BOB, "b1", v3);

        assertEquals(3, index.size(), "Global size should be 3");
        assertEquals(2, index.size(USER_ALICE), "Alice should have 2 vectors");
        assertEquals(1, index.size(USER_BOB), "Bob should have 1 vector");
        assertEquals(0, index.size(USER_CAROL), "Carol should have 0 vectors");

        // topK for Alice should not return Bob's vector
        var aliceHits = index.topK(USER_ALICE, v3, 5);
        assertEquals(2, aliceHits.size(), "Alice's topK should return her 2 vectors");
        for (var hit : aliceHits) {
            assertTrue(hit.id().startsWith("a"), "Hit should be Alice's: " + hit.id());
        }

        // topK for Bob should not return Alice's vectors
        var bobHits = index.topK(USER_BOB, v1, 5);
        assertEquals(1, bobHits.size(), "Bob's topK should return his 1 vector");
        assertEquals("b1", bobHits.get(0).id());
    }

    // ===== DELETE / GDPR =====

    @Test
    @DisplayName("Deleting one user's item doesn't affect the other")
    void deleteIsolation() {
        String aliceId = memory.learnFact(USER_ALICE, "Alice secret fact", "test", 0.9);
        String bobId = memory.learnFact(USER_BOB, "Bob secret fact", "test", 0.9);

        store.delete(aliceId);

        // Alice's fact is gone
        assertTrue(store.findById(aliceId).isEmpty());
        // Bob's fact is untouched
        assertTrue(store.findById(bobId).isPresent());
        assertEquals(1L, memory.stats(USER_BOB).get(MemoryItem.Tier.SEMANTIC));
        assertEquals(0L, memory.stats(USER_ALICE).get(MemoryItem.Tier.SEMANTIC));
    }

    @Test
    @DisplayName("GDPR forgetUser purges one user completely, others untouched")
    void forgetUserGdprPurge() {
        // Alice accumulates data
        memory.learnFact(USER_ALICE, "Alice fact 1 about work at Google", "test", 0.9);
        memory.learnFact(USER_ALICE, "Alice fact 2 about wife Sarah", "test", 0.85);
        memory.remember(USER_ALICE, "Alice said something personal", Map.of(), "test");

        // Bob has data too
        memory.learnFact(USER_BOB, "Bob works at Apple", "test", 0.9);

        // Verify both have data
        assertTrue(memory.stats(USER_ALICE).values().stream().mapToLong(Long::longValue).sum() > 0);
        assertTrue(memory.stats(USER_BOB).values().stream().mapToLong(Long::longValue).sum() > 0);

        // GDPR: forget Alice completely
        int deleted = store.deleteAllForUser(USER_ALICE);
        assertTrue(deleted >= 3, "Should delete at least 3 items for Alice, deleted " + deleted);

        // Alice is completely gone
        long aliceTotal = memory.stats(USER_ALICE).values().stream().mapToLong(Long::longValue).sum();
        assertEquals(0, aliceTotal, "Alice should have zero items after forgetUser");

        // Bob is untouched
        assertEquals(1L, memory.stats(USER_BOB).get(MemoryItem.Tier.SEMANTIC),
                "Bob's data must survive Alice's purge");
    }

    // ===== THREE-USER SCENARIO =====

    @Test
    @DisplayName("Three users on same instance — complete isolation")
    void threeUserFullIsolation() {
        memory.learnFact(USER_ALICE, "Alice is a software engineer at Google", "test", 0.9);
        memory.learnFact(USER_BOB, "Bob is a product manager at Apple", "test", 0.9);
        memory.learnFact(USER_CAROL, "Carol is a data scientist at Meta", "test", 0.9);

        memory.remember(USER_ALICE, "Alice needs to call dentist", Map.of("kind", "action"), "test");
        memory.remember(USER_BOB, "Bob needs to email boss", Map.of("kind", "action"), "test");
        memory.remember(USER_CAROL, "Carol needs to book trip to Japan", Map.of("kind", "action"), "test");

        // Each user queries "what do I do?" — should only see their own action
        for (var entry : Map.of(
                USER_ALICE, "dentist",
                USER_BOB, "email",
                USER_CAROL, "Japan").entrySet()) {
            String user = entry.getKey();
            String expected = entry.getValue();

            List<MemoryItem> actions = memory.recall(new RecallRequest(
                    user, "pending actions", 2000, null, Map.of("kind", "action"), 0.0));
            assertEquals(1, actions.size(),
                    user + " should see exactly 1 action, got " + actions.size());
            assertTrue(actions.get(0).content().contains(expected),
                    user + " should see '" + expected + "', got: " + actions.get(0).content());
            assertEquals(user, actions.get(0).userId());
        }

        // Each user queries "where do I work?" — should only see their own company
        for (var entry : Map.of(
                USER_ALICE, "Google",
                USER_BOB, "Apple",
                USER_CAROL, "Meta").entrySet()) {
            String user = entry.getKey();
            String expected = entry.getValue();

            // Use "engineer manager scientist" to bridge to stored role facts.
            // FakeEmbedder needs lexical overlap; real embedder would match "work" semantically.
            List<MemoryItem> work = memory.recall(RecallRequest.of(user, "engineer manager scientist company", 1000));
            assertFalse(work.isEmpty(), user + " should recall work info");
            String all = work.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
            assertTrue(all.contains(expected),
                    user + " should see '" + expected + "' in work query, got:\n" + all);

            // Verify no cross-contamination
            for (String otherCompany : List.of("Google", "Apple", "Meta")) {
                if (!otherCompany.equals(expected)) {
                    assertFalse(all.contains(otherCompany),
                            user + " must NOT see " + otherCompany + " in results");
                }
            }
        }
    }

    // ===== DEFAULT USER BACKWARD COMPATIBILITY =====

    @Test
    @DisplayName("Null userId defaults to 'default' — backward compatible")
    void nullUserIdDefaultsToDefault() {
        // Pass null — should behave like "default"
        memory.learnFact(null, "Fact stored with null userId", "test", 0.9);

        // Should be retrievable under "default"
        List<MemoryItem> results = memory.recall(RecallRequest.of("default", "Fact stored", 1000));
        assertFalse(results.isEmpty(), "Null userId should map to 'default'");
        assertEquals("default", results.get(0).userId());

        // Stats under "default" should include it
        assertEquals(1L, memory.stats("default").get(MemoryItem.Tier.SEMANTIC));
        assertEquals(1L, memory.stats(null).get(MemoryItem.Tier.SEMANTIC));
    }

    // ===== EDGE CASES =====

    @Test
    @DisplayName("Empty user has zero stats — doesn't crash")
    void emptyUserStats() {
        Map<MemoryItem.Tier, Long> stats = memory.stats("nonexistent-user-xyz");
        assertNotNull(stats);
        for (MemoryItem.Tier tier : MemoryItem.Tier.values()) {
            assertEquals(0L, stats.get(tier), "Nonexistent user should have 0 for tier " + tier);
        }
    }

    @Test
    @DisplayName("Consolidating empty user is a no-op")
    void consolidateEmptyUser() {
        assertDoesNotThrow(() -> memory.consolidate("nonexistent-user"));
        assertEquals(0L, memory.stats("nonexistent-user").values().stream()
                .mapToLong(Long::longValue).sum());
    }

    @Test
    @DisplayName("Two users with identical fact text — both stored independently")
    void sameFactDifferentUsers() {
        memory.learnFact(USER_ALICE, "User prefers dark mode", "test", 0.9);
        memory.learnFact(USER_BOB, "User prefers dark mode", "test", 0.9);

        assertEquals(1L, memory.stats(USER_ALICE).get(MemoryItem.Tier.SEMANTIC));
        assertEquals(1L, memory.stats(USER_BOB).get(MemoryItem.Tier.SEMANTIC));

        // Alice's recall returns her copy
        List<MemoryItem> aliceResult = memory.recall(RecallRequest.of(USER_ALICE, "dark mode preference", 500));
        assertFalse(aliceResult.isEmpty());
        assertEquals(USER_ALICE, aliceResult.get(0).userId());

        // Bob's recall returns his copy
        List<MemoryItem> bobResult = memory.recall(RecallRequest.of(USER_BOB, "dark mode preference", 500));
        assertFalse(bobResult.isEmpty());
        assertEquals(USER_BOB, bobResult.get(0).userId());

        // Different item IDs
        assertNotEquals(aliceResult.get(0).id(), bobResult.get(0).id(),
                "Same fact for different users should have different IDs");
    }
}
