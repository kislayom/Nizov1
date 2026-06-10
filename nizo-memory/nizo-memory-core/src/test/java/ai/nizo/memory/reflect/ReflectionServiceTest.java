package ai.nizo.memory.reflect;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryTags;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * F1 — Reflection worker tests. Uses {@link FakeModelClient} so no Ollama
 * dependency.
 */
class ReflectionServiceTest {

    @Test
    void reflection_producesSemanticFactsFromOlderEpisodes(@TempDir Path tmp) throws Exception {
        Path db = tmp.resolve("mem.db");
        SqliteMemoryStore store = new SqliteMemoryStore(db);
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        FakeEmbedder embedder = new FakeEmbedder(List.of(
                "fact", "bali", "morning", "running", "vegetarian", "vim", "apple", "java", "dog", "pilot"));
        LayeredMemoryService svc = new LayeredMemoryService(
                store, index, embedder, null, null, null,
                999_999, 0.0, 0.01, 0.0);

        String uid = "alice";
        // Seed 8 raw episodes, all dated 3 hours ago so they're "old enough".
        String[] episodes = {
                "Started my morning run at 6am.",
                "Went running for 45 minutes along Marine Drive.",
                "Decided to skip meat this week — trying vegetarian.",
                "Lunch was dal tadka; no meat today.",
                "Switched my editor from VS Code to Vim.",
                "Vim feels much faster for quick edits.",
                "Booked the Bali trip for next month.",
                "Mentioned Bali to my parents; they're excited."
        };
        for (String e : episodes) svc.remember(uid, e, Map.of(), "user");
        // Backdate their created_at so reflection picks them up.
        backdateAll(db, uid, Duration.ofHours(3));

        // Fake LLM produces 3 distilled facts.
        FakeModelClient llm = new FakeModelClient(
                "User runs in the morning along Marine Drive.\n"
                + "User is experimenting with a vegetarian diet.\n"
                + "User switched from VS Code to Vim.\n"
                + "User booked a Bali trip for next month.");

        ReflectionService reflector = new ReflectionService(
                svc, store, index, embedder, llm,
                Duration.ofMinutes(60),
                Duration.ofHours(2),
                6, 40, 0.92);

        int produced = reflector.runOnce();
        reflector.close();

        assertTrue(produced >= 3, "reflection should produce ≥ 3 facts, got " + produced);
        assertEquals(1, llm.invocations.get(),
                "one LLM call per user per tick");

        // The fake reflection call should have stored SEMANTIC facts via learnFact
        List<MemoryItem> found = svc.recall(RecallRequest.of(uid, "morning running", 600));
        boolean hasMorning = found.stream().anyMatch(m ->
                m.content().toLowerCase().contains("morning")
                && m.tier() == MemoryItem.Tier.SEMANTIC
                && "reflection".equals(m.source()));
        assertTrue(hasMorning,
                "a SEMANTIC fact from reflection should surface under 'morning running'");

        // Every consumed episode must be tagged reflected_at.
        for (MemoryItem m : store.olderEpisodes(uid, 0, 20)) {
            assertTrue(m.tags().containsKey(MemoryTags.REFLECTED_AT),
                    "each consumed episode must be marked reflected_at");
        }
    }

    @Test
    void reflection_skipsDuplicateFacts(@TempDir Path tmp) throws Exception {
        Path db = tmp.resolve("mem.db");
        SqliteMemoryStore store = new SqliteMemoryStore(db);
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        // Use a FakeEmbedder seeded with the EXACT tokens used in both the
        // pre-existing fact and the reflection output so they hash to the same
        // dense vector (cosine ≈ 1.0 when token sets match).
        FakeEmbedder embedder = new FakeEmbedder(List.of(
                "user", "runs", "morning", "daily"));
        LayeredMemoryService svc = new LayeredMemoryService(
                store, index, embedder, null, null, null,
                999_999, 0.0, 0.01, 0.0);

        String uid = "bob";
        // Pre-existing SEMANTIC fact
        svc.learnFact(uid, "User runs every morning daily", "user_stated", 0.9);

        // Seed episodes
        for (int i = 0; i < 8; i++) {
            svc.remember(uid, "Morning run at the park today (" + i + ")", Map.of(), "user");
        }
        backdateAll(db, uid, Duration.ofHours(3));

        // LLM produces an identical fact
        FakeModelClient llm = new FakeModelClient("User runs every morning daily");

        ReflectionService reflector = new ReflectionService(
                svc, store, index, embedder, llm,
                Duration.ofMinutes(60), Duration.ofHours(2),
                6, 40, 0.80);  // lower threshold so near-duplicates trip it

        int produced = reflector.runOnce();
        reflector.close();

        assertEquals(0, produced,
                "duplicate fact (cos > threshold) should be suppressed");
    }

    @Test
    void reflection_skipsSmallBatches(@TempDir Path tmp) throws Exception {
        Path db = tmp.resolve("mem.db");
        SqliteMemoryStore store = new SqliteMemoryStore(db);
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        FakeEmbedder embedder = new FakeEmbedder(List.of("x"));
        LayeredMemoryService svc = new LayeredMemoryService(
                store, index, embedder, null, null, null,
                999_999, 0.0, 0.01, 0.0);

        String uid = "carol";
        // Only 3 episodes — below minEpisodesPerTick=6.
        svc.remember(uid, "tiny thing one", Map.of(), "user");
        svc.remember(uid, "tiny thing two", Map.of(), "user");
        svc.remember(uid, "tiny thing three", Map.of(), "user");
        backdateAll(db, uid, Duration.ofHours(3));

        FakeModelClient llm = new FakeModelClient("should not be called");
        ReflectionService reflector = new ReflectionService(
                svc, store, index, embedder, llm,
                Duration.ofMinutes(60), Duration.ofHours(2),
                6, 40, 0.92);

        int produced = reflector.runOnce();
        reflector.close();

        assertEquals(0, produced);
        assertEquals(0, llm.invocations.get(),
                "LLM must not be called when batch < minEpisodesPerTick");
    }

    @Test
    void reflection_doesNotReprocessAlreadyReflectedEpisodes(@TempDir Path tmp) throws Exception {
        Path db = tmp.resolve("mem.db");
        SqliteMemoryStore store = new SqliteMemoryStore(db);
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        FakeEmbedder embedder = new FakeEmbedder(List.of("food", "morning", "run"));
        LayeredMemoryService svc = new LayeredMemoryService(
                store, index, embedder, null, null, null,
                999_999, 0.0, 0.01, 0.0);

        String uid = "dan";
        for (int i = 0; i < 8; i++) svc.remember(uid, "morning run " + i, Map.of(), "user");
        backdateAll(db, uid, Duration.ofHours(3));

        FakeModelClient llm = new FakeModelClient("User runs in the morning.");
        ReflectionService reflector = new ReflectionService(
                svc, store, index, embedder, llm,
                Duration.ofMinutes(60), Duration.ofHours(2),
                6, 40, 0.99);

        int first = reflector.runOnce();
        int second = reflector.runOnce();
        reflector.close();

        assertEquals(1, first, "first tick produces 1 fact");
        assertEquals(0, second, "second tick must not reprocess already-reflected episodes");
        assertEquals(1, llm.invocations.get(), "only one LLM call total");
    }

    /** Backdate all of userId's episodes by {@code age} — so the reflection
     *  worker considers them "old enough" under minEpisodeAgeHours. */
    private static void backdateAll(Path dbPath, String userId, Duration age) throws Exception {
        long cutoff = Instant.now().minus(age.toHours(), ChronoUnit.HOURS).toEpochMilli();
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath.toAbsolutePath());
             PreparedStatement ps = conn.prepareStatement(
                     "UPDATE memory_items SET created_at = ? WHERE user_id = ? AND tier = 'EPISODIC'")) {
            ps.setLong(1, cutoff);
            ps.setString(2, userId);
            ps.executeUpdate();
        }
    }
}
