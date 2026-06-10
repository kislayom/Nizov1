package ai.nizo.memory.seed;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class WorldKnowledgeSeedTest {

    private SqliteMemoryStore store;
    private MemoryService memory;

    @TempDir Path tmp;

    @BeforeEach
    void setup() {
        store = new SqliteMemoryStore(tmp.resolve("seed.db"));
        var index = new InMemoryVectorIndex();
        memory = new LayeredMemoryService(store, index, new FakeEmbedder(List.of()),
                null, 100, 0.0);
    }

    @AfterEach
    void teardown() { store.close(); }

    @Test
    void seedsDefaultHeuristicsForNewUser() {
        int loaded = WorldKnowledgeSeed.seedIfNeeded(memory, "alice");
        assertTrue(loaded > 20, "Should seed many default heuristics, got " + loaded);
        long procedural = memory.stats("alice").get(MemoryItem.Tier.PROCEDURAL);
        assertTrue(procedural >= loaded, "PROCEDURAL tier should hold the seeded items");
    }

    @Test
    void isIdempotent() {
        int first = WorldKnowledgeSeed.seedIfNeeded(memory, "bob");
        int second = WorldKnowledgeSeed.seedIfNeeded(memory, "bob");
        assertTrue(first > 0);
        assertEquals(0, second, "Re-seeding the same user should be a no-op");
    }

    @Test
    void seedsAreIsolatedPerUser() {
        WorldKnowledgeSeed.seedIfNeeded(memory, "user-a");
        long aBefore = memory.stats("user-a").get(MemoryItem.Tier.PROCEDURAL);
        long bBefore = memory.stats("user-b").get(MemoryItem.Tier.PROCEDURAL);
        assertTrue(aBefore > 0);
        assertEquals(0, bBefore, "Other users must not be seeded");

        WorldKnowledgeSeed.seedIfNeeded(memory, "user-b");
        long bAfter = memory.stats("user-b").get(MemoryItem.Tier.PROCEDURAL);
        assertEquals(aBefore, bAfter, "Each user gets the same set of heuristics");
    }

    @Test
    void newVersionMarkerReseeds() {
        int first = WorldKnowledgeSeed.seedIfNeeded(memory, "carol", "v1");
        int againSameVersion = WorldKnowledgeSeed.seedIfNeeded(memory, "carol", "v1");
        int newVersion = WorldKnowledgeSeed.seedIfNeeded(memory, "carol", "v2");
        assertTrue(first > 0);
        assertEquals(0, againSameVersion);
        assertTrue(newVersion > 0, "Bumping the version should trigger a re-seed");
    }

    @Test
    void uploadBatchWorks() {
        List<WorldKnowledgeSeed.Heuristic> custom = List.of(
                new WorldKnowledgeSeed.Heuristic("Custom rule 1", "test"),
                new WorldKnowledgeSeed.Heuristic("Custom rule 2", "test")
        );
        int loaded = WorldKnowledgeSeed.uploadBatch(memory, "dave", custom);
        assertEquals(2, loaded);
        assertEquals(2, memory.stats("dave").get(MemoryItem.Tier.PROCEDURAL));
    }

    @Test
    void emptyBatchIsNoOp() {
        int loaded = WorldKnowledgeSeed.uploadBatch(memory, "eve", List.of());
        assertEquals(0, loaded);
    }

    @Test
    void nullEntriesInBatchAreSkipped() {
        List<WorldKnowledgeSeed.Heuristic> mixed = new java.util.ArrayList<>();
        mixed.add(new WorldKnowledgeSeed.Heuristic("Keeper", "test"));
        mixed.add(null);
        mixed.add(new WorldKnowledgeSeed.Heuristic("", "test"));
        mixed.add(new WorldKnowledgeSeed.Heuristic("   ", "test"));
        mixed.add(new WorldKnowledgeSeed.Heuristic("Another keeper", "test"));
        int loaded = WorldKnowledgeSeed.uploadBatch(memory, "frank", mixed);
        assertEquals(2, loaded);
    }

    @Test
    void skipPropertyMakesSeedingANoOp() {
        // Pinning the SKIP_PROPERTY contract: when set, seedIfNeeded must
        // return 0 and write nothing — even for a fresh user that would
        // normally trigger a full 326-heuristic upload. Used by benchmarks
        // where forgetUser between items would otherwise re-seed every loop
        // (170 s/item observed in the LongMemEval extraction-on run).
        try {
            System.setProperty(WorldKnowledgeSeed.SKIP_PROPERTY, "true");
            int loaded = WorldKnowledgeSeed.seedIfNeeded(memory, "skipped-user");
            assertEquals(0, loaded,
                    "SKIP_PROPERTY=true must short-circuit seedIfNeeded");
            assertEquals(0L, memory.stats("skipped-user").get(MemoryItem.Tier.PROCEDURAL),
                    "no PROCEDURAL items must be written when seeding is skipped");
        } finally {
            System.clearProperty(WorldKnowledgeSeed.SKIP_PROPERTY);
        }
    }

    @Test
    void skipPropertyResetRestoresNormalSeeding() {
        // Defensive: setting and then clearing the property must restore
        // the normal seeding path. Pins the contract that the property is
        // read at call-time, never cached.
        System.setProperty(WorldKnowledgeSeed.SKIP_PROPERTY, "true");
        assertEquals(0, WorldKnowledgeSeed.seedIfNeeded(memory, "user-x"));

        System.clearProperty(WorldKnowledgeSeed.SKIP_PROPERTY);
        int loaded = WorldKnowledgeSeed.seedIfNeeded(memory, "user-x");
        assertTrue(loaded > 20,
                "after clearing the skip flag the seeder must work normally; got " + loaded);
    }

    @Test
    void seededKnowledgeIsRecallable() {
        WorldKnowledgeSeed.seedIfNeeded(memory, "grace");
        // The bundled heuristics include "iPhone user" guardrails — should be findable
        var results = memory.recall(new RecallRequest(
                "grace", "iPhone user phone preference", 1500,
                Set.of(MemoryItem.Tier.PROCEDURAL), Map.of(), 0.0));
        assertFalse(results.isEmpty(), "Seeded heuristics should be recallable");
        boolean foundIphoneRule = results.stream()
                .anyMatch(m -> m.content().toLowerCase().contains("iphone"));
        assertTrue(foundIphoneRule, "Should find the iPhone guardrail heuristic");
    }
}
