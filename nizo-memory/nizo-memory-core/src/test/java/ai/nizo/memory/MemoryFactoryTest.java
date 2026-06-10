package ai.nizo.memory;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.compact.CompactionService;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.verify.FactVerifier;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link MemoryFactory} — verifies that each factory method wires
 * components correctly and handles null/missing dependencies gracefully.
 */
class MemoryFactoryTest {

    // -------------------- local() --------------------

    @Test
    void localReturnsNonNullServiceThatCanRememberAndRecall(@TempDir Path tmp) {
        FakeEmbedder embedder = new FakeEmbedder(List.of("vram", "gpu"));
        FakeModelClient summariser = new FakeModelClient("fact one\nfact two");

        MemoryService svc = MemoryFactory.local(
                tmp.resolve("mem.db"), embedder, summariser, 100, 0.0);

        assertNotNull(svc, "local() must return a non-null MemoryService");

        String id = svc.remember("default", "User has 48 GB of vram", Map.of(), "test");
        assertNotNull(id);
        assertFalse(id.isBlank());

        List<MemoryItem> hits = svc.recall(RecallRequest.of("vram", 400));
        assertFalse(hits.isEmpty(), "recall should find the remembered item");
        assertTrue(hits.get(0).content().contains("vram"));
    }

    @Test
    void localWithNullEmbedderWorksFtsOnly(@TempDir Path tmp) {
        MemoryService svc = MemoryFactory.local(
                tmp.resolve("mem.db"), null, null, 100, 0.0);

        assertNotNull(svc, "local() with null embedder must still return a service");

        svc.remember("default", "User prefers oolong tea", Map.of(), "test");
        List<MemoryItem> hits = svc.recall(RecallRequest.of("oolong tea", 400));
        assertFalse(hits.isEmpty(), "FTS-only recall should still work without embedder");
        assertTrue(hits.get(0).content().contains("oolong"));
    }

    @Test
    void localWithNullSummariserConsolidationSkipsSafely(@TempDir Path tmp) {
        MemoryService svc = MemoryFactory.local(
                tmp.resolve("mem.db"), null, null, 5, 0.0);

        // Insert enough episodes to trigger consolidation threshold
        for (int i = 0; i < 20; i++) {
            svc.remember("default", "episode " + i, Map.of(), "chat");
        }

        // Consolidation should not throw even with null summariser
        assertDoesNotThrow(() -> svc.consolidate("default"),
                "consolidation must skip safely when summariser is null");

        // No semantic facts should have been created
        assertEquals(0L, svc.stats("default").get(MemoryItem.Tier.SEMANTIC),
                "no semantic facts should be created without a summariser");
    }

    // -------------------- onnxEmbedder() --------------------

    @Test
    void onnxEmbedderReturnsNullWhenModelDirDoesNotExist(@TempDir Path tmp) {
        Path nonExistent = tmp.resolve("no-such-model-dir");

        EmbeddingClient result = MemoryFactory.onnxEmbedder(nonExistent, 512);

        assertNull(result,
                "onnxEmbedder should return null when model directory doesn't contain expected files");
    }

    // -------------------- compaction() --------------------

    @Test
    void compactionReturnsServiceWhenModelProvided(@TempDir Path tmp) {
        FakeModelClient model = new FakeModelClient("compacted summary");
        MemoryService memory = MemoryFactory.local(
                tmp.resolve("mem.db"), null, null, 100, 0.0);

        CompactionService result = MemoryFactory.compaction(model, memory);

        assertNotNull(result, "compaction() must return a CompactionService when model is provided");
    }

    @Test
    void compactionReturnsNullWhenModelIsNull(@TempDir Path tmp) {
        MemoryService memory = MemoryFactory.local(
                tmp.resolve("mem.db"), null, null, 100, 0.0);

        CompactionService result = MemoryFactory.compaction(null, memory);

        assertNull(result, "compaction() must return null when model is null");
    }

    // -------------------- verifier() --------------------

    @Test
    void verifierReturnsFactVerifierWhenModelProvided() {
        FakeModelClient model = new FakeModelClient("CONFIRMED");

        FactVerifier result = MemoryFactory.verifier(model, 5, 0.2, 0.7);

        assertNotNull(result, "verifier() must return a FactVerifier when model is provided");

        // Verify it actually verifies (not a pass-through)
        MemoryItem item = new MemoryItem("id-1", "default", MemoryItem.Tier.SEMANTIC,
                "Build uses Maven", null, Map.of(), "test", 0.9,
                java.time.Instant.now(), java.time.Instant.now(), 0, 5);
        List<MemoryItem> verified = result.verify(
                List.of(item), "pom.xml exists, uses Maven", 0.3);
        assertEquals(1, verified.size());
        assertEquals(1, model.invocations.get(),
                "verifier with model should invoke the model for verification");
    }

    @Test
    void verifierReturnsPassThroughWhenModelIsNull() {
        FactVerifier result = MemoryFactory.verifier(null, 5, 0.2, 0.7);

        assertNotNull(result, "verifier() must return a pass-through (not null) when model is null");

        // Verify it is indeed a pass-through: items survive with original confidence
        MemoryItem item = new MemoryItem("id-1", "default", MemoryItem.Tier.SEMANTIC,
                "anything", null, Map.of(), "test", 0.9,
                java.time.Instant.now(), java.time.Instant.now(), 0, 3);
        List<MemoryItem> result2 = result.verify(List.of(item), "context", 0.5);
        assertEquals(1, result2.size());
        assertEquals(0.9, result2.get(0).confidence(), 1e-9,
                "pass-through verifier must not alter confidence");
    }
}
