package ai.nizo.memory.recall;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryTags;
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
 * Behavioural tests for facet-weighted scoring (Wave-1 #2 — OMEGA's
 * {@code _TYPE_WEIGHTS} idea, mapped onto our facet taxonomy).
 *
 * <p>Foundational facets (identity / health / profile / preference) must
 * outrank OTHER / EVENT / ROUTINE items when both hit the same query, and
 * un-faceted items must keep their pre-feature scoring (no regression).
 */
class FacetWeightedScoringTest {

    private static final List<String> VOCAB = List.of(
            "anniversary", "weekend", "trip", "wife",
            "priya", "kids", "thai", "food",
            "doctor", "appointment", "march");

    private SqliteMemoryStore store;
    private InMemoryVectorIndex index;
    private LayeredMemoryService svc;

    @BeforeEach
    void setUp(@TempDir Path tmp) {
        store = new SqliteMemoryStore(tmp.resolve("facet-weighted.db"));
        index = new InMemoryVectorIndex();
        svc = new LayeredMemoryService(store, index, new FakeEmbedder(VOCAB),
                new FakeModelClient("never used"), 999, 0.0);
    }

    @Test
    void identityFactOutranksGenericEventOnSameQuery() {
        // Identity (foundational, weight 2.0×) and EVENT (1.0×) both share
        // the query token "wife" — facet-weighting must put identity first.
        svc.learnFact("u", "User's wife is Priya, a cardiologist", "test", 0.9,
                Map.of(MemoryTags.FACET, MemoryTags.FACET_IDENTITY));
        svc.learnFact("u", "Mentioned wife at the office party last March", "test", 0.9,
                Map.of(MemoryTags.FACET, MemoryTags.FACET_EVENT));

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "wife", 1500));

        assertFalse(hits.isEmpty());
        String firstFacet = hits.get(0).tags().get(MemoryTags.FACET);
        assertEquals(MemoryTags.FACET_IDENTITY, firstFacet,
                "identity must outrank event under facet-weighted scoring; got " + firstFacet);
    }

    @Test
    void healthFactOutranksRoutineOnSameQuery() {
        svc.learnFact("u", "User has severe peanut allergy, EpiPen prescribed", "test", 0.9,
                Map.of(MemoryTags.FACET, MemoryTags.FACET_HEALTH));
        svc.learnFact("u", "User checks pantry for peanut snacks weekly", "test", 0.9,
                Map.of(MemoryTags.FACET, MemoryTags.FACET_ROUTINE));

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "peanut", 1500));

        assertFalse(hits.isEmpty());
        assertEquals(MemoryTags.FACET_HEALTH,
                hits.get(0).tags().get(MemoryTags.FACET),
                "health (safety-critical) must outrank routine under weighted scoring");
    }

    @Test
    void preferenceOutranksOtherOnFoodQuery() {
        svc.learnFact("u", "User prefers Thai food, especially green curry", "test", 0.9,
                Map.of(MemoryTags.FACET, MemoryTags.FACET_PREFERENCE));
        svc.learnFact("u", "Random thai food fact", "test", 0.9,
                Map.of(MemoryTags.FACET, MemoryTags.FACET_OTHER));

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "thai food", 1500));

        assertFalse(hits.isEmpty());
        assertEquals(MemoryTags.FACET_PREFERENCE,
                hits.get(0).tags().get(MemoryTags.FACET),
                "preference must outrank OTHER on a preference-shaped query");
    }

    @Test
    void unfacetedItemsKeepLegacyScoring() {
        // Two items, neither tagged with a facet — the multiplier defaults to
        // 1.0× so ranking depends purely on the base RRF + recency factors.
        // This is the "no regression" contract.
        svc.remember("u", "Old anniversary trip", Map.of(), "raw");
        svc.remember("u", "Recent anniversary plan", Map.of(), "raw");

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "anniversary", 1500));

        assertEquals(2, hits.size(), "both unfaceted items must surface");
        // Order must be deterministic (recency or insertion). We don't assert
        // a specific order — only that the call doesn't crash and both come
        // back, which proves the weight lookup gracefully handled missing tags.
    }

    @Test
    void weightLookupHandlesUnknownFacet() {
        // An item carrying a facet tag that's not in FACET_WEIGHTS must be
        // treated as 1.0× — never crash, never null-pointer.
        svc.learnFact("u", "Custom facet test", "test", 0.9,
                Map.of(MemoryTags.FACET, "nonexistent_custom_facet"));

        List<MemoryItem> hits = svc.recall(RecallRequest.of("u", "custom facet", 1500));

        assertFalse(hits.isEmpty(), "unknown facets must not be filtered out");
    }
}
