package ai.nizo.memory.canonical;

import ai.nizo.memory.api.memory.MemoryTags;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Covers the promotion policy matrix documented on {@link CanonicalPromoter}.
 * Every path that flips an item to canonical is exercised, plus the hard
 * vetoes (hedged / hypothetical) that must never promote regardless of
 * other signals.
 */
class CanonicalPromoterTest {

    private final CanonicalPromoter promoter = new CanonicalPromoter();

    @Test
    void promotesDefinitionalShape() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.FACET, MemoryTags.FACET_IDENTITY);

        Map<String, String> out = promoter.maybePromote("My wife is Priya", tags, 0.9);

        assertEquals("true", out.get(MemoryTags.CANONICAL));
        assertEquals("identity:wife", out.get(MemoryTags.CLUSTER_KEY));
    }

    @Test
    void promotesExplicitPin() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.PINNED, "true");
        tags.put(MemoryTags.FACET, MemoryTags.FACET_PREFERENCE);

        Map<String, String> out = promoter.maybePromote("Small detail about tea", tags, 0.5);

        assertEquals("true", out.get(MemoryTags.CANONICAL),
                "pinning is a user signal and should always promote");
        assertNotNull(out.get(MemoryTags.CLUSTER_KEY));
    }

    @Test
    void promotesFoundationalFacetAtHighConfidence() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.FACET, MemoryTags.FACET_HEALTH);

        Map<String, String> out = promoter.maybePromote("allergic to peanuts", tags, 0.95);

        assertEquals("true", out.get(MemoryTags.CANONICAL));
    }

    @Test
    void skipsFoundationalFacetAtLowConfidence() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.FACET, MemoryTags.FACET_HEALTH);

        Map<String, String> out = promoter.maybePromote(
                "maybe allergic to peanuts", tags, 0.5);

        assertNull(out.get(MemoryTags.CANONICAL),
                "low-confidence health claim must not auto-promote");
    }

    @Test
    void promotesReinforcedBelief() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.FACET, MemoryTags.FACET_PREFERENCE);
        tags.put(MemoryTags.MENTION_COUNT, "3");

        Map<String, String> out = promoter.maybePromote(
                "Tea before coffee", tags, 0.7);

        assertEquals("true", out.get(MemoryTags.CANONICAL));
    }

    @Test
    void vetoedByHypothetical() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.FACET, MemoryTags.FACET_IDENTITY);
        tags.put(MemoryTags.HYPOTHETICAL, "true");
        tags.put(MemoryTags.PINNED, "true");  // even with pin

        Map<String, String> out = promoter.maybePromote(
                "My wife would be Priya", tags, 0.95);

        assertNull(out.get(MemoryTags.CANONICAL),
                "hypothetical items are never promoted, even with other signals");
    }

    @Test
    void idempotentOnAlreadyCanonical() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.CANONICAL, "true");
        tags.put(MemoryTags.CLUSTER_KEY, "identity:self");
        tags.put(MemoryTags.FACET, MemoryTags.FACET_IDENTITY);

        Map<String, String> out = promoter.maybePromote("anything at all", tags, 0.9);

        assertSame(tags, out, "already-canonical items are returned unchanged");
    }

    @Test
    void leavesWeakMaterialAlone() {
        Map<String, String> tags = new HashMap<>();
        tags.put(MemoryTags.FACET, MemoryTags.FACET_EVENT);

        Map<String, String> out = promoter.maybePromote("Went to the shops", tags, 0.6);

        assertNull(out.get(MemoryTags.CANONICAL));
        assertNull(out.get(MemoryTags.CLUSTER_KEY));
    }

    @Test
    void clusterKeyFromSlotNoun() {
        assertEquals("identity:wife",
                CanonicalPromoter.deriveClusterKey(
                        "my wife is Priya", MemoryTags.FACET_IDENTITY));
        assertEquals("location:sydney",
                CanonicalPromoter.deriveClusterKey(
                        "I live in Sydney", MemoryTags.FACET_LOCATION));
        assertEquals("identity:self",
                CanonicalPromoter.deriveClusterKey(
                        "I'm a cardiologist", MemoryTags.FACET_IDENTITY));
    }

    @Test
    void definitionalDetectorCoversCommonShapes() {
        assertTrue(CanonicalPromoter.matchesDefinitional("my wife's name is Priya"));
        assertTrue(CanonicalPromoter.matchesDefinitional("I'm allergic to peanuts"));
        assertTrue(CanonicalPromoter.matchesDefinitional("I am 36"));
        assertTrue(CanonicalPromoter.matchesDefinitional("I have a daughter"));
        assertTrue(CanonicalPromoter.matchesDefinitional("I live in Sydney"));
        assertTrue(CanonicalPromoter.matchesDefinitional("I always order Thai"));
        assertTrue(CanonicalPromoter.matchesDefinitional("I work at Anthropic"));

        assertFalse(CanonicalPromoter.matchesDefinitional("Went to the shops"));
        assertFalse(CanonicalPromoter.matchesDefinitional("thinking about learning tango"));
    }
}
