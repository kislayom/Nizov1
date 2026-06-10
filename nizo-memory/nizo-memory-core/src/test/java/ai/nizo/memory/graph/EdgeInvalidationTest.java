package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * F7: {@link Edge#invalidated()} must not destroy a pre-existing
 * {@code validTo}. Regression coverage for the data-loss bug where an edge
 * with a future-dated validTo (e.g. "license valid until 2026-12-31")
 * lost that date the moment any code path invalidated the edge.
 */
class EdgeInvalidationTest {

    @Test
    void invalidated_preservesOriginalValidTo_whenSet() {
        Instant validFrom = Instant.now().minus(30, ChronoUnit.DAYS);
        Instant originalValidTo = Instant.now().plus(365, ChronoUnit.DAYS); // future
        Edge e = new Edge(
                "e1", "u1", "n1", "n2", "works_at",
                Map.of(), validFrom, originalValidTo, 0.9, "extraction", null);

        Edge out = e.invalidated();

        assertEquals(originalValidTo, out.validTo(),
                "validTo must be preserved on invalidation");
        assertNotNull(out.invalidatedAt(),
                "invalidatedAt must be set");
        assertFalse(out.isCurrent(),
                "edge must no longer be current");
    }

    @Test
    void invalidated_setsValidToNow_whenOriginallyNull() {
        Edge e = new Edge(
                "e2", "u1", "n1", "n2", "works_at",
                Map.of(), Instant.now(), null, 0.9, "extraction", null);

        Edge out = e.invalidated();

        assertNotNull(out.validTo(),
                "validTo must be populated when invalidating a currently-valid edge");
        assertNotNull(out.invalidatedAt());
        // validTo and invalidatedAt should be the same instant when no prior validTo
        assertEquals(out.validTo(), out.invalidatedAt(),
                "when no prior validTo, it takes the invalidation timestamp");
    }

    @Test
    void invalidated_preservesAllOtherFields() {
        Instant validFrom = Instant.now().minus(10, ChronoUnit.DAYS);
        Edge e = new Edge(
                "e3", "u1", "srcN", "tgtN", "prefers",
                Map.of("strength", "strong"), validFrom,
                null, 0.75, "user_stated", null);

        Edge out = e.invalidated();

        assertEquals(e.id(), out.id());
        assertEquals(e.userId(), out.userId());
        assertEquals(e.sourceNodeId(), out.sourceNodeId());
        assertEquals(e.targetNodeId(), out.targetNodeId());
        assertEquals(e.relationship(), out.relationship());
        assertEquals(e.properties(), out.properties());
        assertEquals(e.validFrom(), out.validFrom());
        assertEquals(e.confidence(), out.confidence());
        assertEquals(e.source(), out.source());
    }
}
