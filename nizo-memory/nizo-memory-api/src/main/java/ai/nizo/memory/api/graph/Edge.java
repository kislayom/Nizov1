package ai.nizo.memory.api.graph;

import java.time.Instant;
import java.util.Map;

/**
 * A directed, typed relationship between two {@link Node}s in the knowledge graph.
 *
 * <p>Edges are temporally scoped: {@link #validFrom} and {@link #validTo} bracket
 * the period during which the relationship holds. An edge that has been
 * superseded or retracted carries a non-null {@link #invalidatedAt} timestamp.
 *
 * @param id             stable unique identifier
 * @param userId         owner; edges never span user boundaries
 * @param sourceNodeId   originating node
 * @param targetNodeId   destination node
 * @param relationship   verb or label describing the edge (e.g. "works_at", "prefers")
 * @param properties     arbitrary key-value metadata
 * @param validFrom      start of the edge's validity window
 * @param validTo        end of validity; {@code null} means still active
 * @param confidence     belief strength in [0, 1]
 * @param source         provenance tag
 * @param invalidatedAt  set when the edge is explicitly retracted; {@code null} while live
 */
public record Edge(
        String id,
        String userId,
        String sourceNodeId,
        String targetNodeId,
        String relationship,
        Map<String, Object> properties,
        Instant validFrom,
        Instant validTo,
        double confidence,
        String source,
        Instant invalidatedAt
) {

    /**
     * Returns {@code true} if this edge is currently active: it has no
     * explicit end date and has not been invalidated.
     */
    public boolean isCurrent() {
        return validTo == null && invalidatedAt == null;
    }

    /**
     * Returns a copy of this edge marked as invalidated.
     *
     * <p>{@link #invalidatedAt} is set to the current instant. {@link #validTo}
     * is preserved if it was already set (the fact had a known expiry); if it
     * was {@code null}, it is set to {@code now} because an invalidated edge
     * with no prior expiry ceases to be valid at the moment of retraction.
     *
     * <p>Earlier versions of this method always overwrote {@link #validTo}
     * with {@code now} — that silently destroyed any forward-dated validTo
     * (e.g. "valid until 2026-12-31"). Tests: {@code EdgeInvalidationTest}.
     */
    public Edge invalidated() {
        Instant now = Instant.now();
        return new Edge(
                id, userId, sourceNodeId, targetNodeId,
                relationship, properties, validFrom,
                validTo != null ? validTo : now,
                confidence, source, now
        );
    }
}
