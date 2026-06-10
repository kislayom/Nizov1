package ai.nizo.memory.api.graph;

import java.time.Instant;
import java.util.Map;

/**
 * A node in the user's personal knowledge graph.
 *
 * <p>Nodes represent entities the system has extracted from conversation:
 * people, companies, preferences, goals, etc. Each node belongs to exactly
 * one user and carries a {@link #confidence} score that increases as the
 * entity is re-mentioned.
 *
 * <p>{@link #privacyLevel} gates visibility: nodes marked {@code "deleted"}
 * or {@code "redacted"} are excluded from recall and graph traversal.
 *
 * @param id              stable unique identifier
 * @param userId          owner of this node; queries never cross user boundaries
 * @param category        coarse type (e.g. "person", "company", "preference")
 * @param label           human-readable display name
 * @param properties      arbitrary key-value metadata
 * @param confidence      belief strength in [0, 1]; rises with repeated mentions
 * @param source          provenance tag (e.g. "conversation", "profile-import")
 * @param privacyLevel    access gate: "active", "archived", "redacted", "deleted"
 * @param mentionCount    how many times this entity has been referenced
 * @param firstSeenAt     timestamp of first extraction
 * @param lastConfirmedAt timestamp of most recent mention or verification
 */
public record Node(
        String id,
        String userId,
        String category,
        String label,
        Map<String, Object> properties,
        double confidence,
        String source,
        String privacyLevel,
        int mentionCount,
        Instant firstSeenAt,
        Instant lastConfirmedAt
) {

    /**
     * Returns {@code true} if this node is accessible for recall and traversal.
     * Nodes with privacy level {@code "deleted"} or {@code "redacted"} are excluded.
     */
    public boolean isAccessible() {
        return !"deleted".equals(privacyLevel) && !"redacted".equals(privacyLevel);
    }

    /**
     * Returns a new {@code Node} reflecting an additional mention: the
     * {@link #mentionCount} is incremented, {@link #lastConfirmedAt} is set to
     * now, and {@link #confidence} is nudged upward using exponential decay
     * toward 0.99.
     */
    public Node withMentionRecorded() {
        double updatedConfidence = Math.min(0.99, confidence + (1.0 - confidence) * 0.1);
        return new Node(
                id, userId, category, label, properties,
                updatedConfidence, source, privacyLevel,
                mentionCount + 1, firstSeenAt, Instant.now()
        );
    }
}
