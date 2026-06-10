package ai.nizo.memory.api.memory;

import java.time.Instant;
import java.util.Map;

/**
 * A single fact, utterance, observation or procedure stored by the memory tier.
 *
 * <p>{@code confidence} and {@code source} are the anti-hallucination scaffolding:
 * only facts with a source are fed back to the model as "ground truth"; lower-
 * confidence items are either re-verified or flagged UNKNOWN.
 *
 * <p>{@code userId} isolates memories per tenant/user. Every memory item belongs
 * to exactly one user; recall never crosses user boundaries.
 */
public record MemoryItem(
        String id,
        /** Owner of this memory. All queries are scoped to a single userId. */
        String userId,
        Tier tier,
        String content,
        float[] embedding,
        Map<String, String> tags,
        String source,
        double confidence,
        Instant createdAt,
        Instant lastAccessedAt,
        int accessCount,
        /** Token count of {@link #content} (pre-computed for budgeting). */
        int tokens
) {
    public enum Tier {
        /** Most recent turns; always in the prompt. */
        WORKING,
        /** Raw timestamped events (conversations, tool results). */
        EPISODIC,
        /** Consolidated facts / profile / preferences. */
        SEMANTIC,
        /** Learned skills, routines, heuristics. */
        PROCEDURAL
    }
}
