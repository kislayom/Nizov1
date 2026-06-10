package ai.nizo.memory.api.memory;

import java.util.Map;
import java.util.Set;

/**
 * Parameters for {@link MemoryService#recall(RecallRequest)}.
 *
 * @param userId        owner whose memories to search; {@code null} defaults to "default"
 * @param query         natural-language question or topic
 * @param tokenBudget   cap on total tokens of returned items; ranker prunes aggressively
 * @param tiers         which tiers to consult ({@code null} = all)
 * @param requiredTags  only return items carrying these tags
 * @param minConfidence drop items below this confidence (0..1)
 * @param sessionIds    restrict recall to items whose {@code session_id} tag is in this
 *                      set; {@code null} or empty = no session filter. Used by the
 *                      session-pre-filter path: an LLM picker chooses 3–5 likely-relevant
 *                      sessions from a large haystack, then this filter narrows RRF
 *                      recall to just those sessions.
 */
public record RecallRequest(
        String userId,
        String query,
        int tokenBudget,
        Set<MemoryItem.Tier> tiers,
        Map<String, String> requiredTags,
        double minConfidence,
        Set<String> sessionIds
) {
    /** Backwards-compat constructor — no session filter. */
    public RecallRequest(String userId, String query, int tokenBudget,
                         Set<MemoryItem.Tier> tiers,
                         Map<String, String> requiredTags,
                         double minConfidence) {
        this(userId, query, tokenBudget, tiers, requiredTags, minConfidence, null);
    }

    /** Convenience factory — defaults to the "default" user. */
    public static RecallRequest of(String query, int tokenBudget) {
        return new RecallRequest("default", query, tokenBudget, null, Map.of(), 0.0, null);
    }

    /** Convenience factory with explicit userId. */
    public static RecallRequest of(String userId, String query, int tokenBudget) {
        return new RecallRequest(userId, query, tokenBudget, null, Map.of(), 0.0, null);
    }

    /** Returns a copy of this request with the session filter set. */
    public RecallRequest withSessionFilter(Set<String> sessionIds) {
        return new RecallRequest(userId, query, tokenBudget, tiers, requiredTags, minConfidence,
                sessionIds);
    }
}
