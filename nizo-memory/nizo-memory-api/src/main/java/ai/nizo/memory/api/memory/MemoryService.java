package ai.nizo.memory.api.memory;

import java.util.List;
import java.util.Map;

/**
 * Central memory facade. The agent only ever talks to this interface; the
 * underlying tiers (episodic log, vector index, knowledge graph) are free to
 * evolve behind it.
 *
 * <p>Every operation is scoped to a {@code userId}. Two users' memories are
 * completely isolated: they share the same backing store but never leak data
 * across user boundaries during recall, consolidation, or stats.
 */
public interface MemoryService {

    /** Append a raw event to the episodic tier. Embedding & tagging happen async. */
    String remember(String userId, String content, Map<String, String> tags, String source);

    /**
     * Store a distilled fact in the semantic tier. The service deduplicates
     * against existing semantic items; contradictions are logged for later
     * reconciliation rather than silently overwritten.
     */
    String learnFact(String userId, String fact, String source, double confidence);

    /**
     * Tag-aware variant of {@link #learnFact}. Tags carry semantic metadata
     * (subject / sensitivity / mode / hypothetical / expires_at, see
     * {@link MemoryTags}) that the recall pipeline reads. Default impl
     * delegates to {@link #learnFact} and discards tags — concrete
     * implementations should preserve them on the stored item.
     */
    default String learnFact(String userId, String fact, String source,
                              double confidence, Map<String, String> tags) {
        return learnFact(userId, fact, source, confidence);
    }

    /** Retrieve a ranked, token-budgeted set of items relevant to a query. */
    List<MemoryItem> recall(RecallRequest request);

    /** Force a consolidation pass: summarise episodic → semantic, evict WORKING. */
    void consolidate(String userId);

    /**
     * Demote the confidence of memory items matching a query. Used when a
     * newer fact supersedes older ones (e.g., job change, preference switch).
     * Items matching the query text have their confidence lowered so the
     * newer fact outranks them.
     *
     * @return number of items demoted
     */
    default int demoteContradicted(String userId, String query, double newConfidence) {
        return 0; // no-op default for clients that don't support it
    }

    /**
     * Store a procedural heuristic — a learned rule about how to interpret
     * content. Used by the extraction pipeline to learn user-specific or
     * world-knowledge guardrails ("WFH = working from home", "iPhone user
     * is a phone preference, not a job").
     *
     * <p>Stored in the PROCEDURAL tier and pulled by ExtractionPipeline as
     * additional context for every extraction.
     *
     * @return the new item's id
     */
    default String learnProcedural(String userId, String rule, String source, double confidence) {
        // Default impl falls back to learnFact; concrete implementations
        // should store as PROCEDURAL tier specifically.
        return learnFact(userId, rule, source, confidence);
    }

    /** Current counts per tier (observability). */
    Map<MemoryItem.Tier, Long> stats(String userId);

    /**
     * Retrieve top-K PROCEDURAL items most semantically relevant to the query,
     * BYPASSING the user-facing relevance floor and FTS / graph channels.
     * Used by the extraction pipeline to inject only relevant world-knowledge
     * heuristics into the prompt — instead of dumping every PROCEDURAL item.
     *
     * <p>This separation matters because PROCEDURAL items are meta-knowledge
     * (extraction guardrails) and need a wider net than user-facing facts: an
     * extraction prompt asking about "I just bought BTC" should pull the
     * crypto-related heuristics even if their cosine similarity is moderate.
     *
     * <p>Default returns empty — concrete implementations override.
     *
     * @param userId  scope (procedural items are per-user)
     * @param query   the message being extracted (used as semantic anchor)
     * @param topK    max heuristics to return (8-20 is the useful range)
     */
    default List<MemoryItem> recallProcedural(String userId, String query, int topK) {
        return List.of();
    }

    // ===== Customer-facing controls ==================================

    /**
     * Return everything stored for this user, newest first. Backs the
     * customer-facing "show me what you remember about me" experience.
     *
     * <p>Default returns empty. Concrete implementations override.
     */
    default List<MemoryItem> inspect(String userId, int limit) {
        return List.of();
    }

    /**
     * Surgically forget every fact whose content mentions {@code topic}.
     * Backs "forget about Mike" — finds matching facts (substring +
     * graph-aware lookups) and deletes them.
     *
     * @return number of items deleted
     */
    default int forgetAbout(String userId, String topic) {
        return 0;
    }

    /**
     * Pin (or unpin) a fact. Pinned facts are always recalled and get
     * a strong scoring boost.
     *
     * @param pinned  true to pin, false to unpin
     * @return true if the fact existed and was updated
     */
    default boolean pin(String userId, String factId, boolean pinned, String reason) {
        return false;
    }

    /**
     * Mark a stored preference as still-true today. Updates the
     * {@code last_reconfirmed} tag so the fact escapes age-based decay.
     *
     * @return true if the fact existed and was updated
     */
    default boolean reconfirm(String userId, String factId) {
        return false;
    }

    /**
     * Batch-import facts at user onboarding time. Each entry becomes a
     * SEMANTIC fact tagged with the supplied tags (subject / sensitivity /
     * mode / etc). Source is tagged "imported".
     *
     * @return number of facts stored
     */
    default int importFacts(String userId, List<ImportedFact> facts) {
        if (facts == null) return 0;
        int n = 0;
        for (ImportedFact f : facts) {
            if (f == null || f.content() == null || f.content().isBlank()) continue;
            learnFact(userId, f.content(), "imported",
                    f.confidence() == null ? 0.85 : f.confidence());
            n++;
        }
        return n;
    }

    /** Single fact for {@link #importFacts}. */
    record ImportedFact(String content, Map<String, String> tags, Double confidence) {}

    /**
     * Delete every fact stored for this user. GDPR forget-user. Concrete
     * implementations must cascade through every storage layer (memories,
     * graph nodes/edges, vector index, FTS).
     *
     * @return number of items deleted (memories table only — implementations
     *   should log cascade counts)
     */
    default int forgetUser(String userId) {
        return 0;
    }

    // ===== Active memory (pre-reply proactive surface) ================

    /**
     * Pre-reply memory surface — the calling agent sends the user's latest
     * message; we return a bounded set of facts the agent should know BEFORE
     * generating its reply. The agent calls this every turn unconditionally;
     * we decide whether to surface anything.
     *
     * <p>Properties:
     * <ul>
     *   <li><b>Bounded</b> — capped by {@code maxItems} and
     *       {@code maxSummaryChars}. Never returns a dump.</li>
     *   <li><b>Abstains honestly</b> — {@code surfaced=false} with a
     *       {@code skipReason} for empty / command-only / below-threshold
     *       messages. Never returns noise to fill the budget.</li>
     *   <li><b>No mandatory LLM call</b> — defaults to using the existing
     *       hybrid recall pipeline. Implementations MAY add an optional
     *       LLM summariser for the {@code summary} field.</li>
     *   <li><b>Mode-tuned</b> — {@code mode} adjusts recall thresholds:
     *       {@code balanced} (default), {@code strict} / {@code precision-heavy}
     *       (higher floor, fewer items), {@code recall-heavy} (lower floor,
     *       more items), {@code preference-only} (only SEMANTIC preference
     *       facts).</li>
     * </ul>
     *
     * <p>Analogous to OpenClaw's Active Memory plugin — a "bounded pre-reply
     * memory sub-agent" that runs once per turn.
     */
    default SurfaceResult surface(SurfaceRequest req) {
        return SurfaceResult.skipped("not-implemented");
    }

    /** Input for {@link #surface(SurfaceRequest)}. */
    record SurfaceRequest(
            String userId,
            String message,
            String mode,           // balanced | strict | recall-heavy | precision-heavy | preference-only
            int maxItems,
            int maxSummaryChars,
            List<ConversationTurn> recentTurns  // optional context
    ) {}

    record ConversationTurn(String role, String content) {}

    /** Result of {@link #surface(SurfaceRequest)}. */
    record SurfaceResult(
            boolean surfaced,
            String summary,
            List<MemoryItem> items,
            String skipReason,
            String mode
    ) {
        public static SurfaceResult skipped(String reason) {
            return new SurfaceResult(false, "", List.of(), reason, "balanced");
        }
    }

    // ===== Canonical index (Phase C) =================================

    /**
     * Compact "table of contents" of the user's canonical facts — the stable,
     * foundational statements the memory pipeline has promoted to
     * always-surface status. Intended for inclusion at the top of every
     * assistant turn's system prompt so the model can answer simple queries
     * ("what's my wife's name", "am I allergic to anything") without
     * triggering the full recall pipeline.
     *
     * <p>Entries are grouped by {@link MemoryTags#CLUSTER_KEY} so conflicting
     * or superseded claims for the same attribute collapse to the most
     * recent value.
     *
     * <p>Default returns an empty list — concrete implementations override.
     *
     * @param userId     owner scope
     * @param maxEntries hard cap on entries returned
     */
    default List<IndexEntry> canonicalIndex(String userId, int maxEntries) {
        return List.of();
    }

    /**
     * One row of the {@link #canonicalIndex}. Designed to render as a single
     * line in a Markdown "## What I know about you" block.
     *
     * @param clusterKey      namespaced {@code facet:slot} identifier
     * @param fact            the fact text (≤ ~140 chars for prompt readability)
     * @param facet           semantic facet (identity/health/preference/…)
     * @param lastReconfirmed last time this fact was explicitly reconfirmed,
     *                        {@code null} if it has never been reconfirmed
     *                        since first ingest
     */
    record IndexEntry(
            String clusterKey,
            String fact,
            String facet,
            java.time.Instant lastReconfirmed
    ) {}
}
