package ai.nizo.memory.session;

import ai.nizo.memory.store.SqliteMemoryStore;

import java.util.List;
import java.util.Set;

/**
 * Picks a small subset of session IDs from a larger manifest that are most
 * likely to contain the answer to a user's query. Used as a pre-filter for
 * recall on multi-session ("haystack") workloads: instead of ranking every
 * item across every session, we first narrow the search to 3–5 likely
 * sessions and then run the full hybrid recall pipeline inside those.
 *
 * <p>Implementations must:
 * <ul>
 *   <li>Be idempotent for the same query + manifest.</li>
 *   <li>Return an empty set to <em>abstain</em> when none of the sessions
 *       look relevant — the caller falls back to unfiltered recall.</li>
 *   <li>Handle errors gracefully — any runtime exception from the backing
 *       LLM should yield an empty set, never propagate.</li>
 *   <li>Use time-bounded calls — session selection is on the critical path
 *       of every turn once enabled.</li>
 * </ul>
 *
 * <p>The contract intentionally hides the picking strategy (LLM, embedding
 * nearest-neighbour over session previews, hand-coded heuristics) so the
 * recall pipeline can swap implementations without change.
 */
public interface SessionPicker {

    /**
     * Pick up to {@code topN} session IDs the caller should search within.
     *
     * @param query     the user query that will drive the final recall
     * @param manifest  one row per session (from {@link SqliteMemoryStore#sessionManifest})
     * @param topN      maximum number of IDs to return; implementations may
     *                  return fewer (including zero to abstain)
     * @return selected session IDs; never {@code null}. Empty = abstain.
     */
    Set<String> pick(String query, List<SqliteMemoryStore.SessionInfo> manifest, int topN);

    /**
     * No-op picker — never selects anything. Useful as a default when an
     * implementation isn't configured and the recall pipeline should fall
     * through to the legacy unfiltered path.
     */
    SessionPicker NO_OP = (query, manifest, topN) -> Set.of();
}
