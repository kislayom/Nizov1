package ai.nizo.api.condense;

import java.util.List;

/**
 * Tracks files the agent has read during the session so we can re-attach the most relevant
 * ones after a {@code condense} (otherwise the post-condense agent would have the summary but
 * lose access to the actual file bytes it had loaded).
 *
 * <p>Implementations must be thread-safe — multiple tools can write concurrently in virtual-thread
 * land. The interface lives in {@code nizo-api} so the file tools (which record reads) and the
 * condense engine (which consumes the cache) can both depend on it without a module cycle.
 */
public interface FileCache {

    /**
     * Record (or refresh) a file read. If the path already exists in cache its entry is replaced
     * — the most-recent read of a file is the one we want to re-inject.
     */
    void record(String path, String content, int tokensRead);

    /** Drop everything. Called immediately before a condense rebuilds the working set. */
    void clear();

    /**
     * Return up to {@code maxCount} most-recently-read files, with each entry truncated so its
     * content fits within {@code perFileTokenCap} tokens, and the total set under
     * {@code totalTokenBudget} tokens. Newest-first.
     */
    List<FileCacheEntry> topN(int totalTokenBudget, int perFileTokenCap, int maxCount);

    /** Read-only snapshot. */
    List<FileCacheEntry> snapshot();

    /** A no-op cache for callers that don't want re-injection wired. */
    FileCache NOOP = new FileCache() {
        @Override public void record(String path, String content, int tokensRead) {}
        @Override public void clear() {}
        @Override public List<FileCacheEntry> topN(int t, int p, int m) { return List.of(); }
        @Override public List<FileCacheEntry> snapshot() { return List.of(); }
    };
}
