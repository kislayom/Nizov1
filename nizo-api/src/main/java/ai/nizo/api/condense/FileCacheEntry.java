package ai.nizo.api.condense;

import java.time.Instant;

/**
 * One file-read tracked during the session, suitable for re-injecting into history after a
 * condense so the agent doesn't lose touch with files it had already loaded.
 *
 * @param path         workspace-relative path
 * @param content      file contents at read time (may be a truncated view)
 * @param tokensRead   estimated token count of {@link #content}
 * @param readAt       when it was read
 */
public record FileCacheEntry(String path, String content, int tokensRead, Instant readAt) {
    public FileCacheEntry {
        if (path == null || path.isBlank()) throw new IllegalArgumentException("path required");
        if (content == null) content = "";
        if (readAt == null) readAt = Instant.now();
        if (tokensRead < 0) tokensRead = 0;
    }
}
