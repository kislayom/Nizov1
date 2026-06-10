package ai.nizo.agent.condense;

import ai.nizo.api.condense.FileCache;
import ai.nizo.api.condense.FileCacheEntry;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * In-process {@link FileCache}. Backed by a {@link LinkedHashMap} keyed by path: writing the
 * same path again moves it to the most-recent slot. {@link #topN} truncates each entry's
 * content to fit the per-file cap and stops adding entries once the global token budget is hit.
 *
 * <p>All public methods synchronize on {@code this} — contention is negligible (file reads are
 * tens-per-session at most) and the simpler model wins over a {@code ConcurrentLinkedDeque} dance.
 */
public final class InMemoryFileCache implements FileCache {

    private final LinkedHashMap<String, FileCacheEntry> entries = new LinkedHashMap<>(16, 0.75f, false);

    @Override
    public synchronized void record(String path, String content, int tokensRead) {
        if (path == null || path.isBlank()) return;
        // Move-to-end semantics: remove first so re-insertion bumps it to the tail.
        entries.remove(path);
        entries.put(path, new FileCacheEntry(path, content == null ? "" : content, tokensRead, Instant.now()));
    }

    @Override
    public synchronized void clear() { entries.clear(); }

    @Override
    public synchronized List<FileCacheEntry> snapshot() {
        List<FileCacheEntry> all = new ArrayList<>(entries.values());
        all.sort(Comparator.comparing(FileCacheEntry::readAt).reversed());
        return List.copyOf(all);
    }

    @Override
    public synchronized List<FileCacheEntry> topN(int totalTokenBudget, int perFileTokenCap, int maxCount) {
        if (maxCount <= 0 || totalTokenBudget <= 0) return List.of();

        // Newest first.
        List<Map.Entry<String, FileCacheEntry>> ordered = new ArrayList<>(entries.entrySet());
        ordered.sort((a, b) -> b.getValue().readAt().compareTo(a.getValue().readAt()));

        List<FileCacheEntry> out = new ArrayList<>();
        int spent = 0;
        for (Map.Entry<String, FileCacheEntry> e : ordered) {
            if (out.size() >= maxCount) break;
            FileCacheEntry orig = e.getValue();
            int tokens = orig.tokensRead();
            String content = orig.content();

            if (perFileTokenCap > 0 && tokens > perFileTokenCap) {
                // Approximate: trim the content's char length to the per-file cap.
                int charCap = (int) Math.floor(perFileTokenCap * TokenEstimator.CHARS_PER_TOKEN);
                if (charCap < content.length()) {
                    content = content.substring(0, charCap)
                            + "\n... [truncated for re-injection at " + perFileTokenCap + " tokens]";
                }
                tokens = perFileTokenCap;
            }

            if (spent + tokens > totalTokenBudget) {
                int remaining = Math.max(0, totalTokenBudget - spent);
                if (remaining < 256) break; // not worth a stub
                int charCap = (int) Math.floor(remaining * TokenEstimator.CHARS_PER_TOKEN);
                if (charCap < content.length()) {
                    content = content.substring(0, charCap)
                            + "\n... [truncated for re-injection budget]";
                }
                tokens = remaining;
            }

            out.add(new FileCacheEntry(orig.path(), content, tokens, orig.readAt()));
            spent += tokens;
        }
        return List.copyOf(out);
    }
}
