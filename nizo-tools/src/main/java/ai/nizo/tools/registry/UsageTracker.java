package ai.nizo.tools.registry;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-memory tool/skill usage telemetry.
 *
 * <p>Per-name counters: total calls, total errors, last-used timestamp. Plus a ring
 * buffer of the most recent {@value #MAX_RECENT} invocations across all tools, each
 * tagged with the {@code chatId} that triggered it (so the UI can deep-link into the
 * conversation that used the tool).
 *
 * <p>Thread-safety: counters are {@link AtomicLong}s in a {@link ConcurrentHashMap};
 * the recent-deque is a {@link ConcurrentLinkedDeque}. All writes are O(1) and lock-free.
 *
 * <p>Intentionally NOT persisted — fresh on every server start. Fine for a single-user
 * agent. Adding sqlite persistence later is a 50-line change.
 *
 * <p>{@code currentChatId} is propagated via {@link ThreadLocal} — the agent loop sets it
 * before invoking a tool and clears it after. Tools don't have to know about it.
 */
public final class UsageTracker {

    public record Usage(
            String toolName,
            String chatId,
            long   timestamp,
            boolean ok,
            long   durationMs,
            String argsPreview
    ) {}

    /** Per-name aggregate snapshot for the UI. */
    public record Stats(
            String toolName,
            long   count,
            long   errors,
            long   lastUsed,
            long   avgDurationMs
    ) {}

    private static final int MAX_RECENT = 500;

    private final ConcurrentMap<String, AtomicLong> counts        = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, AtomicLong> errors        = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, AtomicLong> lastUsed      = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, AtomicLong> totalDuration = new ConcurrentHashMap<>();
    private final ConcurrentLinkedDeque<Usage> recent = new ConcurrentLinkedDeque<>();

    private static final ThreadLocal<String> CURRENT_CHAT = new ThreadLocal<>();

    /** Called by the agent loop right before dispatching a tool. */
    public static void setCurrentChatId(String chatId) { CURRENT_CHAT.set(chatId); }
    /** Called by the agent loop right after a tool returns (success or fail). */
    public static void clearCurrentChatId() { CURRENT_CHAT.remove(); }
    /** Read by {@link MeasuredTool} when recording. */
    public static String currentChatId() { return CURRENT_CHAT.get(); }

    /** Record one tool invocation. {@code argsPreview} is truncated by the caller. */
    public void record(String toolName, boolean ok, long durationMs, String argsPreview) {
        if (toolName == null || toolName.isBlank()) return;
        long now = System.currentTimeMillis();
        counts.computeIfAbsent(toolName, k -> new AtomicLong()).incrementAndGet();
        if (!ok) errors.computeIfAbsent(toolName, k -> new AtomicLong()).incrementAndGet();
        lastUsed.computeIfAbsent(toolName, k -> new AtomicLong()).set(now);
        totalDuration.computeIfAbsent(toolName, k -> new AtomicLong()).addAndGet(Math.max(0, durationMs));
        recent.addFirst(new Usage(toolName, currentChatId(), now, ok, durationMs, argsPreview));
        // Cheap trim — addFirst + pollLast keeps the deque bounded.
        while (recent.size() > MAX_RECENT) recent.pollLast();
    }

    /** Per-name aggregate stats. Map keyed by tool name. */
    public Map<String, Stats> snapshot() {
        Map<String, Stats> out = new LinkedHashMap<>();
        for (Map.Entry<String, AtomicLong> e : counts.entrySet()) {
            String name = e.getKey();
            long count  = e.getValue().get();
            long err    = errors.getOrDefault(name, new AtomicLong()).get();
            long last   = lastUsed.getOrDefault(name, new AtomicLong()).get();
            long totalD = totalDuration.getOrDefault(name, new AtomicLong()).get();
            long avg    = count > 0 ? totalD / count : 0;
            out.put(name, new Stats(name, count, err, last, avg));
        }
        return out;
    }

    /** Recent invocations, newest first. {@code limit} clamps how many to return. */
    public List<Usage> recent(int limit) {
        if (limit <= 0) return List.of();
        Collection<Usage> snap = recent;
        List<Usage> out = new ArrayList<>(Math.min(limit, snap.size()));
        for (Usage u : snap) {
            if (out.size() >= limit) break;
            out.add(u);
        }
        return out;
    }
}
