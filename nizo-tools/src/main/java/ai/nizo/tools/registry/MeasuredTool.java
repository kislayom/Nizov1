package ai.nizo.tools.registry;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;

/**
 * Decorator that wraps any {@link Tool} and forwards each {@code execute()} call to
 * the underlying tool, measuring duration and recording success/failure into a shared
 * {@link UsageTracker}.
 *
 * <p>The current {@code chatId} (set by the agent loop into a ThreadLocal in
 * {@link UsageTracker}) is captured automatically — the wrapped tool itself doesn't
 * need to know it exists.
 *
 * <p>Used by {@code Bootstrap}: every tool added to the registry gets wrapped with
 * {@code new MeasuredTool(tool, tracker)} so the UI can show "X calls, last used N min ago"
 * for both built-in tools, MCP-bridged tools, and skill-tools.
 */
public final class MeasuredTool implements Tool {

    private final Tool delegate;
    private final UsageTracker tracker;

    public MeasuredTool(Tool delegate, UsageTracker tracker) {
        this.delegate = delegate;
        this.tracker  = tracker;
    }

    @Override public String name() { return delegate.name(); }
    @Override public String description() { return delegate.description(); }
    @Override public String parametersJsonSchema() { return delegate.parametersJsonSchema(); }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        long t0 = System.currentTimeMillis();
        String preview = previewArgs(argumentsJson);
        try {
            ToolResult r = delegate.execute(argumentsJson);
            tracker.record(name(), r != null && r.ok(), System.currentTimeMillis() - t0, preview);
            return r;
        } catch (Exception e) {
            tracker.record(name(), false, System.currentTimeMillis() - t0, preview);
            throw e;
        }
    }

    private static String previewArgs(String json) {
        if (json == null) return "";
        String s = json.replaceAll("\\s+", " ").trim();
        return s.length() <= 200 ? s : s.substring(0, 200) + "…";
    }
}
