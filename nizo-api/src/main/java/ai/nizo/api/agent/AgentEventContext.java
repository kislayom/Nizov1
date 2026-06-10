package ai.nizo.api.agent;

/**
 * Thread-local pointer to the current {@link AgentEventSink}.
 *
 * <p>The agent loop sets this before invoking a tool so that nested executions (e.g. a sub-agent
 * skill running its own inner LLM loop with its own tool calls) can surface their activity through
 * the same SSE event stream the user sees. Without this, sub-agent tool calls are completely
 * invisible to the UI ("0 tool calls" for analysts that actually made dozens).
 *
 * <p>Always read via {@link #current()} which returns {@link AgentEventSink#NOOP} if nothing is
 * set — that way callers don't need null checks.
 */
public final class AgentEventContext {

    private static final ThreadLocal<AgentEventSink> CURRENT = new ThreadLocal<>();

    private AgentEventContext() {}

    public static void set(AgentEventSink sink) {
        if (sink == null) CURRENT.remove();
        else CURRENT.set(sink);
    }

    /** Never returns null — {@link AgentEventSink#NOOP} when no sink is bound. */
    public static AgentEventSink current() {
        AgentEventSink s = CURRENT.get();
        return s != null ? s : AgentEventSink.NOOP;
    }

    public static void clear() {
        CURRENT.remove();
    }
}
