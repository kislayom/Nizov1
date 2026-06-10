package ai.nizo.api.agent;

/**
 * Where the agent loop sends {@link AgentEvent}s. Implementations forward to
 * SSE clients, log files, telemetry, the TUI, etc.
 *
 * <p>Implementations MUST be thread-safe; the agent may invoke from worker threads.
 * Implementations SHOULD be non-blocking — slow sinks throttle the agent.
 */
@FunctionalInterface
public interface AgentEventSink {

    void emit(AgentEvent event);

    /** No-op sink. Use when streaming isn't wanted; results come via the return value. */
    AgentEventSink NOOP = e -> {};
}
