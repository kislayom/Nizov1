package ai.nizo.api.agent;

/**
 * Events emitted by the agent loop while servicing one user turn.
 *
 * <p>UI/channels subscribe via an {@link AgentEventSink} to render token-by-token
 * output, tool-call activity, and thinking content separately from final reply.
 *
 * <p>Sealed so the consumer can pattern-match exhaustively. Adding a new event type
 * is intentionally a compile error at every consumer.
 */
public sealed interface AgentEvent {

    /** Iteration number this event belongs to (0-based). */
    int iteration();

    /** Streaming token of the model's user-visible content. */
    record TokenChunk(int iteration, String text) implements AgentEvent {}

    /** Streaming token of the model's hidden thinking (when reasoning_content is enabled). */
    record ThinkingChunk(int iteration, String text) implements AgentEvent {}

    /** A tool call has just been requested by the model and is about to run. */
    record ToolCallStart(int iteration, String callId, String toolName, String argumentsJson) implements AgentEvent {}

    /** A tool call has completed (or errored). */
    record ToolCallResult(int iteration, String callId, String toolName, boolean ok,
                          String content, long durationMs) implements AgentEvent {}

    /** The final assistant reply for this turn, after all tool iterations. */
    record FinalReply(int iteration, String text, int promptTokens, int completionTokens, String stopReason) implements AgentEvent {}

    /** A non-fatal error occurred (e.g. tool throw, model HTTP error). The loop may continue. */
    record Warning(int iteration, String message) implements AgentEvent {}
}
