package ai.nizo.api.llm;

/**
 * Streaming chat consumer. The {@link LlmClient#streamChat} implementation invokes the
 * callbacks in order as deltas arrive on the SSE wire.
 *
 * <p>Method order for a typical turn:
 * <pre>
 *   onToken("Hello")
 *   onToken(" ")
 *   onToken("world")
 *   onComplete(response)         // total text already accumulated; toolCalls if any
 * </pre>
 *
 * For thinking-mode models (e.g. Qwen3.6 with {@code --reasoning-format deepseek}):
 * <pre>
 *   onThinking("Let me…")
 *   onThinking(" think.")
 *   onToken("Result is 42.")
 *   onComplete(response)
 * </pre>
 *
 * For tool-using turns:
 * <pre>
 *   onToolCallDelta(callId, toolName, argumentsChunk)*  // multiple chunks
 *   onComplete(response with tool_calls populated)
 * </pre>
 */
public interface ChatStreamHandler {

    /** A user-visible content delta. */
    default void onToken(String token) {}

    /** A reasoning_content / thinking delta (Qwen3.6 emits this when using deepseek format). */
    default void onThinking(String token) {}

    /**
     * A tool-call delta. {@code argumentsChunk} accumulates over multiple calls to form the full
     * JSON arguments string. The final form is also available in {@link ChatResponse#toolCalls()}
     * via {@link #onComplete}.
     */
    default void onToolCallDelta(String callId, String toolName, String argumentsChunk) {}

    /** Called once when the stream finishes successfully. */
    void onComplete(ChatResponse response);

    /** Called if the stream errors out. {@link #onComplete} will not be invoked after this. */
    void onError(Throwable t);
}
